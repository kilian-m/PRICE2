"""A single-context GPU deconvolution broker for PRICE2.

Motivation: each CPU worker that touches
the GPU pays a ~272 MiB CUDA/torch context that MPS does not share, so N workers
cost N contexts. The broker inverts that: **one** long-lived GPU process holds
**one** context and does all the group-LASSO MU deconvolution; the many
short-lived CPU workers (which stay on the CPU for read loading / EG building)
ship it their sparse systems over **zero-copy shared memory** and get back the
optimized activities. This decouples CPU worker count (e.g. 80) from GPU
concurrency (a handful of CUDA streams inside the broker), so device memory no
longer scales with the pool size.

Protocol per job (a small :class:`Job` over a plain mp.Queue; big arrays via
multiprocessing.shared_memory, never pickled):
  worker: put X, Xᵀ (CSR arrays), y (and an optional warm-start w0) into shm,
          already in the broker's dtype; pre-allocate a result-w shm, a 1-byte
          status shm and an error-message shm; listen on a private abstract
          AF_UNIX address; enqueue the job; block in accept() until the broker
          connects; read the status byte and w (or the error message).
  broker: N worker threads, each with its own torch.cuda.Stream, pull jobs, run
          the IRLS-Huber MU loop on the GPU, write w, set the status (done or
          failed), then connect to the job's address to wake the worker.

The wake-up is a socket rather than a polled flag so a worker queued behind a
busy GPU blocks in the kernel instead of waking thousands of times a second.
The address lives in Linux's abstract namespace, so it needs no filesystem
entry, no cleanup, and — unlike a semaphore or an fd — it is just a string and
travels through the ordinary job.

The update rule, including the overflow floor on ``delta``, is the weighted
Richardson-Lucy + group-LASSO step of ``price2.mu_solver``; note that this
copy checks convergence only every :data:`CHECK_EVERY` inner iterations (a
``.item()`` is a GPU sync), so results are close to but not identical with
the in-worker CPU/GPU path.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any, NamedTuple

import numpy as np
from scipy.sparse import csr_matrix

from price2.mu_solver import OVERFLOW_CAP, silence_sparse_beta_warning

logger = logging.getLogger(__name__)

#: Inner iterations between two convergence checks: reading the relative
#: change (``.item()``) synchronises the stream, so it is not done every step.
CHECK_EVERY = 25
#: How long :meth:`GpuBroker.start` waits for every broker process to hold
#: its CUDA context.
START_TIMEOUT_SECONDS = 90.0
#: Floor under ``delta`` in the Huber weights and under ``‖w‖`` in the
#: stopping rules — the floors of :func:`price2.likelihood.huber_weights` and
#: :func:`price2.mu_solver.relative_change`.
_FLOOR = 1e-14
#: How long the broker waits for a worker to accept its wake-up connect.
_WAKE_CONNECT_SECONDS = 5.0
#: How often an idle stream-thread looks at the stop flag.
_QUEUE_POLL_SECONDS = 0.2
#: Grace period for the broker processes to exit at :meth:`GpuBroker.stop`.
_STOP_JOIN_SECONDS = 30.0
#: Bytes reserved for the failure message a broker hands back to the worker.
_ERROR_BYTES = 512

#: Values of a job's status byte.
_PENDING, _DONE, _FAILED = 0, 1, 2


# ------------------------------------------------------------------ #
# job completion signalling (abstract AF_UNIX socket, no fd passing)   #
# ------------------------------------------------------------------ #
def _done_listener(timeout: float) -> tuple[socket.socket, str]:
    """Bind a private abstract-namespace socket the broker can connect to.

    Returns ``(server_socket, address)``.  Binding and listening happen before
    the job is enqueued, so a broker that finishes immediately still lands in
    the backlog rather than racing the worker's ``accept()``.
    """
    address = f"\0price2-mu-{os.getpid()}-{uuid.uuid4().hex}"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(address)
        server.listen(1)
        server.settimeout(timeout)
    except BaseException:
        server.close()
        raise
    return server, address


def _signal_done(address: str) -> None:
    """Wake the worker waiting on *address*; ignore a worker that gave up."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(_WAKE_CONNECT_SECONDS)
            client.connect(address)
    except OSError:
        pass  # worker timed out or died; its shm is unlinked either way


# ------------------------------------------------------------------ #
# shared-memory arrays                                                 #
# ------------------------------------------------------------------ #
class ArrayMeta(NamedTuple):
    """How to attach to one array in shared memory."""

    name: str
    shape: tuple[int, ...]
    dtype: str


class CsrMeta(NamedTuple):
    """The three arrays of a CSR matrix in shared memory, and its shape."""

    data: ArrayMeta
    indices: ArrayMeta
    indptr: ArrayMeta
    shape: tuple[int, int]


class _SharedBlocks:
    """The shared-memory blocks one side of a job holds.

    The worker creates the blocks (``owner=True``) and unlinks them when the
    job is over; the broker only attaches to them.  Either side closes its
    mappings in :meth:`release`.

    The single owning ``unlink()`` is the one-and-only resource-tracker
    unregister: attaching must not let the tracker manage the block, or it
    may unlink shm it does not own (bpo-38119).  ``track=False`` (py3.13+)
    keeps the tracker out entirely; on older Pythons attaching re-registers
    the name, but the tracker cache is an idempotent set, so the owner's
    single unlink still balances it — do NOT unregister anywhere else, a
    second remove KeyErrors at teardown.
    """

    def __init__(self, owner: bool) -> None:
        self.owner = owner
        self._blocks: list[shared_memory.SharedMemory] = []

    def alloc(self, shape: tuple[int, ...], dtype: Any) -> tuple[np.ndarray, ArrayMeta]:
        """Create a zero-filled block; return the view and how to attach to it."""
        dtype = np.dtype(dtype)
        size = max(int(np.prod(shape)) * dtype.itemsize, 1)
        try:
            shm = shared_memory.SharedMemory(create=True, size=size, track=False)
        except TypeError:
            shm = shared_memory.SharedMemory(create=True, size=size)
        self._blocks.append(shm)
        view = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        view[...] = 0
        return view, ArrayMeta(shm.name, tuple(shape), str(dtype))

    def put(self, arr: np.ndarray) -> ArrayMeta:
        """Copy *arr* into a fresh block."""
        arr = np.ascontiguousarray(arr)
        view, meta = self.alloc(arr.shape, arr.dtype)
        view[...] = arr
        return meta

    def put_csr(self, A: csr_matrix, dtype: np.dtype) -> CsrMeta:
        """Copy the arrays of *A* into fresh blocks, its values as *dtype*."""
        return CsrMeta(
            self.put(A.data.astype(dtype, copy=False)),
            self.put(A.indices.astype(np.int64, copy=False)),
            self.put(A.indptr.astype(np.int64, copy=False)),
            (int(A.shape[0]), int(A.shape[1])),
        )

    def get(self, meta: ArrayMeta) -> np.ndarray:
        """Attach to an existing block."""
        try:
            shm = shared_memory.SharedMemory(name=meta.name, track=False)
        except TypeError:
            shm = shared_memory.SharedMemory(name=meta.name)
        self._blocks.append(shm)
        return np.ndarray(meta.shape, dtype=np.dtype(meta.dtype), buffer=shm.buf)

    def get_csr(self, meta: CsrMeta) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Attach to the ``(data, indices, indptr)`` arrays of a CSR matrix."""
        return self.get(meta.data), self.get(meta.indices), self.get(meta.indptr)

    def release(self) -> None:
        """Close every mapping; the owner also unlinks the blocks."""
        for shm in self._blocks:
            try:
                shm.close()
            except OSError:
                pass
            if self.owner:
                try:
                    shm.unlink()
                except OSError:
                    pass
        self._blocks.clear()


# ------------------------------------------------------------------ #
# what travels over the queue                                          #
# ------------------------------------------------------------------ #
@dataclass
class Params:
    """The solver settings of one job (see :class:`price2.config.Config`)."""

    num_rgrs: int
    num_runs: int
    lam: float
    pseudo_min: float
    huber_c: float
    max_outer: int
    huber_tol: float
    mu_inner_max_iter: int = 3000
    mu_inner_tol: float = 1e-5
    #: Negative-binomial dispersion θ, or ``None`` for the Poisson model.
    theta: float | None = None


@dataclass
class Job:
    """One deconvolution request: where its arrays are and what to solve.

    Attributes
    ----------
    req_id : int
        The worker's running job number, for log messages.
    X, XT : CsrMeta
        The design matrix and its transpose, values in the broker's dtype.
    y : ArrayMeta
        The counts, in the broker's dtype.
    w0 : ArrayMeta or None
        A warm start; ``None`` starts from all ones.
    result : ArrayMeta
        Where the broker writes the activities (float64).
    status : ArrayMeta
        One byte: ``_PENDING`` until the broker sets ``_DONE`` or ``_FAILED``.
    error : ArrayMeta
        Where the broker writes the failure message, NUL-padded UTF-8.
    done : str
        The worker's wake-up address.
    params : Params
    """

    req_id: int
    X: CsrMeta
    XT: CsrMeta
    y: ArrayMeta
    w0: ArrayMeta | None
    result: ArrayMeta
    status: ArrayMeta
    error: ArrayMeta
    done: str
    params: Params


def _write_error(buffer: np.ndarray, message: str) -> None:
    encoded = message.encode("utf-8", "replace")[: buffer.size]
    buffer[...] = 0
    buffer[: len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)


def _read_error(buffer: np.ndarray) -> str:
    return buffer.tobytes().rstrip(b"\0").decode("utf-8", "replace")


# ------------------------------------------------------------------ #
# GPU numerical core (runs inside a broker thread, on its own stream)  #
# ------------------------------------------------------------------ #
def _csr_tensor(torch, arrays: tuple, shape: tuple[int, int], dtype):
    data, indices, indptr = arrays
    return torch.sparse_csr_tensor(
        torch.from_numpy(indptr).cuda(),
        torch.from_numpy(indices).cuda(),
        torch.from_numpy(data).to(dtype).cuda(),
        size=shape,
        dtype=dtype,
        device="cuda",
    )


def _gpu_deconvolve(torch, job: Job, dtype, blocks: _SharedBlocks) -> int:
    """Full IRLS-Huber group-LASSO deconvolution of one locus on the GPU.

    Returns the number of outer iterations run.
    """
    tiny = 1e-30 if dtype == torch.float32 else 1e-300
    cap = OVERFLOW_CAP["float32" if dtype == torch.float32 else "float64"]
    p = job.params
    n = job.X.shape[1]
    nr, ns = p.num_rgrs, p.num_runs
    lam, pmin, c = p.lam, p.pseudo_min, p.huber_c
    theta = p.theta
    poisson = theta is None

    Xc = _csr_tensor(torch, blocks.get_csr(job.X), job.X.shape, dtype)
    XcT = _csr_tensor(torch, blocks.get_csr(job.XT), job.XT.shape, dtype)
    yt = torch.from_numpy(blocks.get(job.y)).to(dtype).cuda()
    # Warm-start from the caller's w0 when provided (the EM light M-step,
    # which runs a single non-converged outer iteration, relies on it);
    # otherwise fall back to an all-ones cold start.
    if job.w0 is not None:
        w = torch.from_numpy(blocks.get(job.w0)).to(dtype).cuda()
    else:
        w = torch.ones(n, dtype=dtype, device="cuda")

    outer = 0
    for outer in range(p.max_outer):
        delta = torch.mv(Xc, w).clamp_min(_FLOOR)
        # Pearson residual variance: δ (Poisson) or δ + δ²/θ (negative binomial)
        var = delta if poisson else delta + delta * delta / theta
        pr = (yt - delta) / var.sqrt()
        ar = pr.abs()
        omega = torch.where(ar <= c, torch.ones_like(ar), c / ar.clamp_min(_FLOOR))
        omega_y = omega * yt
        # The per-element delta floor of mu_solver.mu_inner_cpu: it keeps
        # omega_y/delta from overflowing the X^T mat-vec to +Inf, which
        # would cascade into NaN through the group-norm penalty.
        delta_floor = (omega_y * (1.0 / cap)).clamp_min(tiny)
        # Poisson denominator data term X^T ω is constant across inner iterations.
        if poisson:
            Xt_omega = torch.mv(XcT, omega)
        w_outer0 = w
        for it in range(p.mu_inner_max_iter):
            delta = torch.maximum(torch.mv(Xc, w), delta_floor)
            num = torch.mv(XcT, omega_y / delta)
            if poisson:
                data_den = Xt_omega
            else:
                # NB denominator data term X^T(ω (y+θ)/(θ+δ)) depends on δ.
                data_den = torch.mv(XcT, omega * (yt + theta) / (theta + delta))
            if lam > 0.0:
                W = w.view(nr, ns)
                norms = (W * W).sum(1).sqrt()
                pen = (lam * (W / norms.clamp_min(tiny).view(-1, 1))).reshape(-1)
                den = (data_den + pen).clamp_min(tiny)
            else:
                den = data_den.clamp_min(tiny)
            w_new = (w * num / den).clamp_min(pmin)
            if (it + 1) % CHECK_EVERY == 0:
                rel = ((w_new - w).norm() / w.norm().clamp_min(_FLOOR)).item()
                w = w_new
                if rel < p.mu_inner_tol:
                    break
            else:
                w = w_new
        rel_o = ((w - w_outer0).norm() / w_outer0.norm().clamp_min(_FLOOR)).item()
        if rel_o < p.huber_tol:
            break

    blocks.get(job.result)[...] = w.double().cpu().numpy()
    return outer + 1


def _serve(torch, stream, dtype, job: Job) -> None:
    """Run one job on *stream*, record its outcome and wake the worker."""
    blocks = _SharedBlocks(owner=False)
    try:
        try:
            status = blocks.get(job.status)
            with torch.cuda.stream(stream):
                _gpu_deconvolve(torch, job, dtype, blocks)
            status[0] = _DONE
        except Exception as exc:  # noqa: BLE001 — reported to the worker
            message = f"{type(exc).__name__}: {exc}"
            logger.error("job %d failed: %s", job.req_id, message)
            try:
                _write_error(blocks.get(job.error), message)
                blocks.get(job.status)[0] = _FAILED
            except OSError:
                pass  # the worker gave up and unlinked its blocks
    finally:
        blocks.release()
        # Always wake the worker, including on failure — otherwise it
        # blocks in accept() until its (long) timeout.
        _signal_done(job.done)


# ------------------------------------------------------------------ #
# Broker server process                                               #
# ------------------------------------------------------------------ #
def _broker_main(req_q, stop_evt, ready_val, n_streams: int, dtype_str: str) -> None:
    # A forkserver child carries no log handlers; report to the inherited
    # stderr, which is where the parent's own log goes.
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [gpu-broker %(process)d] %(levelname)s %(message)s",
        )
    # Before any stream-thread builds its first CSR tensor, and in the broker
    # process only: the parent and the CPU workers keep their own filters.
    silence_sparse_beta_warning()
    import torch

    dtype = getattr(torch, dtype_str)
    torch.zeros(1, device="cuda")  # create THIS process's context (one per broker)
    with ready_val.get_lock():
        ready_val.value += 1

    def worker_thread() -> None:
        stream = torch.cuda.Stream()
        while True:
            try:
                job = req_q.get(timeout=_QUEUE_POLL_SECONDS)
            except queue.Empty:
                if stop_evt.is_set():
                    return
                continue
            except (EOFError, OSError):
                return  # the manager is gone: the parent has shut down
            if job is None:
                return
            _serve(torch, stream, dtype, job)

    threads = [
        threading.Thread(target=worker_thread, daemon=True) for _ in range(n_streams)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


# ------------------------------------------------------------------ #
# Broker handle (parent process) + client (worker process)            #
# ------------------------------------------------------------------ #
class GpuBroker:
    """A pool of ``n_procs`` broker processes sharing one request queue.

    Each process holds ONE CUDA context and runs ``n_streams`` stream-threads.
    Using separate PROCESSES (not just threads) is what matters: a single Python
    process is GIL-bound — its stream-threads serialise on kernel dispatch and
    ``.item()`` syncs, so one process only drives the GPU to ~a third and cannot
    feed an 80-worker pool (observed on the full run). ``n_procs`` independent
    interpreters have independent GILs, so they actually saturate the GPU, while
    the context count stays at ``n_procs`` (bounded), not one-per-worker.

    Create in the parent; pass ``.req_q`` to the workers.
    """

    def __init__(
        self, n_procs: int = 1, n_streams: int = 1, dtype_str: str = "float32"
    ) -> None:
        if dtype_str not in OVERFLOW_CAP:
            raise ValueError(f"mu_dtype must be float32 or float64, got {dtype_str!r}")
        ctx = mp.get_context("forkserver")
        # Manager queue: pickles cleanly to every worker; carries only small job
        # metadata (shm names + params). Multiple broker processes consume it.
        self._mgr = ctx.Manager()
        self.req_q = self._mgr.Queue()
        self._stop = ctx.Event()
        self._ready = ctx.Value("i", 0)  # count of processes that reached ready
        self.n_procs = n_procs
        self.n_streams = n_streams
        self._procs = [
            ctx.Process(
                target=_broker_main,
                args=(self.req_q, self._stop, self._ready, n_streams, dtype_str),
                daemon=True,
            )
            for _ in range(n_procs)
        ]

    def start(self, timeout: float = START_TIMEOUT_SECONDS) -> None:
        """Start the processes and wait until every one holds its context."""
        for p in self._procs:
            p.start()
        t0 = time.monotonic()
        while True:
            with self._ready.get_lock():
                ready = self._ready.value
            if ready >= self.n_procs:
                return
            dead = [p for p in self._procs if not p.is_alive()]
            if dead:
                raise RuntimeError(
                    f"{len(dead)} broker process(es) died during startup "
                    "(torch/CUDA unavailable?)"
                )
            if time.monotonic() - t0 > timeout:
                raise RuntimeError("GPU broker pool failed to become ready")
            time.sleep(_QUEUE_POLL_SECONDS)

    def stop(self) -> None:
        """Wake every stream-thread with a sentinel and end the processes."""
        self._stop.set()
        try:
            # One sentinel per stream-thread; a thread that already left on
            # the stop flag leaves its sentinel behind, which is harmless.
            for _ in range(self.n_procs * self.n_streams):
                self.req_q.put(None)
        except (EOFError, OSError):
            pass  # the manager is already gone
        for p in self._procs:
            p.join(timeout=_STOP_JOIN_SECONDS)
        for p in self._procs:
            if p.is_alive():
                p.terminate()
        try:
            self._mgr.shutdown()
        except (EOFError, OSError):
            pass


class BrokerClient:
    """Used inside a CPU worker: ship a locus system to the broker, get back w.

    Parameters
    ----------
    req_q
        The broker pool's request queue (``GpuBroker.req_q``).
    dtype_str : str
        The broker's ``mu_dtype``; the arrays are shipped in it, so a
        ``float32`` broker is not sent float64 values it would only narrow.
    """

    def __init__(self, req_q, dtype_str: str = "float32") -> None:
        if dtype_str not in OVERFLOW_CAP:
            raise ValueError(f"mu_dtype must be float32 or float64, got {dtype_str!r}")
        self.req_q = req_q
        self.dtype = np.dtype(dtype_str)
        self._counter = 0

    def solve(
        self,
        X: csr_matrix,
        XT: csr_matrix,
        y: np.ndarray,
        params: Params,
        w0: np.ndarray | None = None,
        *,
        timeout: float,
    ) -> np.ndarray:
        """Deconvolve ``(X, y)`` on the broker; block until it answers.

        Parameters
        ----------
        X, XT : csr_matrix
            The design matrix and its transpose in CSR layout.
        y : np.ndarray
            The counts.
        params : Params
            The solver settings.
        w0 : np.ndarray, optional
            A warm start; ``None`` starts from all ones.
        timeout : float
            Seconds to wait for the answer — the locus's own budget, so a
            worker never outlives the timeout the pool enforces on it.

        Returns
        -------
        np.ndarray
            The activities, float64.

        Raises
        ------
        TimeoutError
            When the broker did not answer within *timeout*.
        RuntimeError
            When the broker reported a failure; carries its message.
        """
        self._counter += 1
        n = params.num_rgrs * params.num_runs
        blocks = _SharedBlocks(owner=True)
        server = None
        try:
            result, result_meta = blocks.alloc((n,), np.float64)
            status, status_meta = blocks.alloc((1,), np.uint8)
            error, error_meta = blocks.alloc((_ERROR_BYTES,), np.uint8)
            # Listen before enqueuing, so the broker cannot signal into a
            # socket that does not exist yet.
            server, done_address = _done_listener(timeout)
            job = Job(
                req_id=self._counter,
                X=blocks.put_csr(X, self.dtype),
                XT=blocks.put_csr(XT, self.dtype),
                y=blocks.put(np.asarray(y, dtype=self.dtype)),
                w0=None if w0 is None else blocks.put(np.asarray(w0, dtype=self.dtype)),
                result=result_meta,
                status=status_meta,
                error=error_meta,
                done=done_address,
                params=params,
            )
            self.req_q.put(job)

            try:
                conn, _ = server.accept()  # blocks in the kernel, no wake-ups
                conn.close()
            except (socket.timeout, TimeoutError):
                raise TimeoutError(
                    f"GPU broker did not answer job {job.req_id} within {timeout:.0f} s"
                ) from None

            if status[0] == _FAILED:
                raise RuntimeError(
                    f"GPU broker failed on job {job.req_id}: {_read_error(error)}"
                )
            if status[0] != _DONE:
                raise RuntimeError(
                    f"GPU broker signalled job {job.req_id} without writing a status"
                )
            return result.copy()
        finally:
            if server is not None:
                server.close()
            blocks.release()
