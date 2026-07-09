"""A single-context GPU deconvolution broker for PRICE2.

Motivation (see ../mps_experiment/MPS_REPORT.md): each CPU worker that touches
the GPU pays a ~272 MiB CUDA/torch context that MPS does not share, so N workers
cost N contexts. The broker inverts that: **one** long-lived GPU process holds
**one** context and does all the group-LASSO MU deconvolution; the many
short-lived CPU workers (which stay on the CPU for read loading / EG building)
ship it their sparse systems over **zero-copy shared memory** and get back the
optimized activities. This decouples CPU worker count (e.g. 80) from GPU
concurrency (a handful of CUDA streams inside the broker), so device memory no
longer scales with the pool size.

Protocol per job (small metadata over a plain mp.Queue; big arrays via
multiprocessing.shared_memory, never pickled):
  worker: put X, Xᵀ (CSR arrays), y (and an optional warm-start w0) into shm;
          pre-allocate a result-w shm and a 1-byte status shm; listen on a
          private abstract AF_UNIX address; enqueue the job; block in accept()
          until the broker connects; read the status byte and w.
  broker: N worker threads, each with its own torch.cuda.Stream, pull jobs, run
          the IRLS-Huber MU loop on the GPU, write w, set status=1 (or 2 on
          error), then connect to the job's address to wake the worker.

The wake-up is a socket rather than a polled flag so a worker queued behind a
busy GPU blocks in the kernel instead of waking thousands of times a second.
The address lives in Linux's abstract namespace, so it needs no filesystem
entry, no cleanup, and — unlike a semaphore or an fd — it is just a string and
travels through the ordinary job dict.

Numerics match price2.mu_solver / solvers.py (weighted Richardson-Lucy +
group-LASSO), so broker results equal the in-worker GPU/CPU MU results.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from multiprocessing import shared_memory

import numpy as np

from price2.mu_solver import silence_sparse_beta_warning


# ------------------------------------------------------------------ #
# job completion signalling (abstract AF_UNIX socket, no fd passing)   #
# ------------------------------------------------------------------ #
def _done_listener(timeout):
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


def _signal_done(address):
    """Wake the worker waiting on *address*; ignore a worker that gave up."""
    if not address:
        return
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(5.0)
            client.connect(address)
    except OSError:
        pass  # worker timed out or died; its shm is unlinked either way


# ------------------------------------------------------------------ #
# shared-memory array helpers                                          #
# ------------------------------------------------------------------ #
def _shm_create(size):
    """Create a SharedMemory block whose lifecycle we manage manually.

    track=False (py3.13+) keeps the resource_tracker out entirely. On older
    Pythons the block is tracked; the single owning ``unlink()`` (in the client's
    finally) is the one-and-only unregister, so we must NOT unregister anywhere
    else — the tracker cache is a shared set and a second remove KeyErrors.
    """
    try:
        return shared_memory.SharedMemory(create=True, size=size, track=False)
    except TypeError:
        return shared_memory.SharedMemory(create=True, size=size)


def _shm_put(arr):
    """Copy a numpy array into a fresh SharedMemory block; return (shm, meta)."""
    arr = np.ascontiguousarray(arr)
    shm = _shm_create(max(arr.nbytes, 1))
    view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    view[...] = arr
    return shm, (shm.name, arr.shape, str(arr.dtype))


def _shm_get(meta):
    """Attach to an existing SharedMemory block (owned by the caller/worker).

    The attaching process must NOT let its resource_tracker manage the block, or
    it may unlink shm it does not own (bpo-38119). Use track=False on 3.13+, and
    the resource_tracker.unregister workaround on older versions.
    """
    name, shape, dtype = meta
    try:
        shm = shared_memory.SharedMemory(name=name, track=False)  # py3.13+
    except TypeError:
        # <3.13: attaching re-registers the name, but the tracker cache is an
        # idempotent set, so the client's single unlink() still balances it. Do
        # NOT unregister here (that would double-remove and KeyError at teardown).
        shm = shared_memory.SharedMemory(name=name)
    arr = np.ndarray(tuple(shape), dtype=np.dtype(dtype), buffer=shm.buf)
    return shm, arr


@dataclass
class Params:
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


# ------------------------------------------------------------------ #
# GPU numerical core (runs inside a broker thread, on its own stream)  #
# ------------------------------------------------------------------ #
def _gpu_deconvolve(torch, job, dtype, check_every=25):
    """Full IRLS-Huber group-LASSO deconvolution of one locus on the GPU."""
    tiny = 1e-30 if dtype == torch.float32 else 1e-300
    shms = []
    try:
        sX, Xd = _shm_get(job["X_data"]); shms.append(sX)
        sXi, Xi = _shm_get(job["X_indices"]); shms.append(sXi)
        sXp, Xp = _shm_get(job["X_indptr"]); shms.append(sXp)
        sTd, Td = _shm_get(job["XT_data"]); shms.append(sTd)
        sTi, Ti = _shm_get(job["XT_indices"]); shms.append(sTi)
        sTp, Tp = _shm_get(job["XT_indptr"]); shms.append(sTp)
        sy, y = _shm_get(job["y"]); shms.append(sy)
        p = job["params"]
        m, n = job["X_shape"]
        nr, ns = p.num_rgrs, p.num_runs
        lam, pmin, c = p.lam, p.pseudo_min, p.huber_c
        theta = p.theta
        poisson = theta is None

        def csr(data, indices, indptr, shape):
            return torch.sparse_csr_tensor(
                torch.from_numpy(indptr.astype(np.int64)).cuda(),
                torch.from_numpy(indices.astype(np.int64)).cuda(),
                torch.from_numpy(data).to(dtype).cuda(),
                size=shape, dtype=dtype, device="cuda")

        Xc = csr(Xd, Xi, Xp, (m, n))
        XcT = csr(Td, Ti, Tp, (n, m))
        yt = torch.from_numpy(np.ascontiguousarray(y)).to(dtype).cuda()
        # Warm-start from the caller's w0 when provided (the EM light M-step,
        # which runs a single non-converged outer iteration, relies on it);
        # otherwise fall back to an all-ones cold start.
        w0_meta = job.get("w0")
        if w0_meta is not None:
            sw0, w0arr = _shm_get(w0_meta); shms.append(sw0)
            w = torch.from_numpy(np.ascontiguousarray(w0arr)).to(dtype).cuda()
        else:
            w = torch.ones(n, dtype=dtype, device="cuda")

        for outer in range(p.max_outer):
            delta = torch.mv(Xc, w).clamp_min(1e-14)
            # Pearson residual variance: δ (Poisson) or δ + δ²/θ (negative binomial)
            var = delta if poisson else delta + delta * delta / theta
            pr = (yt - delta) / var.sqrt()
            ar = pr.abs()
            omega = torch.where(ar <= c, torch.ones_like(ar), c / ar.clamp_min(1e-14))
            omega_y = omega * yt
            # Poisson denominator data term X^T ω is constant across inner iterations.
            if poisson:
                Xt_omega = torch.mv(XcT, omega)
            w_outer0 = w
            for it in range(p.mu_inner_max_iter):
                delta = torch.mv(Xc, w).clamp_min(tiny)
                num = torch.mv(XcT, omega_y / delta)
                if poisson:
                    data_den = Xt_omega
                else:
                    # NB denominator data term X^T(ω (y+θ)/(θ+δ)) depends on δ.
                    data_den = torch.mv(XcT, omega * (yt + theta) / (theta + delta))
                W = w.view(nr, ns)
                norms = (W * W).sum(1).sqrt()
                pen = (lam * (W / norms.clamp_min(tiny).view(-1, 1))).reshape(-1)
                den = (data_den + pen).clamp_min(tiny)
                w_new = (w * num / den).clamp_min(pmin)
                if (it + 1) % check_every == 0:
                    rel = ((w_new - w).norm() / w.norm().clamp_min(1e-14)).item()
                    w = w_new
                    if rel < p.mu_inner_tol:
                        break
                else:
                    w = w_new
            rel_o = ((w - w_outer0).norm() / w_outer0.norm().clamp_min(1e-14)).item()
            if rel_o < p.huber_tol:
                break

        w_np = w.double().cpu().numpy()
        sr, wr = _shm_get(job["result"])
        wr[...] = w_np
        sr.close()
        return outer + 1
    finally:
        for s in shms:
            s.close()


# ------------------------------------------------------------------ #
# Broker server process                                               #
# ------------------------------------------------------------------ #
def _broker_main(req_q, stop_evt, ready_val, n_streams, dtype_str):
    # Before any stream-thread builds its first CSR tensor, and in the broker
    # process only: the parent and the CPU workers keep their own filters.
    silence_sparse_beta_warning()
    import torch
    dtype = getattr(torch, dtype_str)
    torch.zeros(1, device="cuda")  # create THIS process's context (one per broker)
    with ready_val.get_lock():
        ready_val.value += 1

    def worker_thread():
        stream = torch.cuda.Stream()
        while True:
            try:
                job = req_q.get(timeout=0.2)
            except Exception:
                if stop_evt.is_set():
                    return
                continue
            if job is None:
                return
            status_shm, status = _shm_get(job["status"])
            try:
                with torch.cuda.stream(stream):
                    _gpu_deconvolve(torch, job, dtype)
                status[0] = 1
            except Exception as e:  # noqa: BLE001
                print(f"[broker] job {job.get('req_id')} FAILED: "
                      f"{type(e).__name__}: {str(e)[:160]}", flush=True)
                status[0] = 2
            finally:
                status_shm.close()
                # Always wake the worker, including on failure — otherwise it
                # blocks in accept() until its (long) timeout.
                _signal_done(job.get("done"))

    threads = [threading.Thread(target=worker_thread, daemon=True)
               for _ in range(n_streams)]
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

    def __init__(self, n_procs=1, n_streams=1, dtype_str="float32"):
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
                daemon=True)
            for _ in range(n_procs)
        ]

    def start(self, timeout=90):
        for p in self._procs:
            p.start()
        t0 = time.time()
        while True:
            with self._ready.get_lock():
                ready = self._ready.value
            if ready >= self.n_procs:
                return
            dead = [p for p in self._procs if not p.is_alive()]
            if dead and ready < self.n_procs:
                raise RuntimeError(
                    f"{len(dead)} broker process(es) died during startup "
                    "(torch/CUDA unavailable?)")
            if time.time() - t0 > timeout:
                raise RuntimeError("GPU broker pool failed to become ready")
            time.sleep(0.2)

    def stop(self):
        self._stop.set()
        try:
            for _ in range(self.n_procs * (self.n_streams + 2)):
                self.req_q.put(None)  # unblock each stream-thread's queue.get
        except Exception:
            pass
        for p in self._procs:
            p.join(timeout=30)
        for p in self._procs:
            if p.is_alive():
                p.terminate()
        try:
            self._mgr.shutdown()
        except Exception:
            pass


class BrokerClient:
    """Used inside a CPU worker: ship a locus system to the broker, get back w."""

    def __init__(self, req_q):
        self.req_q = req_q
        self._counter = 0

    def solve(self, X, XT, y, params: Params, w0=None, timeout=1200.0):
        self._counter += 1
        n = params.num_rgrs * params.num_runs
        shms = []
        server = None
        try:
            sXd, mXd = _shm_put(X.data.astype(np.float64)); shms.append(sXd)
            sXi, mXi = _shm_put(X.indices.astype(np.int64)); shms.append(sXi)
            sXp, mXp = _shm_put(X.indptr.astype(np.int64)); shms.append(sXp)
            sTd, mTd = _shm_put(XT.data.astype(np.float64)); shms.append(sTd)
            sTi, mTi = _shm_put(XT.indices.astype(np.int64)); shms.append(sTi)
            sTp, mTp = _shm_put(XT.indptr.astype(np.int64)); shms.append(sTp)
            sy, my = _shm_put(y.astype(np.float64)); shms.append(sy)
            mw0 = None
            if w0 is not None:
                sw0, mw0 = _shm_put(np.asarray(w0, dtype=np.float64)); shms.append(sw0)
            sr = _shm_create(n * 8); shms.append(sr)
            wr = np.ndarray((n,), dtype=np.float64, buffer=sr.buf); wr[...] = 0.0
            ss = _shm_create(1); shms.append(ss)
            status = np.ndarray((1,), dtype=np.uint8, buffer=ss.buf); status[0] = 0

            # Listen before enqueuing, so the broker cannot signal into a
            # socket that does not exist yet.
            server, done_address = _done_listener(timeout)

            job = {
                "req_id": self._counter,
                "X_data": mXd, "X_indices": mXi, "X_indptr": mXp,
                "XT_data": mTd, "XT_indices": mTi, "XT_indptr": mTp,
                "y": my, "X_shape": tuple(int(s) for s in X.shape),
                "w0": mw0,
                "result": (sr.name, (n,), "float64"),
                "status": (ss.name, (1,), "uint8"),
                "done": done_address,
                "params": params,
            }
            self.req_q.put(job)

            try:
                conn, _ = server.accept()  # blocks in the kernel, no wake-ups
                conn.close()
            except (socket.timeout, TimeoutError):
                raise TimeoutError("broker job timed out") from None

            if status[0] == 2:
                raise RuntimeError("broker reported job failure")
            if status[0] == 0:
                raise RuntimeError("broker signalled without writing a status")
            return wr.copy()
        finally:
            if server is not None:
                server.close()
            for s in shms:
                try:
                    s.close(); s.unlink()
                except Exception:
                    pass
