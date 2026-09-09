"""The stages of a run and how they are driven.

A :class:`Stage` names one step of the pipeline, knows how to run it, and
optionally how to tell that a previous invocation already finished it.
:func:`run_stages` executes a list of them in order, skipping the finished
ones and logging the wall time of the rest, so that
:func:`price2.price.run_pipeline` reads as the list of steps it is.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Stage:
    """One step of the pipeline.

    Parameters
    ----------
    name : str
        What the step does, as logged.
    run : callable
        Performs the step.  Every step must be safe to re-run after an
        interruption; it picks up what an earlier invocation persisted.
    enabled : bool
        Whether the configuration asks for the step at all.
    done : callable or None
        Returns ``True`` when a previous invocation completed the step and
        nothing is left to do; the step is then skipped.
    """

    name: str
    run: Callable[[], object]
    enabled: bool = True
    done: Callable[[], bool] | None = None


def run_stages(stages: Iterable[Stage]) -> None:
    """Run *stages* in order, skipping disabled and finished ones."""
    for stage in stages:
        if not stage.enabled:
            continue
        if stage.done is not None and stage.done():
            logger.info("%s: already done", stage.name)
            continue
        run_stage(stage.name, stage.run)


def run_stage(name: str, run: Callable[[], object]) -> object:
    """Run one step and log its wall time."""
    start = time.time()
    result = run()
    logger.info("%s... %.2e seconds", name, time.time() - start)
    return result
