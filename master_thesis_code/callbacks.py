"""Simulation callback protocol for decoupling side effects from computation.

The :class:`SimulationCallback` protocol defines hooks that
:func:`~master_thesis_code.main.data_simulation` calls at key points in
the simulation loop.  Implementations can collect data for plotting,
logging, monitoring, etc. without coupling those concerns into the loop
itself.
"""

from typing import Protocol


class SimulationCallback(Protocol):
    """Hook points for the EMRI simulation loop."""

    def on_simulation_start(self, total_steps: int) -> None:
        """Called once before the simulation loop begins."""
        ...

    def on_snr_computed(self, step: int, snr: float, passed: bool) -> None:
        """Called after each SNR evaluation.

        *step* is the current detection count, *snr* the computed value,
        and *passed* whether it exceeds the threshold.
        """
        ...

    def on_detection(
        self, step: int, snr: float, cramer_rao: dict[str, float], host_idx: int
    ) -> None:
        """Called when a detection is recorded (SNR above threshold + Cramer-Rao computed)."""
        ...

    def on_step_end(self, step: int, iteration: int) -> None:
        """Called at the end of every loop iteration (detection or not)."""
        ...

    def on_simulation_end(self, total_detections: int, total_iterations: int) -> None:
        """Called once after the simulation loop completes."""
        ...


class NullCallback:
    """Default no-op implementation of :class:`SimulationCallback`."""

    def on_simulation_start(self, total_steps: int) -> None:
        pass

    def on_snr_computed(self, step: int, snr: float, passed: bool) -> None:
        pass

    def on_detection(
        self, step: int, snr: float, cramer_rao: dict[str, float], host_idx: int
    ) -> None:
        pass

    def on_step_end(self, step: int, iteration: int) -> None:
        pass

    def on_simulation_end(self, total_detections: int, total_iterations: int) -> None:
        pass
