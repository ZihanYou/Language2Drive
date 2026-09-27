"""Official single-worker runtime with evaluator-local metrics binding."""

from functools import partial
import runpy


def main():
    from .raw_lane_metadata import install
    from .runtime_map_evidence import install as install_map_evidence

    install()
    install_map_evidence()
    import alpasim_runtime.worker.main as worker

    # Wizard's container deployment relies on Docker port publication. On the
    # host, explicitly use the existing TelemetryContext bind_host parameter.
    worker.TelemetryContext = partial(worker.TelemetryContext, bind_host="127.0.0.1")
    runpy.run_module("alpasim_runtime.simulate", run_name="__main__")


if __name__ == "__main__":
    main()
