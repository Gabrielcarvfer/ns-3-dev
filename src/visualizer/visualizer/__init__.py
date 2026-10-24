"""PyViz: live, browser based visualizer for ns-3 Python simulations."""


def start():
    """Open the HTML5 visualizer and run the configured simulation."""
    from .web import start

    return start()


def add_initialization_hook(hook, *args):
    """Call hook(viz, *args) once the visualizer is attached to the simulation."""
    from .web import add_initialization_hook as add_hook

    return add_hook(hook, *args)


def register_plugin(plugin):
    """Call plugin(viz) when the visualizer starts."""
    from .web import register_plugin as register

    return register(plugin)


def enable_flow_monitor(helper=None):
    """Plot FlowMonitor statistics (throughput, bytes, latency, jitter) in the browser.

    Optional: pass a FlowMonitorHelper the scenario already installed, or let
    the visualizer install one on all nodes.
    """
    from .web import enable_flow_monitor as enable

    return enable(helper)


def set_bounds(x1, y1, x2, y2):
    """Set the initially visible area, in simulation coordinates."""
    from .web import set_bounds as set_visual_bounds

    return set_visual_bounds(x1, y1, x2, y2)
