"""Built-in PyViz plugins.

Each module exposes ``register(viz)``, which adds panels, overlays, periodic
updates or tooltip information through the ``web.Visualizer`` API.
"""

import importlib
import sys
import traceback

PLUGINS = [
    "node_info",
    "interface_statistics",
    "ipv4_routing_table",
    "olsr",
    "show_last_packets",
    "wifi_links",
    "flow_monitor",
    "energy",
]


def register_all(viz):
    for name in PLUGINS:
        try:
            importlib.import_module("." + name, __name__).register(viz)
        except Exception:
            sys.stderr.write("PyViz: plugin %s failed to load\n" % name)
            traceback.print_exc()


def interface_name(device, index=None):
    """Name a device by its Names entry, or by its interface index."""
    from ns import ns

    if device is None:
        return "lo"
    name = str(ns.Names.FindName(device))
    if name:
        return name
    return "(interface %i)" % (device.GetIfIndex() if index is None else index)
