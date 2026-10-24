"""OLSR routing table, available when the olsr module is built."""

from ns import ns

from ..traces import type_is_registered
from . import interface_name


def _olsr(node_index):
    return ns.NodeList.GetNode(node_index).GetObject[ns.olsr.RoutingProtocol]()


def build(viz, node_index):
    ipv4 = ns.NodeList.GetNode(node_index).GetObject[ns.Ipv4]()
    rows = []
    for route in _olsr(node_index).__deref__().GetRoutingTableEntries():
        device = ipv4.GetNetDevice(route.interface)
        rows.append(
            [
                str(route.destAddr),
                str(route.nextAddr),
                interface_name(device if device else None, int(route.interface)),
                int(route.distance),
            ]
        )
    return {
        "sections": [
            {
                "title": "OLSR routing table",
                "columns": ["Destination", "Next hop", "Interface", "Num. Hops"],
                "rows": rows,
            }
        ]
    }


def register(viz):
    if not type_is_registered("ns3::olsr::RoutingProtocol"):
        return
    viz.add_node_panel(
        "olsr-routing", "OLSR routing table", build, applies=lambda viz, n: bool(_olsr(n))
    )
