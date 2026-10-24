"""Per-interface packet and bit counters and rates."""

from ns import ns

from . import interface_name

NODE_STATISTICS_MEMORY = 10


class StatisticsCollector:
    def __init__(self):
        # node id -> list of (simulation time, [(txP, txB, rxP, rxB) per interface])
        self.node_statistics = {}

    def periodic_update(self, viz):
        for stats in viz.sim_helper.GetNodesStatistics():
            history = self.node_statistics.setdefault(int(stats.nodeId), [])
            history.append(
                (
                    float(ns.Simulator.Now().GetSeconds()),
                    [
                        (
                            int(s.transmittedPackets),
                            int(s.transmittedBytes),
                            int(s.receivedPackets),
                            int(s.receivedBytes),
                        )
                        for s in stats.statistics
                    ],
                )
            )
            del history[: max(0, len(history) - NODE_STATISTICS_MEMORY)]

    def build(self, viz, node_index):
        columns = [
            "Interface",
            "Tx Packets",
            "Tx Bytes",
            "Tx pkt/1s",
            "Tx bit/1s",
            "Rx Packets",
            "Rx Bytes",
            "Rx pkt/1s",
            "Rx bit/1s",
        ]
        history = self.node_statistics.get(node_index, [])
        if len(history) < 2:
            return {
                "sections": [
                    {
                        "title": "Interface statistics",
                        "columns": columns,
                        "rows": [],
                        "note": "Collecting samples...",
                    }
                ]
            }
        node = ns.NodeList.GetNode(node_index)
        # Rates use whatever history is available so the table fills
        # before the full window is collected.
        span = history[-1][0] - history[0][0]
        if span <= 0:
            return {
                "sections": [
                    {
                        "title": "Interface statistics",
                        "columns": columns,
                        "rows": [],
                        "note": "Collecting samples...",
                    }
                ]
            }
        first, last = history[0][1], history[-1][1]
        rows = []
        for iface, (tx_p, tx_b, rx_p, rx_b) in enumerate(last):
            tx_p0, tx_b0, rx_p0, rx_b0 = first[iface] if iface < len(first) else (0, 0, 0, 0)
            device = node.GetDevice(iface) if iface < int(node.GetNDevices()) else None
            rows.append(
                [
                    interface_name(device, iface),
                    tx_p,
                    tx_b,
                    round((tx_p - tx_p0) / span, 2),
                    round((tx_b - tx_b0) * 8 / span, 2),
                    rx_p,
                    rx_b,
                    round((rx_p - rx_p0) / span, 2),
                    round((rx_b - rx_b0) * 8 / span, 2),
                ]
            )
        return {"sections": [{"title": "Interface statistics", "columns": columns, "rows": rows}]}


def register(viz):
    collector = StatisticsCollector()
    viz.add_periodic_update(collector.periodic_update)
    viz.add_node_panel("interface-statistics", "Interface statistics", collector.build)
