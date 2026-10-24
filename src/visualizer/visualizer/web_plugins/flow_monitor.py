#
# Copyright (c) 2026 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
#
"""Optional FlowMonitor time series: throughput, bytes, latency and jitter.

Disabled unless the scenario calls ``visualizer.enable_flow_monitor()`` or the
user enables it from the panel. Flows are filtered to those whose source or
destination is one of the selected nodes, so a selection of several nodes
shows the traffic between them.
"""

import sys

from ns import ns

from ..traces import type_is_registered

## Number of samples kept per flow.
HISTORY = 600  # samples kept per flow
# Simulated seconds between samples. Steps vary in length with the browser's
# poll rate, so sampling on a fixed simulated interval keeps the series even.
## Simulated seconds between samples.
SAMPLE_INTERVAL = 0.1
## Maximum number of flows plotted in a chart.
MAX_SERIES = 8  # categorical palette size; further flows are listed but not plotted
## Names of IP protocol numbers.
PROTOCOLS = {6: "TCP", 17: "UDP"}

## Flow monitor enable request made before the visualizer starts.
_requested = {"enabled": False, "helper": None}


def enable(helper=None):
    """Request the flow monitor to be enabled when the visualizer starts.

    @param helper Optional FlowMonitorHelper already installed by the scenario.
    """
    _requested["enabled"] = True
    _requested["helper"] = helper


class FlowMonitorSeries:
    """Sample FlowMonitor statistics into per-flow time series."""

    ## @var helper
    ## FlowMonitorHelper in use, or None
    ## @var monitor
    ## FlowMonitor in use, or None while disabled
    ## @var classifier
    ## Ipv4FlowClassifier mapping flow ids to five-tuples
    ## @var address_nodes
    ## Map from IPv4 address string to node index
    ## @var flows
    ## Map from flow id to its label, endpoints, last counters and points
    ## @var last_sample
    ## Simulation time of the previous sample, or None
    ## @var enabled
    ## Whether the flow monitor is running

    def __init__(self):
        """Initializer.

        @param self The object pointer.
        """
        self.helper = None
        self.monitor = None
        self.classifier = None
        self.address_nodes = {}  # Ipv4 address string -> node id
        self.flows = {}  # flow id -> dict(label, source, target, last, points)
        self.last_sample = None  # simulation time of the previous sample

    @property
    def enabled(self):
        """Whether the flow monitor is running.

        @param self The object pointer.
        @return True if a FlowMonitor is installed.
        """
        return self.monitor is not None

    def start(self, viz, helper=None):
        """Start collecting flow statistics.

        @param self The object pointer.
        @param viz The web visualizer.
        @param helper Optional FlowMonitorHelper to use. If None, the FlowMonitor
               the scenario created is used, or one is installed on all nodes.
        """
        if helper is None:
            # A new helper returns the FlowMonitor the scenario already
            # created, if any; one FlowMonitor is shared per simulation.
            helper = ns.FlowMonitorHelper()
            self.monitor = (
                helper.GetMonitor() if ns.FlowMonitorHelper.HasMonitor() else helper.InstallAll()
            )
        else:
            self.monitor = helper.GetMonitor()
        self.helper = helper
        self.classifier = ns.DynamicCast[ns.Ipv4FlowClassifier](helper.GetClassifier())
        for index in range(int(ns.NodeList.GetNNodes())):
            ipv4 = ns.NodeList.GetNode(index).GetObject[ns.Ipv4]()
            if not ipv4:
                continue
            for interface in range(int(ipv4.GetNInterfaces())):
                for address in range(int(ipv4.GetNAddresses(interface))):
                    self.address_nodes[str(ipv4.GetAddress(interface, address).GetLocal())] = index

    def periodic_update(self, viz):
        """Sample the flow statistics at SAMPLE_INTERVAL.

        @param self The object pointer.
        @param viz The web visualizer.
        """
        if not self.enabled:
            return
        now = float(ns.Simulator.Now().GetSeconds())
        if self.last_sample is not None and now - self.last_sample < SAMPLE_INTERVAL:
            return
        self.last_sample = now
        self.monitor.CheckForLostPackets()
        for item in self.monitor.GetFlowStats():
            flow_id, stats = int(item.first), item.second
            flow = self.flows.get(flow_id)
            if flow is None:
                tuple_ = self.classifier.FindFlow(flow_id)
                source, target = str(tuple_.sourceAddress), str(tuple_.destinationAddress)
                flow = self.flows[flow_id] = {
                    "label": "Flow %d: %s:%d -> %s:%d %s"
                    % (
                        flow_id,
                        source,
                        tuple_.sourcePort,
                        target,
                        tuple_.destinationPort,
                        PROTOCOLS.get(int(tuple_.protocol), str(int(tuple_.protocol))),
                    ),
                    "source": self.address_nodes.get(source),
                    "target": self.address_nodes.get(target),
                    "last": None,
                    "points": [],
                }
            current = (
                now,
                int(stats.txBytes),
                int(stats.rxBytes),
                int(stats.rxPackets),
                float(stats.delaySum.GetSeconds()),
                float(stats.jitterSum.GetSeconds()),
                int(stats.txPackets),
                int(stats.lostPackets),
            )
            last = flow["last"]
            if last is not None and now > last[0]:
                dt = now - last[0]
                rx_packets = current[3] - last[3]
                flow["points"].append(
                    {
                        "t": now,
                        "throughput": (current[2] - last[2]) * 8 / dt / 1e3,
                        "tx": current[1],
                        "rx": current[2],
                        # Interval averages; no received packets means no sample.
                        "latency": (
                            (current[4] - last[4]) / rx_packets * 1e3 if rx_packets else None
                        ),
                        "jitter": (current[5] - last[5]) / rx_packets * 1e3 if rx_packets else None,
                    }
                )
                del flow["points"][: max(0, len(flow["points"]) - HISTORY)]
            flow["last"] = current

    def _selected_flows(self, viz):
        """Get the flows from or to the selected nodes (all if none selected).

        @param self The object pointer.
        @param viz The web visualizer.
        @return List of (flow id, flow) pairs.
        """
        selected = set(viz.selected_nodes)
        return [
            (flow_id, flow)
            for flow_id, flow in sorted(self.flows.items())
            if not selected or flow["source"] in selected or flow["target"] in selected
        ]

    def build(self, viz):
        """Build the flow monitor panel.

        @param self The object pointer.
        @param viz The web visualizer.
        @return The panel content dictionary.
        """
        if not self.enabled:
            return {
                "sections": [
                    {
                        "title": "Flow monitor is off",
                        "note": "Enable it to collect per-flow statistics from now on. "
                        "Scenarios can enable it from the start with "
                        "visualizer.enable_flow_monitor().",
                    }
                ],
                "controls": [{"name": "enable", "kind": "button", "label": "Enable flow monitor"}],
            }
        flows = self._selected_flows(viz)
        plotted = flows[:MAX_SERIES]
        scope = (
            "flows to or from nodes %s" % ", ".join(map(str, viz.selected_nodes))
            if viz.selected_nodes
            else "all flows (Shift/Ctrl-click nodes to filter)"
        )
        note = "Showing %s." % scope
        if len(flows) > MAX_SERIES:
            note += " Plotting the first %d of %d flows; all are in the table." % (
                MAX_SERIES,
                len(flows),
            )

        def series(key):
            """Build the plotted series of one metric.

            @param key The point metric key.
            @return List of series dictionaries.
            """
            return [
                {
                    "name": flow["label"],
                    "slot": index,
                    "points": [[p["t"], p[key]] for p in flow["points"]],
                }
                for index, (_id, flow) in enumerate(plotted)
            ]

        # Totals over the plotted flows; each flow holds its latest value
        # between its own samples.
        samples = sorted(
            (p["t"], index, p["tx"], p["rx"])
            for index, (_id, flow) in enumerate(plotted)
            for p in flow["points"]
        )
        latest = {}
        tx_total, rx_total = [], []
        for t, index, tx, rx in samples:
            latest[index] = (tx, rx)
            point_tx = sum(v[0] for v in latest.values())
            point_rx = sum(v[1] for v in latest.values())
            if tx_total and tx_total[-1][0] == t:
                tx_total[-1][1], rx_total[-1][1] = point_tx, point_rx
            else:
                tx_total.append([t, point_tx])
                rx_total.append([t, point_rx])

        rows = []
        for flow_id, flow in flows:
            last = flow["last"]
            rx_packets = last[3] if last else 0
            rows.append(
                [
                    flow["label"],
                    last[6] if last else 0,
                    rx_packets,
                    last[7] if last else 0,
                    last[1] if last else 0,
                    last[2] if last else 0,
                    round(last[4] / rx_packets * 1e3, 3) if rx_packets else "-",
                    round(last[5] / (rx_packets - 1) * 1e3, 3) if rx_packets > 1 else "-",
                ]
            )
        return {
            "sections": [
                {
                    "title": "Throughput",
                    "note": note,
                    "plot": {"unit": "kbit/s", "series": series("throughput")},
                },
                {
                    "title": "Total bytes",
                    "plot": {
                        "unit": "bytes",
                        "series": [
                            {"name": "Transmitted", "slot": 0, "points": tx_total},
                            {"name": "Received", "slot": 1, "points": rx_total},
                        ],
                    },
                },
                {
                    "title": "Latency (interval mean)",
                    "plot": {"unit": "ms", "series": series("latency")},
                },
                {
                    "title": "Jitter (interval mean)",
                    "plot": {"unit": "ms", "series": series("jitter")},
                },
                {
                    "title": "Flows",
                    "columns": [
                        "Flow",
                        "Tx pkts",
                        "Rx pkts",
                        "Lost",
                        "Tx bytes",
                        "Rx bytes",
                        "Mean delay (ms)",
                        "Mean jitter (ms)",
                    ],
                    "rows": rows,
                },
            ]
        }

    def action(self, viz, values):
        """Handle the panel controls; enable the monitor on request.

        @param self The object pointer.
        @param viz The web visualizer.
        @param values Dictionary of submitted control values.
        """
        if values.get("enable") and not self.enabled:
            self.start(viz)


def register(viz):
    """Add the flow monitor panel, greyed out when the flow-monitor module is not built.

    @param viz The web visualizer.
    """
    if not type_is_registered("ns3::FlowMonitor"):
        reason = "Flow monitor unavailable: the flow-monitor module is not built"
        if _requested["enabled"]:
            sys.stderr.write("PyViz: %s\n" % reason)
        viz.add_panel("flow-monitor", "Flow monitor", lambda _viz: {}, unavailable=reason)
        return
    series = FlowMonitorSeries()
    # A FlowMonitor the scenario created is shown from the start.
    if _requested["enabled"] or ns.FlowMonitorHelper.HasMonitor():
        series.start(viz, _requested["helper"])
    viz.add_periodic_update(series.periodic_update)
    viz.add_panel("flow-monitor", "Flow monitor", series.build, series.action)
