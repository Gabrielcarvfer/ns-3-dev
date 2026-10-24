"""Batteries of nodes with an energy source (energy module).

Adds to each node's snapshot a "battery" entry (charge fraction, remaining
energy and mAh) drawn as a battery icon, a per-node "Battery" panel (opened by
clicking the icon) with the charge over time, its drainers (device energy
models) and charge sources (energy harvesters), and a simulation-wide
"Energy" panel with the network totals.
"""

from ns import ns

from ..traces import type_is_registered

SAMPLE_INTERVAL = 0.1  # simulated seconds between history samples
HISTORY_LIMIT = 1200  # samples kept; older history is thinned out, not dropped
MAX_SERIES = 8  # charts plot at most this many nodes (categorical palette size)


def _mah(joules, voltage):
    """Charge in mAh for an energy at a supply voltage: J / V = A*s, / 3.6 = mAh."""
    return joules / voltage / 3.6 if voltage > 0 else 0.0


def _sources(node_index):
    node = ns.NodeList.GetNode(node_index)
    container = node.GetObject[ns.energy.EnergySourceContainer]()
    if container:
        return [container.Get(index) for index in range(int(container.GetN()))]
    # Some scenarios aggregate a single energy source directly to the node.
    source = node.GetObject[ns.energy.EnergySource]()
    return [source] if source else []


def _harvesters(node_index):
    container = ns.NodeList.GetNode(node_index).GetObject[ns.energy.EnergyHarvesterContainer]()
    if not container:
        return []
    return [container.Get(index) for index in range(int(container.GetN()))]


def _device_model_types():
    """All registered DeviceEnergyModel subclasses (EnergySource only finds models by exact type)."""
    base = ns.TypeId.LookupByName("ns3::energy::DeviceEnergyModel")
    types = []
    for index in range(int(ns.TypeId.GetRegisteredN())):
        type_id = ns.TypeId.GetRegistered(index)
        parent = type_id
        while parent.HasParent() and parent != base:
            parent = parent.GetParent()
        if parent == base and type_id != base:
            types.append(type_id)
    return types


def battery(node_index):
    """Remaining charge summed over the node's energy sources, or None."""
    sources = _sources(node_index)
    if not sources:
        return None
    remaining = initial = remaining_mah = initial_mah = 0.0
    for source in sources:
        voltage = float(source.GetSupplyVoltage())
        source_remaining = float(source.GetRemainingEnergy())
        source_initial = float(source.GetInitialEnergy())
        remaining += source_remaining
        initial += source_initial
        remaining_mah += _mah(source_remaining, voltage)
        initial_mah += _mah(source_initial, voltage)
    return {
        "fraction": max(0.0, min(1.0, remaining / initial)) if initial > 0 else 0.0,
        "remainingJ": remaining,
        "initialJ": initial,
        "remainingmAh": remaining_mah,
        "capacitymAh": initial_mah,
    }


class EnergyMonitor:
    def __init__(self):
        self.model_types = _device_model_types()
        self.nodes = [i for i in range(int(ns.NodeList.GetNNodes())) if _sources(i)]
        self.history = {}  # node -> [[t, percent, remaining J], ...]
        self.last_sample = None

    # -- sampling --------------------------------------------------------

    def periodic_update(self, viz):
        now = float(ns.Simulator.Now().GetSeconds())
        if self.last_sample is not None and now - self.last_sample < SAMPLE_INTERVAL:
            return
        self.last_sample = now
        for node in self.nodes:
            state = battery(node)
            history = self.history.setdefault(node, [])
            history.append([now, state["fraction"] * 100, state["remainingJ"]])
            if len(history) > HISTORY_LIMIT:
                # Keep the whole run visible: halve the resolution of the
                # older half instead of discarding the start.
                half = len(history) // 2
                history[:half] = history[:half:2]

    def total_history(self):
        """Network remaining energy over time, summed over nodes at shared sample times."""
        series = {}
        for history in self.history.values():
            for t, _percent, joules in history:
                series[t] = series.get(t, 0.0) + joules
        return [[t, joules] for t, joules in sorted(series.items())]

    # -- drainers and charge sources --------------------------------------

    def drainers(self, node):
        rows = []
        for source_index, source in enumerate(_sources(node)):
            for type_id in self.model_types:
                models = source.FindDeviceEnergyModels(type_id)
                for index in range(int(models.GetN())):
                    model = models.Get(index)
                    rows.append(
                        [
                            node,
                            source_index,
                            str(type_id.GetName()),
                            round(float(model.GetCurrentA()) * 1e3, 3),
                            round(float(model.GetTotalEnergyConsumption()), 3),
                        ]
                    )
        return rows

    def harvesters(self, node):
        rows = []
        for harvester in _harvesters(node):
            rows.append(
                [
                    node,
                    str(harvester.GetInstanceTypeId().GetName()),
                    round(float(harvester.GetPower()) * 1e3, 3),
                ]
            )
        return rows

    def source_rows(self, node):
        rows = []
        for index, source in enumerate(_sources(node)):
            voltage = float(source.GetSupplyVoltage())
            remaining, initial = float(source.GetRemainingEnergy()), float(
                source.GetInitialEnergy()
            )
            rows.append(
                [
                    node,
                    index,
                    str(source.GetInstanceTypeId().GetName()),
                    round(voltage, 3),
                    round(initial, 3),
                    round(remaining, 3),
                    round(_mah(remaining, voltage), 3),
                    round(100 * remaining / initial, 2) if initial > 0 else 0.0,
                ]
            )
        return rows

    # -- panels ------------------------------------------------------------

    def build_node(self, viz, node):
        state = battery(node)
        history = self.history.get(node, [])
        sections = [
            {
                "title": "Battery",
                "items": [
                    ["Charge", "%.2f%%" % (state["fraction"] * 100)],
                    [
                        "Remaining",
                        "%.3f mAh (%.3f J)" % (state["remainingmAh"], state["remainingJ"]),
                    ],
                    ["Capacity", "%.3f mAh (%.3f J)" % (state["capacitymAh"], state["initialJ"])],
                ],
            },
            {
                "title": "Charge",
                "plot": {
                    "unit": "%",
                    "series": [
                        {
                            "name": "Node %d" % node,
                            "slot": 0,
                            "points": [[t, p] for t, p, _j in history],
                        }
                    ],
                },
            },
            {
                "title": "Remaining energy",
                "plot": {
                    "unit": "J",
                    "series": [
                        {
                            "name": "Node %d" % node,
                            "slot": 0,
                            "points": [[t, j] for t, _p, j in history],
                        }
                    ],
                },
            },
            {"title": "Sources", "columns": SOURCE_COLUMNS, "rows": self.source_rows(node)},
            {
                "title": "Drainers (device energy models)",
                "columns": DRAINER_COLUMNS,
                "rows": self.drainers(node),
            },
        ]
        harvesters = self.harvesters(node)
        if harvesters:
            sections.append(
                {
                    "title": "Charge sources (harvesters)",
                    "columns": HARVESTER_COLUMNS,
                    "rows": harvesters,
                }
            )
        return {"sections": sections}

    def build_global(self, viz):
        if not self.nodes:
            return {"sections": [{"title": "Energy", "note": "No node has an energy source."}]}
        states = {node: battery(node) for node in self.nodes}
        initial = sum(s["initialJ"] for s in states.values())
        remaining = sum(s["remainingJ"] for s in states.values())
        plotted = self.nodes[:MAX_SERIES]
        note = None
        if len(self.nodes) > MAX_SERIES:
            note = "Plotting the first %d of %d nodes; all are in the tables." % (
                MAX_SERIES,
                len(self.nodes),
            )
        rows = lambda method: [row for node in self.nodes for row in method(node)]
        return {
            "sections": [
                {
                    "title": "Network",
                    "items": [
                        ["Nodes with an energy source", str(len(self.nodes))],
                        [
                            "Energy at the start",
                            "%.3f J (%.3f mAh)"
                            % (initial, sum(s["capacitymAh"] for s in states.values())),
                        ],
                        [
                            "Energy now",
                            "%.3f J (%.3f mAh)"
                            % (remaining, sum(s["remainingmAh"] for s in states.values())),
                        ],
                        [
                            "Consumed",
                            "%.3f J (%.2f%%)"
                            % (
                                initial - remaining,
                                100 * (initial - remaining) / initial if initial else 0,
                            ),
                        ],
                    ],
                },
                {
                    "title": "Charge per node",
                    "note": note,
                    "plot": {
                        "unit": "%",
                        "series": [
                            {
                                "name": "Node %d" % node,
                                "slot": slot,
                                "points": [[t, p] for t, p, _j in self.history.get(node, [])],
                            }
                            for slot, node in enumerate(plotted)
                        ],
                    },
                },
                {
                    "title": "Total remaining energy",
                    "plot": {
                        "unit": "J",
                        "series": [{"name": "Total", "slot": 0, "points": self.total_history()}],
                    },
                },
                {"title": "Sources", "columns": SOURCE_COLUMNS, "rows": rows(self.source_rows)},
                {
                    "title": "Drainers (device energy models)",
                    "columns": DRAINER_COLUMNS,
                    "rows": rows(self.drainers),
                },
                {
                    "title": "Charge sources (harvesters)",
                    "columns": HARVESTER_COLUMNS,
                    "rows": rows(self.harvesters),
                },
            ]
        }

    # -- node snapshot and summary ------------------------------------------

    def node_info(self, viz, node):
        state = battery(node) if node in self.nodes else None
        return {"battery": state} if state else {}

    def tooltip_info(self, viz, node):
        if node not in self.nodes:
            return []
        state = battery(node)
        return [
            ["Battery", "%.1f%%" % (state["fraction"] * 100)],
            [
                "Remaining",
                "%.2f mAh (%.2f J) of %.2f mAh"
                % (state["remainingmAh"], state["remainingJ"], state["capacitymAh"]),
            ],
        ]


SOURCE_COLUMNS = [
    "Node",
    "Source",
    "Type",
    "Voltage (V)",
    "Initial (J)",
    "Remaining (J)",
    "Remaining (mAh)",
    "Charge (%)",
]
DRAINER_COLUMNS = ["Node", "Source", "Model", "Current (mA)", "Consumed (J)"]
HARVESTER_COLUMNS = ["Node", "Harvester", "Power (mW)"]


def register(viz):
    if not type_is_registered("ns3::energy::EnergySourceContainer"):
        return
    monitor = EnergyMonitor()
    viz.add_periodic_update(monitor.periodic_update)
    viz.add_node_info(monitor.node_info)
    viz.add_tooltip_info(monitor.tooltip_info)
    viz.add_node_panel(
        "battery", "Battery", monitor.build_node, applies=lambda viz, node: node in monitor.nodes
    )
    viz.add_panel("energy", "Energy", monitor.build_global)
