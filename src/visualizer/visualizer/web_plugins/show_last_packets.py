"""Last transmitted, received and dropped packets, with a header filter."""

from ns import ns

from . import interface_name


def _header_types():
    header = ns.TypeId.LookupByName("ns3::Header")
    trailer = ns.TypeId.LookupByName("ns3::Trailer")
    result = {}
    for index in range(int(ns.TypeId.GetRegisteredN())):
        type_id = ns.TypeId.GetRegistered(index)
        if type_id == header or type_id == trailer:
            continue
        parent = type_id
        while True:
            if parent == header or parent == trailer:
                result[str(type_id.GetName())] = type_id
                break
            if not parent.HasParent():
                break
            parent = parent.GetParent()
    return result


class LastPackets:
    def __init__(self):
        self.headers = _header_types()
        self.options = {}  # node -> {"headers": [...], "mode": "OR"|"AND"}

    def _apply(self, viz, node_index):
        selection = self.options[node_index]
        options = ns.visualizer.PacketCaptureOptions()
        options.numLastPackets = 100
        options.mode = (
            ns.visualizer.PACKET_CAPTURE_FILTER_HEADERS_AND
            if selection["mode"] == "AND"
            else ns.visualizer.PACKET_CAPTURE_FILTER_HEADERS_OR
        )
        for name in selection["headers"]:
            options.headers.insert(self.headers[name])
        viz.sim_helper.SetPacketCaptureOptions(node_index, options)

    def _rows(self, samples):
        rows = []
        for sample in samples:
            device = sample.device if sample.device else None
            rows.append(
                [
                    round(float(sample.time.GetSeconds()), 6),
                    "(unknown)" if device is None else interface_name(device),
                    int(sample.packet.GetSize()),
                    str(sample.packet.ToString()),
                ]
            )
        return rows

    def build(self, viz, node_index):
        if node_index not in self.options:
            self.options[node_index] = {"headers": sorted(self.headers), "mode": "OR"}
            self._apply(viz, node_index)
        last = viz.sim_helper.GetLastPackets(node_index)
        columns = ["Time", "Interface", "Size", "Contents"]
        selection = self.options[node_index]
        return {
            "sections": [
                {
                    "title": "Last transmitted packets",
                    "columns": columns,
                    "rows": self._rows(last.lastTransmittedPackets),
                },
                {
                    "title": "Last received packets",
                    "columns": columns,
                    "rows": self._rows(last.lastReceivedPackets),
                },
                {
                    "title": "Last dropped packets",
                    "columns": columns,
                    "rows": self._rows(last.lastDroppedPackets),
                },
            ],
            "controls": [
                {
                    "name": "mode",
                    "kind": "radio",
                    "label": "Header match",
                    "options": ["OR", "AND"],
                    "value": selection["mode"],
                },
                {
                    "name": "headers",
                    "kind": "checklist",
                    "label": "Headers",
                    "options": sorted(self.headers),
                    "value": selection["headers"],
                },
            ],
        }

    def action(self, viz, node_index, values):
        headers = [h for h in values.get("headers", []) if h in self.headers]
        mode = "AND" if values.get("mode") == "AND" else "OR"
        self.options[node_index] = {"headers": headers, "mode": mode}
        self._apply(viz, node_index)


def register(viz):
    last_packets = LastPackets()
    viz.add_node_panel("last-packets", "Last packets", last_packets.build, last_packets.action)
