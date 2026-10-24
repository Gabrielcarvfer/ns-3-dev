"""Trace sources the visualizer listens to, connected only for built modules.

The C++ PyViz helper only provides generic handlers ("Tx", "Rx", "Drop" for
trace sources with the signature void (Ptr<const Packet>), and RecordTx(),
RecordRx() and RecordDrop() to call directly), so libvisualizer does not link
against any device or protocol module. This file decides which trace sources
of which types feed them. Trace sources with other signatures get small C++
adapters, compiled at run time by Cppyy in this process, where the modules
that define those types are already loaded.
"""

from ns import ns

_DEVICE = "/NodeList/*/DeviceList/*/"

# (required TypeId, generic handler, Config path)
TRACE_SOURCES = [
    ("ns3::WifiNetDevice", "Tx", _DEVICE + "$ns3::WifiNetDevice/Mac/MacTx"),
    ("ns3::WifiNetDevice", "Rx", _DEVICE + "$ns3::WifiNetDevice/Mac/MacRx"),
    ("ns3::lrwpan::LrWpanNetDevice", "Tx", _DEVICE + "$ns3::lrwpan::LrWpanNetDevice/Mac/MacTx"),
    ("ns3::lrwpan::LrWpanNetDevice", "Rx", _DEVICE + "$ns3::lrwpan::LrWpanNetDevice/Mac/MacRx"),
    ("ns3::CsmaNetDevice", "Tx", _DEVICE + "$ns3::CsmaNetDevice/MacTx"),
    ("ns3::CsmaNetDevice", "Rx", _DEVICE + "$ns3::CsmaNetDevice/MacRx"),
    ("ns3::PointToPointNetDevice", "Tx", _DEVICE + "$ns3::PointToPointNetDevice/MacTx"),
    ("ns3::PointToPointNetDevice", "Rx", _DEVICE + "$ns3::PointToPointNetDevice/MacRx"),
    ("ns3::LteNetDevice", "Tx", _DEVICE + "$ns3::LteNetDevice/Tx"),
    ("ns3::LteNetDevice", "Rx", _DEVICE + "$ns3::LteNetDevice/Rx"),
    (None, "Drop", _DEVICE + "TxQueue/Drop"),
]

# C++ adapters for trace sources whose signature is not void (Ptr<const Packet>).
# Each is compiled only when its module is part of the build.
_ADAPTERS = {
    # Ipv4L3Protocol Drop: (header, packet, reason, ipv4, interface). The
    # header was already removed from the packet, so it is added back.
    "ipv4-drop": (
        "ns3::Ipv4L3Protocol",
        r"""
        #include "ns3/ipv4-l3-protocol.h"
        namespace pyviz_adapters
        {
        void Ipv4Drop(ns3::visualizer::PyViz* viz, uint32_t node, std::string,
                      const ns3::Ipv4Header& header, ns3::Ptr<const ns3::Packet> packet,
                      ns3::Ipv4L3Protocol::DropReason, ns3::Ptr<ns3::Ipv4>, uint32_t)
        {
            ns3::Ptr<ns3::Packet> copy = packet->Copy();
            copy->AddHeader(header);
            viz->RecordDrop(node, copy);
        }
        bool ConnectIpv4Drop(ns3::visualizer::PyViz* viz, uint32_t node, const std::string& path)
        {
            return ns3::Config::ConnectFailSafe(path, ns3::MakeBoundCallback(&Ipv4Drop, viz, node));
        }
        }
        """,
    ),
    # With EPC, eNB user data bypasses the eNB device's Send()/Receive(); the
    # EpcEnbApplication sees it instead: RxFromEnb (arrived over the radio)
    # and RxFromS1u (about to be sent over the radio). They are recorded on
    # the node's eNB device, which Python looks up.
    "lte-enb": (
        "ns3::EpcEnbApplication",
        r"""
        namespace pyviz_adapters
        {
        void EnbFromRadio(ns3::visualizer::PyViz* viz, uint32_t node, uint32_t device,
                          std::string, ns3::Ptr<ns3::Packet> packet)
        {
            viz->RecordRx(node, device, packet);
        }
        void EnbToRadio(ns3::visualizer::PyViz* viz, uint32_t node, uint32_t device,
                        std::string, ns3::Ptr<ns3::Packet> packet)
        {
            viz->RecordTx(node, device, packet);
        }
        bool ConnectEnb(ns3::visualizer::PyViz* viz, uint32_t node, uint32_t device,
                        const std::string& path, bool fromRadio)
        {
            return ns3::Config::ConnectFailSafe(
                path,
                ns3::MakeBoundCallback(fromRadio ? &EnbFromRadio : &EnbToRadio, viz, node, device));
        }
        }
        """,
    ),
}
_compiled = set()


def type_is_registered(type_name):
    """Return whether a TypeId exists, i.e. whether its module was built."""
    return bool(ns.TypeId.LookupByNameFailSafe(type_name, ns.TypeId()))


def _adapter(name):
    """Compile an adapter once; return the pyviz_adapters namespace, or None."""
    type_name, code = _ADAPTERS[name]
    if not type_is_registered(type_name):
        return None
    if name not in _compiled:
        ns.cppyy.cppdef(code)
        _compiled.add(name)
    return ns.cppyy.gbl.pyviz_adapters


def _type_name(obj):
    return str(obj.GetInstanceTypeId().GetName())


def connect_traces(sim_helper, sources=TRACE_SOURCES):
    """Connect each trace source whose TypeId is registered; return the paths.

    Config path resolution aborts on unknown type names, so the TypeId is
    checked before connecting.
    """
    connected = []
    for type_name, handler, path in sources:
        if type_name is not None and not type_is_registered(type_name):
            continue
        if sim_helper.ConnectTrace(handler, path):
            connected.append(path)
    if sources is TRACE_SOURCES:
        connected.extend(_connect_adapted(sim_helper))
    return connected


def _connect_adapted(sim_helper):
    connected = []
    ipv4 = _adapter("ipv4-drop")
    enb = _adapter("lte-enb")
    for index in range(int(ns.NodeList.GetNNodes())):
        node = ns.NodeList.GetNode(index)
        if ipv4 is not None and node.GetObject[ns.Ipv4L3Protocol]():
            path = "/NodeList/%d/$ns3::Ipv4L3Protocol/Drop" % index
            if ipv4.ConnectIpv4Drop(sim_helper, index, path):
                connected.append(path)
        if enb is not None:
            devices = [
                d
                for d in range(int(node.GetNDevices()))
                if _type_name(node.GetDevice(d)) == "ns3::LteEnbNetDevice"
            ]
            if not devices:
                continue
            for source, from_radio in (("RxFromEnb", True), ("RxFromS1u", False)):
                path = "/NodeList/%d/ApplicationList/*/$ns3::EpcEnbApplication/%s" % (index, source)
                if enb.ConnectEnb(sim_helper, index, devices[0], path, from_radio):
                    connected.append(path)
    return connected
