#
# Copyright (c) 2026 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
#
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

## Config path prefix matching every device of every node.
_DEVICE = "/NodeList/*/DeviceList/*/"

## Trace sources as (required TypeId, generic handler, Config path) tuples.
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
    ("ns3::NrNetDevice", "Rx", _DEVICE + "$ns3::NrNetDevice/Rx"),
    ("ns3::NrNetDevice", "Drop", _DEVICE + "$ns3::NrNetDevice/Drop"),
    (None, "Drop", _DEVICE + "TxQueue/Drop"),
]

# C++ adapters for trace sources whose signature is not void (Ptr<const Packet>).
# Each is compiled only when its module is part of the build.
## C++ adapter sources, keyed by name, as (required TypeId, C++ code) tuples.
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
    # NrNetDevice Tx also passes the destination Address.
    "nr-tx": (
        "ns3::NrNetDevice",
        r"""
        namespace pyviz_adapters
        {
        void NrTx(ns3::visualizer::PyViz* viz, uint32_t node, uint32_t device, std::string,
                  ns3::Ptr<const ns3::Packet> packet, const ns3::Address&)
        {
            viz->RecordTx(node, device, packet);
        }
        bool ConnectNrTx(ns3::visualizer::PyViz* viz, uint32_t node, uint32_t device,
                         const std::string& path)
        {
            return ns3::Config::ConnectFailSafe(path,
                                                ns3::MakeBoundCallback(&NrTx, viz, node, device));
        }
        }
        """,
    ),
    # NR counterpart of "lte-enb": RxFromGnb and RxFromS1u of NrEpcGnbApplication.
    "nr-gnb": (
        "ns3::NrEpcGnbApplication",
        r"""
        namespace pyviz_adapters
        {
        void GnbFromRadio(ns3::visualizer::PyViz* viz, uint32_t node, uint32_t device,
                          std::string, ns3::Ptr<ns3::Packet> packet)
        {
            viz->RecordRx(node, device, packet);
        }
        void GnbToRadio(ns3::visualizer::PyViz* viz, uint32_t node, uint32_t device,
                        std::string, ns3::Ptr<ns3::Packet> packet)
        {
            viz->RecordTx(node, device, packet);
        }
        bool ConnectGnb(ns3::visualizer::PyViz* viz, uint32_t node, uint32_t device,
                        const std::string& path, bool fromRadio)
        {
            return ns3::Config::ConnectFailSafe(
                path,
                ns3::MakeBoundCallback(fromRadio ? &GnbFromRadio : &GnbToRadio, viz, node, device));
        }
        }
        """,
    ),
}
## Names of the adapters already compiled.
_compiled = set()


def type_is_registered(type_name):
    """Return whether a TypeId exists, i.e. whether its module was built.

    @param type_name The TypeId name to look up.
    @return True if the TypeId is registered.
    """
    return bool(ns.TypeId.LookupByNameFailSafe(type_name, ns.TypeId()))


def _adapter(name):
    """Compile an adapter once; return the pyviz_adapters namespace, or None.

    @param name The adapter key in _ADAPTERS.
    @return The cppyy pyviz_adapters namespace, or None if the type is not built.
    """
    type_name, code = _ADAPTERS[name]
    if not type_is_registered(type_name):
        return None
    if name not in _compiled:
        ns.cppyy.cppdef(code)
        _compiled.add(name)
    return ns.cppyy.gbl.pyviz_adapters


def _type_name(obj):
    """Get the TypeId name of an object instance.

    @param obj The ns-3 object.
    @return The instance TypeId name as a string.
    """
    return str(obj.GetInstanceTypeId().GetName())


def connect_traces(sim_helper, sources=TRACE_SOURCES):
    """Connect each trace source whose TypeId is registered; return the paths.

    Config path resolution aborts on unknown type names, so the TypeId is
    checked before connecting.

    @param sim_helper The C++ PyViz helper providing ConnectTrace().
    @param sources Trace source tuples to connect.
    @return List of connected Config paths.
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
    """Connect the trace sources that need compiled C++ adapters.

    @param sim_helper The C++ PyViz helper the adapters record into.
    @return List of connected Config paths.
    """
    connected = []
    ipv4 = _adapter("ipv4-drop")
    enb = _adapter("lte-enb")
    nr_tx = _adapter("nr-tx")
    gnb = _adapter("nr-gnb")
    for index in range(int(ns.NodeList.GetNNodes())):
        node = ns.NodeList.GetNode(index)
        if ipv4 is not None and node.GetObject[ns.Ipv4L3Protocol]():
            path = "/NodeList/%d/$ns3::Ipv4L3Protocol/Drop" % index
            if ipv4.ConnectIpv4Drop(sim_helper, index, path):
                connected.append(path)
        types = [_type_name(node.GetDevice(d)) for d in range(int(node.GetNDevices()))]
        if nr_tx is not None:
            for d, type_name in enumerate(types):
                if type_name in ("ns3::NrUeNetDevice", "ns3::NrGnbNetDevice"):
                    path = "/NodeList/%d/DeviceList/%d/$ns3::NrNetDevice/Tx" % (index, d)
                    if nr_tx.ConnectNrTx(sim_helper, index, d, path):
                        connected.append(path)
        for adapter, device_type, app_type, from_radio_source, prefix in (
            (enb, "ns3::LteEnbNetDevice", "ns3::EpcEnbApplication", "RxFromEnb", "Enb"),
            (gnb, "ns3::NrGnbNetDevice", "ns3::NrEpcGnbApplication", "RxFromGnb", "Gnb"),
        ):
            if adapter is None or device_type not in types:
                continue
            connect = getattr(adapter, "Connect" + prefix)
            for source, from_radio in ((from_radio_source, True), ("RxFromS1u", False)):
                path = "/NodeList/%d/ApplicationList/*/$%s/%s" % (index, app_type, source)
                if connect(sim_helper, index, types.index(device_type), path, from_radio):
                    connected.append(path)
    return connected
