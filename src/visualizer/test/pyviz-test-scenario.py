#
# SPDX-License-Identifier: GPL-2.0-only
#
"""Scenario used by pyviz-web-test.py; not meant to be run by hand.

Topology:
  - n0 --p2p-- n1: n0 forwards traffic for 10.99.0.0/16 to n1, which has no
    route for it, so n1's IPv4 layer drops it (Ipv4L3Protocol Drop trace).
  - n1, n2, n3 on one CSMA channel: drawn as a hub.
  - n4 (access point) with stations n5 and n6 (random walk mobility). The
    stations run on a small battery (20 J) drained by their Wi-Fi radio.
  - n7: no mobility model, so it is placed by the automatic layout. It also
    gets a NetDevice created from Python, which Restart must keep alive.
  - UDP echo between station n6 and CSMA node n3.

Options (command line, or environment variables as defaults, which is how
the tests set them):
  --visualSimulatorImpl   end with Simulator::Run() under VisualSimulatorImpl
                          instead of visualizer.start()  (PYVIZ_TEST_MODE=visual)
  --stopTime=SECONDS      when the simulation stops        (PYVIZ_TEST_STOP, 6)
  --flowMonitor           call visualizer.enable_flow_monitor()  (PYVIZ_TEST_FLOWMON=1)
  --hooks                 register an initialization hook and a plugin
                                                           (PYVIZ_TEST_HOOKS=1)
"""

import os
import sys
from ctypes import c_bool, c_double

from ns import ns

VISUAL = c_bool(os.environ.get("PYVIZ_TEST_MODE", "start") == "visual")
STOP_TIME = c_double(float(os.environ.get("PYVIZ_TEST_STOP", "6")))
FLOW_MONITOR = c_bool(os.environ.get("PYVIZ_TEST_FLOWMON") == "1")
HOOKS = c_bool(os.environ.get("PYVIZ_TEST_HOOKS") == "1")


def parse_command_line(argv):
    cmd = ns.CommandLine(__file__)
    cmd.AddValue(
        "visualSimulatorImpl", "Run under VisualSimulatorImpl instead of visualizer.start()", VISUAL
    )
    cmd.AddValue("stopTime", "Simulation stop time in seconds", STOP_TIME)
    cmd.AddValue("flowMonitor", "Enable the visualizer's flow monitor plots", FLOW_MONITOR)
    cmd.AddValue("hooks", "Register a test initialization hook and plugin", HOOKS)
    cmd.Parse(argv)


def build():
    if VISUAL.value:
        ns.GlobalValue.Bind(
            "SimulatorImplementationType", ns.StringValue("ns3::VisualSimulatorImpl")
        )

    nodes = ns.NodeContainer()
    nodes.Create(8)
    ns.Names.Add("source", nodes.Get(0))

    # Positions for everything except n7.
    mobility = ns.MobilityHelper()
    positions = ns.ListPositionAllocator()
    for x, y in ((0, 0), (40, 0), (80, -20), (80, 20), (120, 0), (130, 15), (130, -15)):
        positions.Add(ns.Vector(x, y, 0))
    mobility.SetPositionAllocator(positions)
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel")
    fixed = ns.NodeContainer()
    for index in range(5):
        fixed.Add(nodes.Get(index))
    mobility.Install(fixed)
    mobility.SetMobilityModel(
        "ns3::RandomWalk2dMobilityModel",
        "Bounds",
        ns.RectangleValue(ns.Rectangle(110, 150, -30, 30)),
    )
    stations = ns.NodeContainer()
    stations.Add(nodes.Get(5))
    stations.Add(nodes.Get(6))
    mobility.Install(stations)

    p2p = ns.PointToPointHelper()
    p2p.SetDeviceAttribute("DataRate", ns.StringValue("1Mbps"))
    p2p.SetChannelAttribute("Delay", ns.StringValue("2ms"))
    p2p_devices = p2p.Install(nodes.Get(0), nodes.Get(1))

    csma = ns.CsmaHelper()
    csma_nodes = ns.NodeContainer()
    for index in (1, 2, 3):
        csma_nodes.Add(nodes.Get(index))
    csma_devices = csma.Install(csma_nodes)

    wifi_channel = ns.YansWifiChannelHelper.Default()
    phy = ns.YansWifiPhyHelper()
    phy.SetChannel(wifi_channel.Create())
    wifi = ns.WifiHelper()
    mac = ns.WifiMacHelper()
    ssid = ns.Ssid("pyviz-test")
    mac.SetType("ns3::StaWifiMac", "Ssid", ns.SsidValue(ssid))
    sta_devices = wifi.Install(phy, mac, stations)
    mac.SetType("ns3::ApWifiMac", "Ssid", ns.SsidValue(ssid))
    ap_devices = wifi.Install(phy, mac, nodes.Get(4))
    ap_link = p2p.Install(nodes.Get(3), nodes.Get(4))

    battery = ns.BasicEnergySourceHelper()
    battery.Set("BasicEnergySourceInitialEnergyJ", ns.DoubleValue(20))
    energy_sources = battery.Install(stations)
    ns.WifiRadioEnergyModelHelper().Install(sta_devices, energy_sources)

    # Created from Python and owned by its wrapper.
    loose_device = ns.CsmaNetDevice()
    loose_device.SetAddress(ns.Mac48Address.Allocate().ConvertTo())
    nodes.Get(7).AddDevice(loose_device)

    stack = ns.InternetStackHelper()
    stack.Install(nodes)
    address = ns.Ipv4AddressHelper()
    address.SetBase(ns.Ipv4Address("10.1.1.0"), ns.Ipv4Mask("255.255.255.0"))
    p2p_interfaces = address.Assign(p2p_devices)
    address.SetBase(ns.Ipv4Address("10.1.2.0"), ns.Ipv4Mask("255.255.255.0"))
    csma_interfaces = address.Assign(csma_devices)
    address.SetBase(ns.Ipv4Address("10.1.3.0"), ns.Ipv4Mask("255.255.255.0"))
    address.Assign(ap_link)
    address.SetBase(ns.Ipv4Address("10.1.4.0"), ns.Ipv4Mask("255.255.255.0"))
    address.Assign(sta_devices)
    address.Assign(ap_devices)
    # n0 sends 10.99.0.0/16 to n1, which has no route for it and drops it.
    static_routing = ns.Ipv4StaticRoutingHelper()
    n0_routes = static_routing.GetStaticRouting(nodes.Get(0).GetObject[ns.Ipv4]())
    n0_routes.AddNetworkRouteTo(
        ns.Ipv4Address("10.99.0.0"), ns.Ipv4Mask("255.255.0.0"), p2p_interfaces.GetAddress(1), 1
    )

    unroutable = ns.InetSocketAddress(ns.Ipv4Address("10.99.0.1"), 9)
    onoff = ns.OnOffHelper("ns3::UdpSocketFactory", unroutable.ConvertTo())
    onoff.SetConstantRate(ns.DataRate("200kbps"), 500)
    flood = onoff.Install(nodes.Get(0))
    flood.Start(ns.Seconds(0.5))
    flood.Stop(ns.Seconds(STOP_TIME.value))

    server = ns.UdpEchoServerHelper(9)
    server_apps = server.Install(nodes.Get(3))
    server_apps.Start(ns.Seconds(0.1))
    client = ns.UdpEchoClientHelper(csma_interfaces.GetAddress(2).ConvertTo(), 9)
    client.SetAttribute("MaxPackets", ns.UintegerValue(100000))
    client.SetAttribute("Interval", ns.TimeValue(ns.MilliSeconds(50)))
    client.SetAttribute("PacketSize", ns.UintegerValue(512))
    client_apps = client.Install(nodes.Get(6))
    client_apps.Start(ns.Seconds(0.5))
    client_apps.Stop(ns.Seconds(STOP_TIME.value))

    ns.Ipv4GlobalRoutingHelper.PopulateRoutingTables()
    ns.Simulator.Stop(ns.Seconds(STOP_TIME.value))
    # Returned so the caller's frame keeps them alive like a normal scenario.
    return nodes, loose_device, flood, server_apps, client_apps, energy_sources


def register_test_extensions():
    import visualizer

    if FLOW_MONITOR.value:
        visualizer.enable_flow_monitor()
    if not HOOKS.value:
        return

    def hook(viz, color):
        viz.set_node_color(0, color)
        viz.set_node_label(1, "router")
        viz.set_node_size(2, 2.5)
        viz.set_node_icon(
            3, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pyviz-test-icon.svg")
        )

    def plugin(viz):
        state = {"updates": 0, "hold": False, "actions": []}

        def periodic(viz):
            state["updates"] += 1

        def build_node(viz, node_index):
            return {"sections": [{"title": "Echo", "items": [["node", str(node_index)]]}]}

        def build_global(viz):
            return {
                "sections": [
                    {"title": "Counters", "items": [["updates", str(state["updates"])]]},
                    {
                        "title": "Actions",
                        "columns": ["value"],
                        "rows": [[a] for a in state["actions"]],
                    },
                ],
                "controls": [
                    {
                        "name": "mode",
                        "kind": "radio",
                        "label": "Mode",
                        "options": ["a", "b"],
                        "value": "a",
                    },
                    {"name": "hold", "kind": "button", "label": "Toggle hold"},
                ],
            }

        def action(viz, values):
            if values.get("hold"):
                state["hold"] = not state["hold"]
            if values.get("mode"):
                state["actions"].append(values["mode"])

        viz.add_periodic_update(periodic)
        viz.add_step_guard(lambda viz: state["hold"])
        viz.add_overlay(
            lambda viz: [{"source": 0, "target": 7, "class": "test-overlay", "title": "test"}]
        )
        viz.add_tooltip_info(lambda viz, node: [["Test info", "node %d" % node]])
        viz.add_node_panel(
            "test-node", "Test node panel", build_node, applies=lambda viz, n: n == 0
        )
        viz.add_panel("test-global", "Test global panel", build_global, action)

    visualizer.add_initialization_hook(hook, "#ff00ff")
    visualizer.register_plugin(plugin)
    visualizer.set_bounds(-10, -40, 160, 40)


def main(argv):
    parse_command_line(argv)
    objects = build()  # noqa: F841 (kept alive while the visualizer runs)
    register_test_extensions()
    if VISUAL.value:
        ns.Simulator.Run()
        ns.Simulator.Destroy()
    else:
        import visualizer

        visualizer.start()


if __name__ == "__main__":
    main(sys.argv)
