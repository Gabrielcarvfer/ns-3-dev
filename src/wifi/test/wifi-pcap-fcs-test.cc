/*
 * Copyright (c) 2026 Centre Tecnològic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
 */

#include "ns3/ampdu-subframe-header.h"
#include "ns3/config.h"
#include "ns3/mobility-helper.h"
#include "ns3/node-container.h"
#include "ns3/packet-socket-client.h"
#include "ns3/packet-socket-helper.h"
#include "ns3/packet.h"
#include "ns3/pcap-file.h"
#include "ns3/simulator.h"
#include "ns3/ssid.h"
#include "ns3/string.h"
#include "ns3/uinteger.h"
#include "ns3/test.h"
#include "ns3/wifi-helper.h"
#include "ns3/wifi-mac-helper.h"
#include "ns3/yans-wifi-helper.h"

#include <vector>

using namespace ns3;

/**
 * @ingroup wifi-test
 *
 * @brief Test that pcap traces do not include the FCS (see issue #300)
 *
 * The frames passed to the monitor sniffer traces carry a WifiMacTrailer
 * whose four octets are zero, since the simulator does not compute a real
 * FCS. The pcap traces must not include those four octets (nor set the
 * FCS-included radiotap frame flag), so that packet analyzers do not check
 * the always-zero FCS. The frames of A-MPDUs must not include the A-MPDU
 * delimiter and padding either, which follow the FCS.
 *
 * The test runs a short infrastructure network simulation with pcap tracing
 * enabled and checks, for every frame, that the size written to the pcap
 * file is four bytes smaller than the size of the sniffed frame and that,
 * in radiotap mode, the FCS-included frame flag is not set.
 */
class WifiPcapFcsTest : public TestCase
{
  public:
    /**
     * @brief Constructor.
     * @param radiotap Whether to use the radiotap data link type (or plain 802.11).
     * @param aggregation Whether to use 802.11n and send A-MPDUs (or 802.11a).
     */
    WifiPcapFcsTest(bool radiotap, bool aggregation)
        : TestCase(std::string("Pcap traces do not include the FCS (") +
                   (radiotap ? "radiotap" : "IEEE 802.11") + " data link type" +
                   (aggregation ? ", A-MPDUs" : "") + ")"),
          m_radiotap(radiotap),
          m_aggregation(aggregation),
          m_aggregated(0)
    {
    }

  private:
    void DoRun() override;

    /**
     * Run the simulation, keeping every helper and container in a local
     * scope, so that once this method returns the pcap file wrapper is
     * destroyed and the pcap file is flushed and closed.
     */
    void RunSimulation();

    /**
     * Record the size of a sniffed frame (transmit direction).
     *
     * @param packet The sniffed packet.
     * @param channelFreqMhz The channel frequency.
     * @param txVector The TX vector.
     * @param aMpdu The A-MPDU information.
     * @param staId The STA identifier.
     */
    void SnifferTx(Ptr<const Packet> packet,
                   uint16_t channelFreqMhz,
                   WifiTxVector txVector,
                   MpduInfo aMpdu,
                   uint16_t staId);

    /**
     * Record the size of a sniffed frame (receive direction).
     *
     * @param packet The sniffed packet.
     * @param channelFreqMhz The channel frequency.
     * @param txVector The TX vector.
     * @param aMpdu The A-MPDU information.
     * @param signalNoise The signal and noise powers.
     * @param staId The STA identifier.
     */
    void SnifferRx(Ptr<const Packet> packet,
                   uint16_t channelFreqMhz,
                   WifiTxVector txVector,
                   MpduInfo aMpdu,
                   SignalNoiseDbm signalNoise,
                   uint16_t staId);

    /**
     * Record the expected size of a sniffed frame in the pcap file: that of the
     * MPDU, without A-MPDU delimiter and padding, nor FCS.
     *
     * @param packet The sniffed packet.
     * @param txVector The TX vector.
     */
    void Record(Ptr<const Packet> packet, const WifiTxVector& txVector);

    bool m_radiotap;                    //!< True to use the radiotap data link type.
    bool m_aggregation;                 //!< True to use 802.11n and send A-MPDUs.
    uint32_t m_aggregated;              //!< Sniffed frames that were part of an A-MPDU.
    std::string m_pcapPrefix;           //!< Prefix of the pcap file name.
    std::vector<uint32_t> m_frameSizes; //!< Expected sizes of the frames in the pcap file.
};

void
WifiPcapFcsTest::Record(Ptr<const Packet> packet, const WifiTxVector& txVector)
{
    if (txVector.IsAggregation())
    {
        AmpduSubframeHeader hdr;
        packet->PeekHeader(hdr);
        m_aggregated++;
        m_frameSizes.push_back(hdr.GetLength() - 4);
        return;
    }
    m_frameSizes.push_back(packet->GetSize() - 4);
}

void
WifiPcapFcsTest::SnifferTx(Ptr<const Packet> packet,
                           uint16_t channelFreqMhz,
                           WifiTxVector txVector,
                           MpduInfo aMpdu,
                           uint16_t staId)
{
    Record(packet, txVector);
}

void
WifiPcapFcsTest::SnifferRx(Ptr<const Packet> packet,
                           uint16_t channelFreqMhz,
                           WifiTxVector txVector,
                           MpduInfo aMpdu,
                           SignalNoiseDbm signalNoise,
                           uint16_t staId)
{
    Record(packet, txVector);
}

void
WifiPcapFcsTest::RunSimulation()
{
    NodeContainer nodes;
    nodes.Create(2);
    MobilityHelper mobility;
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(nodes);

    YansWifiPhyHelper phy;
    YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
    phy.SetChannel(channel.Create());
    phy.SetPcapDataLinkType(m_radiotap ? WifiPhyHelper::DLT_IEEE802_11_RADIO
                                       : WifiPhyHelper::DLT_IEEE802_11);

    WifiHelper wifi;
    if (m_aggregation)
    {
        wifi.SetStandard(WIFI_STANDARD_80211n);
    }
    WifiMacHelper mac;
    Ssid ssid("pcap-fcs-test");
    mac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid));
    NetDeviceContainer apDevice = wifi.Install(phy, mac, nodes.Get(0));
    mac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid));
    NetDeviceContainer staDevice = wifi.Install(phy, mac, nodes.Get(1));

    if (m_aggregation)
    {
        // packets queued at once, sent in A-MPDUs
        PacketSocketHelper packetSocket;
        packetSocket.Install(nodes);
        PacketSocketAddress socket;
        socket.SetSingleDevice(apDevice.Get(0)->GetIfIndex());
        socket.SetPhysicalAddress(staDevice.Get(0)->GetAddress());
        socket.SetProtocol(1);
        auto client = CreateObject<PacketSocketClient>();
        client->SetAttribute("PacketSize", UintegerValue(500));
        client->SetAttribute("MaxPackets", UintegerValue(20));
        client->SetAttribute("Interval", TimeValue(MicroSeconds(0)));
        client->SetRemote(socket);
        nodes.Get(0)->AddApplication(client);
        client->SetStartTime(Seconds(0.5)); // once associated
    }

    // Trace pcap on the AP device only, and record the sniffed frame sizes
    // through the same traces feeding the pcap file, in the same order
    m_pcapPrefix = CreateTempDirFilename("wifi-pcap-fcs");
    phy.EnablePcap(m_pcapPrefix, apDevice);
    Config::ConnectWithoutContext(
        "/NodeList/0/DeviceList/*/$ns3::WifiNetDevice/Phy/MonitorSnifferTx",
        MakeCallback(&WifiPcapFcsTest::SnifferTx, this));
    Config::ConnectWithoutContext(
        "/NodeList/0/DeviceList/*/$ns3::WifiNetDevice/Phy/MonitorSnifferRx",
        MakeCallback(&WifiPcapFcsTest::SnifferRx, this));

    Simulator::Stop(Seconds(m_aggregation ? 1 : 0.2));
    Simulator::Run();
    Simulator::Destroy();
}

void
WifiPcapFcsTest::DoRun()
{
    RunSimulation();

    NS_TEST_ASSERT_MSG_GT(m_frameSizes.size(), 0, "No frame was sniffed");
    if (m_aggregation)
    {
        NS_TEST_ASSERT_MSG_GT(m_aggregated, 0, "No A-MPDU was sniffed");
    }

    PcapFile pcap;
    pcap.Open(m_pcapPrefix + "-0-0.pcap", std::ios::in);
    NS_TEST_ASSERT_MSG_EQ(pcap.Fail(), false, "Could not open the pcap file");

    uint8_t data[65536];
    uint32_t tsSec;
    uint32_t tsUsec;
    uint32_t inclLen;
    uint32_t origLen;
    uint32_t readLen;
    for (const auto& frameSize : m_frameSizes)
    {
        pcap.Read(data, sizeof(data), tsSec, tsUsec, inclLen, origLen, readLen);
        NS_TEST_ASSERT_MSG_EQ(pcap.Fail(), false, "Missing frame in the pcap file");

        uint32_t headerLen = 0;
        if (m_radiotap)
        {
            // Radiotap header length is at bytes 2-3 (little endian)
            headerLen = data[2] | (data[3] << 8);

            // The frame flags field follows the (8 bytes, aligned to 8 bytes)
            // TSFT field, which follows the presence bitmaps (bit 31 of a
            // presence word indicates that another presence word follows)
            uint32_t offset = 4;
            uint32_t present;
            do
            {
                present = data[offset] | (data[offset + 1] << 8) | (data[offset + 2] << 16) |
                          (data[offset + 3] << 24);
                offset += 4;
            } while (present & 0x80000000);
            offset = ((offset + 7) / 8) * 8 + 8;
            NS_TEST_ASSERT_MSG_EQ((data[offset] & 0x10),
                                  0,
                                  "The FCS-included radiotap frame flag is set");
        }

        NS_TEST_ASSERT_MSG_EQ(inclLen - headerLen,
                              frameSize,
                              "The frame in the pcap file is not the MPDU without FCS");
    }
}

/**
 * @ingroup wifi-test
 *
 * @brief Wifi pcap FCS TestSuite
 */
class WifiPcapFcsTestSuite : public TestSuite
{
  public:
    WifiPcapFcsTestSuite()
        : TestSuite("wifi-pcap-fcs", Type::UNIT)
    {
        AddTestCase(new WifiPcapFcsTest(true, false), TestCase::Duration::QUICK);
        AddTestCase(new WifiPcapFcsTest(false, false), TestCase::Duration::QUICK);
        AddTestCase(new WifiPcapFcsTest(true, true), TestCase::Duration::QUICK);
        AddTestCase(new WifiPcapFcsTest(false, true), TestCase::Duration::QUICK);
    }
};

/// Static variable for test initialization
static WifiPcapFcsTestSuite g_wifiPcapFcsTestSuite;
