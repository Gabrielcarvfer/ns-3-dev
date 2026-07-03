/*
 * Copyright (c) 2026 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
 */

#include "ns3/boolean.h"
#include "ns3/config.h"
#include "ns3/double.h"
#include "ns3/ht-phy.h"
#include "ns3/llc-snap-header.h"
#include "ns3/mobility-helper.h"
#include "ns3/packet-socket-client.h"
#include "ns3/packet-socket-helper.h"
#include "ns3/packet-socket-server.h"
#include "ns3/packet.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/ssid.h"
#include "ns3/string.h"
#include "ns3/test.h"
#include "ns3/uinteger.h"
#include "ns3/wifi-mac.h"
#include "ns3/wifi-phy.h"
#include "ns3/wifi-psdu.h"
#include "ns3/wifi-net-device.h"
#include "ns3/yans-wifi-helper.h"

#include <tuple>
#include <vector>

using namespace ns3;

/**
 * @ingroup wifi-test
 * @ingroup tests
 *
 * @brief Regression test for @issueid{1199}.
 *
 * An HT (802.11n) STA with A-MPDU aggregation disabled (BE_MaxAmpduSize=0)
 * and a small FragmentationThreshold sends MSDUs larger than the threshold to
 * its AP, with a BE_BlockAckThreshold that establishes a Block Ack agreement.
 * Before the fix, acknowledging a fragment sent under the agreement dequeued
 * the MSDU twice and aborted the simulation (reported with VHT, which always
 * establishes an agreement, so that no MSDU is ever sent without one).
 *
 * MSDUs are fragmented while there is no agreement, but an MSDU sent under a
 * Block Ack agreement is not: the recipient's reordering buffer cannot hold
 * fragments (Linux tears the agreement down when it receives one). The test
 * checks that all the MSDUs are delivered, that the agreement is established,
 * and that no fragment is transmitted while it is.
 *
 * Deterministic: fixed RngSeed/RngRun, ConstantPositionMobilityModel,
 * IdealWifiManager (no random rate selection), short fixed-duration run.
 */
class WifiFragmentationBlockAckTestCase : public TestCase
{
  public:
    WifiFragmentationBlockAckTestCase();
    ~WifiFragmentationBlockAckTestCase() override;

    void DoRun() override;

  private:
    /**
     * Callback invoked when the server application receives a packet.
     *
     * @param context the trace context
     * @param p the received packet
     * @param addr the address of the sender
     */
    void L7Receive(std::string context, Ptr<const Packet> p, const Address& addr);

    /**
     * Callback invoked when the STA PHY starts transmitting a PSDU.
     *
     * @param psduMap the PSDU map
     * @param txVector the TX vector
     * @param txPowerW the TX power in Watts
     */
    void StaTx(WifiConstPsduMap psduMap, WifiTxVector txVector, double txPowerW);

    uint32_t m_received;     ///< number of packets received by the server application
    Ptr<WifiMac> m_staMac;   ///< the MAC of the STA
    Mac48Address m_apAddr;   ///< the address of the AP
    uint32_t m_underBa;      ///< QoS data frames sent by the STA under the agreement
    uint32_t m_fragUnderBa;  ///< fragments sent by the STA under the agreement
    uint32_t m_fragments;    ///< fragments sent by the STA
};

WifiFragmentationBlockAckTestCase::WifiFragmentationBlockAckTestCase()
    : TestCase("Issue #1199: A-MPDU disabled, no fragmentation under a Block Ack agreement"),
      m_received(0),
      m_underBa(0),
      m_fragUnderBa(0),
      m_fragments(0)
{
}

WifiFragmentationBlockAckTestCase::~WifiFragmentationBlockAckTestCase()
{
}

void
WifiFragmentationBlockAckTestCase::L7Receive(std::string context,
                                             Ptr<const Packet> p,
                                             const Address& addr)
{
    m_received++;
}

void
WifiFragmentationBlockAckTestCase::StaTx(WifiConstPsduMap psduMap,
                                         WifiTxVector txVector,
                                         double txPowerW)
{
    for (const auto& [staId, psdu] : psduMap)
    {
        for (const auto& mpdu : *PeekPointer(psdu))
        {
            const auto& hdr = mpdu->GetHeader();
            if (!hdr.IsQosData())
            {
                continue;
            }
            const auto isFragment = hdr.IsMoreFragments() || hdr.GetFragmentNumber() > 0;
            m_fragments += isFragment ? 1 : 0;
            if (m_staMac->GetBaAgreementEstablishedAsOriginator(m_apAddr, hdr.GetQosTid()))
            {
                m_underBa++;
                m_fragUnderBa += isFragment ? 1 : 0;
            }
        }
    }
}

void
WifiFragmentationBlockAckTestCase::DoRun()
{
    // Deterministic RNG configuration.
    RngSeedManager::SetSeed(1);
    RngSeedManager::SetRun(1);

    NodeContainer wifiStaNode;
    wifiStaNode.Create(1);

    NodeContainer wifiApNode;
    wifiApNode.Create(1);

    YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
    YansWifiPhyHelper phy;
    phy.SetChannel(channel.Create());

    WifiHelper wifi;
    // HT (802.11n): unlike VHT, a Block Ack agreement is only established once
    // BE_BlockAckThreshold MSDUs are queued, so MSDUs are first sent without one.
    wifi.SetStandard(WIFI_STANDARD_80211n);
    // IdealWifiManager keeps rate selection deterministic and, together with the
    // small FragmentationThreshold, guarantees the data PSDU is fragmented.
    wifi.SetRemoteStationManager("ns3::IdealWifiManager",
                                 "FragmentationThreshold",
                                 UintegerValue(400));

    Ssid ssid = Ssid("ns-3-ssid");
    WifiMacHelper mac;

    // A-MPDU aggregation disabled (BE_MaxAmpduSize=0) and A-MSDU disabled, so
    // each MSDU is transmitted as a single MPDU and is fragmented when larger
    // than the fragmentation threshold. BE_BlockAckThreshold=2 forces a Block
    // Ack agreement to be established once more than one frame is queued.
    mac.SetType("ns3::StaWifiMac",
                "Ssid",
                SsidValue(ssid),
                "BE_MaxAmsduSize",
                UintegerValue(0),
                "BE_MaxAmpduSize",
                UintegerValue(0),
                "BE_BlockAckThreshold",
                UintegerValue(2),
                "ActiveProbing",
                BooleanValue(false));

    NetDeviceContainer staDevices = wifi.Install(phy, mac, wifiStaNode);

    mac.SetType("ns3::ApWifiMac",
                "Ssid",
                SsidValue(ssid),
                "BE_MaxAmsduSize",
                UintegerValue(0),
                "BE_MaxAmpduSize",
                UintegerValue(0),
                "BeaconGeneration",
                BooleanValue(true));

    NetDeviceContainer apDevices = wifi.Install(phy, mac, wifiApNode);

    MobilityHelper mobility;
    Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator>();
    positionAlloc->Add(Vector(0.0, 0.0, 0.0));
    positionAlloc->Add(Vector(1.0, 0.0, 0.0));
    mobility.SetPositionAllocator(positionAlloc);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(wifiApNode);
    mobility.Install(wifiStaNode);

    Ptr<WifiNetDevice> apDevice = DynamicCast<WifiNetDevice>(apDevices.Get(0));
    Ptr<WifiNetDevice> staDevice = DynamicCast<WifiNetDevice>(staDevices.Get(0));

    PacketSocketAddress socket;
    socket.SetSingleDevice(staDevice->GetIfIndex());
    socket.SetPhysicalAddress(apDevice->GetAddress());
    socket.SetProtocol(1);

    PacketSocketHelper packetSocket;
    packetSocket.Install(wifiStaNode);
    packetSocket.Install(wifiApNode);

    // A single large packet first, larger than the fragmentation threshold: alone
    // in the queue, it does not trigger a Block Ack agreement and is fragmented.
    Ptr<PacketSocketClient> single = CreateObject<PacketSocketClient>();
    single->SetAttribute("PacketSize", UintegerValue(1400));
    single->SetAttribute("MaxPackets", UintegerValue(1));
    single->SetRemote(socket);
    wifiStaNode.Get(0)->AddApplication(single);
    single->SetStartTime(Seconds(1.0));
    single->SetStopTime(Seconds(3.0));

    // Then several large packets at once: having more than one packet queued
    // triggers the Block Ack agreement (BE_BlockAckThreshold).
    Ptr<PacketSocketClient> client = CreateObject<PacketSocketClient>();
    client->SetAttribute("PacketSize", UintegerValue(1400));
    client->SetAttribute("MaxPackets", UintegerValue(5));
    client->SetAttribute("Interval", TimeValue(MicroSeconds(0)));
    client->SetRemote(socket);
    wifiStaNode.Get(0)->AddApplication(client);
    client->SetStartTime(Seconds(2.0));
    client->SetStopTime(Seconds(3.0));

    Ptr<PacketSocketServer> server = CreateObject<PacketSocketServer>();
    server->SetLocal(socket);
    wifiApNode.Get(0)->AddApplication(server);
    server->SetStartTime(Seconds(0.0));
    server->SetStopTime(Seconds(4.0));

    Config::Connect("/NodeList/*/ApplicationList/*/$ns3::PacketSocketServer/Rx",
                    MakeCallback(&WifiFragmentationBlockAckTestCase::L7Receive, this));

    m_staMac = staDevice->GetMac();
    m_apAddr = apDevice->GetMac()->GetAddress();
    staDevice->GetPhy()->TraceConnectWithoutContext(
        "PhyTxPsduBegin",
        MakeCallback(&WifiFragmentationBlockAckTestCase::StaTx, this));

    Simulator::Stop(Seconds(4.0));

    Simulator::Run();
    Simulator::Destroy();

    // If we get here, the double-dequeue abort did not happen.
    NS_TEST_EXPECT_MSG_EQ(m_received, 6, "Expected all the packets to be delivered");
    NS_TEST_EXPECT_MSG_GT(m_fragments, 0, "Expected the first MSDU to be fragmented");
    NS_TEST_EXPECT_MSG_GT(m_underBa, 0, "Expected a Block Ack agreement to be established");
    NS_TEST_EXPECT_MSG_EQ(m_fragUnderBa,
                          0,
                          "Expected no fragment to be sent under the Block Ack agreement");
}

/**
 * @ingroup wifi-test
 * @ingroup tests
 *
 * @brief Regression test for @issueid{1199}, recipient side.
 *
 * Once a Block Ack agreement is established between an HT STA (originator) and its AP
 * (recipient), the STA sends an MSDU in two fragments under the agreement, as a device
 * that does not follow ns-3 (or Linux) in never fragmenting under an agreement could,
 * then three more MSDUs with the next sequence numbers. The recipient window tracks MSDUs
 * by sequence number and cannot hold fragments: the AP must terminate the agreement (as
 * Linux does), deliver the reassembled MSDU, and deliver the MSDUs sent afterwards,
 * rather than stall its window on the sequence number of the fragmented MSDU.
 */
class WifiFragmentUnderBlockAckRecipientTestCase : public TestCase
{
  public:
    WifiFragmentUnderBlockAckRecipientTestCase();

  private:
    void DoSetup() override;
    void DoRun() override;

    /**
     * Send an MSDU in two fragments, then three whole MSDUs, under the agreement, on the
     * PHY of the STA.
     */
    void SendFragments();

    /**
     * Callback invoked when the STA PHY starts transmitting a PSDU.
     *
     * @param psduMap the PSDU map
     * @param txVector the TX vector
     * @param txPowerW the TX power in Watts
     */
    void StaTx(WifiConstPsduMap psduMap, WifiTxVector txVector, double txPowerW);

    /**
     * Callback invoked when the AP application receives a packet.
     *
     * @param packet the packet
     * @param from the sender
     */
    void ApRx(Ptr<const Packet> packet, const Address& from);

    NodeContainer m_nodes;                        ///< AP, STA
    Ptr<WifiNetDevice> m_ap;                      ///< the AP device
    Ptr<WifiNetDevice> m_sta;                     ///< the STA device
    uint16_t m_lastSeqNo{0};                      ///< last sequence number sent by the STA
    uint32_t m_received{0};                       ///< MSDUs received by the AP application
    bool m_agreementAfterFragments{true};         ///< agreement still established afterwards
    static constexpr uint32_t m_packetSize{1000}; ///< MSDU payload size
};

WifiFragmentUnderBlockAckRecipientTestCase::WifiFragmentUnderBlockAckRecipientTestCase()
    : TestCase("Issue #1199: a recipient terminates its agreement on a fragment")
{
}

void
WifiFragmentUnderBlockAckRecipientTestCase::StaTx(WifiConstPsduMap psduMap,
                                                  WifiTxVector txVector,
                                                  double txPowerW)
{
    for (const auto& [staId, psdu] : psduMap)
    {
        for (const auto& mpdu : *PeekPointer(psdu))
        {
            if (mpdu->GetHeader().IsQosData())
            {
                m_lastSeqNo = mpdu->GetHeader().GetSequenceNumber();
            }
        }
    }
}

void
WifiFragmentUnderBlockAckRecipientTestCase::ApRx(Ptr<const Packet> packet, const Address& from)
{
    ++m_received;
}

void
WifiFragmentUnderBlockAckRecipientTestCase::SendFragments()
{
    NS_TEST_ASSERT_MSG_EQ(
        m_ap->GetMac()->GetBaAgreementEstablishedAsRecipient(m_sta->GetMac()->GetAddress(), 0).has_value(),
        true,
        "No Block Ack agreement before the fragments");

    // MSDUs for the packet socket of the AP (protocol 1)
    auto msdu = [this]() {
        auto packet = Create<Packet>(m_packetSize);
        LlcSnapHeader llc;
        llc.SetType(1);
        packet->AddHeader(llc);
        return packet;
    };
    const auto fragmented = msdu();
    const auto half = fragmented->GetSize() / 2;
    // sequence number, fragment number, more fragments, packet
    std::vector<std::tuple<uint16_t, uint8_t, bool, Ptr<Packet>>> mpdus{
        {1, 0, true, fragmented->CreateFragment(0, half)},
        {1, 1, false, fragmented->CreateFragment(half, fragmented->GetSize() - half)}};
    for (uint16_t i = 2; i <= 4; ++i)
    {
        mpdus.emplace_back(i, 0, false, msdu());
    }
    const WifiTxVector txVector(HtPhy::GetHtMcs0(),
                                WIFI_MIN_TX_PWR_LEVEL,
                                WIFI_PREAMBLE_HT_MF,
                                NanoSeconds(800),
                                1,
                                1,
                                0,
                                MHz_u{20},
                                false);
    for (std::size_t i = 0; i < mpdus.size(); ++i)
    {
        const auto& [seqOffset, fragNo, moreFragments, packet] = mpdus[i];
        WifiMacHeader hdr(WIFI_MAC_QOSDATA);
        hdr.SetAddr1(m_ap->GetMac()->GetAddress());
        hdr.SetAddr2(m_sta->GetMac()->GetAddress());
        hdr.SetAddr3(m_ap->GetMac()->GetAddress());
        hdr.SetDsTo();
        hdr.SetDsNotFrom();
        hdr.SetQosTid(0);
        hdr.SetQosAckPolicy(WifiMacHeader::NORMAL_ACK);
        // the sequence numbers that follow those the STA used
        hdr.SetSequenceNumber((m_lastSeqNo + seqOffset) % 4096);
        hdr.SetFragmentNumber(fragNo);
        moreFragments ? hdr.SetMoreFragments() : hdr.SetNoMoreFragments();
        const auto psdu = Create<WifiPsdu>(packet, hdr);
        Simulator::Schedule(MilliSeconds(10 * i), [this, psdu, txVector]() {
            m_sta->GetPhy()->Send(psdu, txVector);
        });
    }
    Simulator::Schedule(MilliSeconds(100), [this]() {
        m_agreementAfterFragments =
            m_ap->GetMac()->GetBaAgreementEstablishedAsRecipient(m_sta->GetMac()->GetAddress(), 0).has_value();
    });
}

void
WifiFragmentUnderBlockAckRecipientTestCase::DoSetup()
{
    RngSeedManager::SetSeed(1);
    RngSeedManager::SetRun(1);

    m_nodes.Create(2); // AP, STA
    MobilityHelper mobility;
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(m_nodes);

    YansWifiPhyHelper phy;
    phy.SetChannel(YansWifiChannelHelper::Default().Create());
    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211n);
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                 "DataMode",
                                 StringValue("HtMcs0"),
                                 "ControlMode",
                                 StringValue("HtMcs0"));
    WifiMacHelper mac;
    const Ssid ssid("fragment-under-ba");
    // single MPDUs (no A-MPDU), an agreement once two MSDUs are queued
    mac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid), "BE_MaxAmpduSize", UintegerValue(0));
    m_ap = DynamicCast<WifiNetDevice>(wifi.Install(phy, mac, m_nodes.Get(0)).Get(0));
    mac.SetType("ns3::StaWifiMac",
                "Ssid",
                SsidValue(ssid),
                "BE_MaxAmpduSize",
                UintegerValue(0),
                "BE_BlockAckThreshold",
                UintegerValue(2));
    m_sta = DynamicCast<WifiNetDevice>(wifi.Install(phy, mac, m_nodes.Get(1)).Get(0));

    PacketSocketHelper packetSocket;
    packetSocket.Install(m_nodes);
    PacketSocketAddress socket;
    socket.SetSingleDevice(m_sta->GetIfIndex());
    socket.SetPhysicalAddress(m_ap->GetAddress());
    socket.SetProtocol(1);

    // five MSDUs before the fragments: they establish the agreement
    auto client = CreateObject<PacketSocketClient>();
    client->SetAttribute("PacketSize", UintegerValue(m_packetSize));
    client->SetAttribute("MaxPackets", UintegerValue(5));
    client->SetAttribute("Interval", TimeValue(MicroSeconds(0)));
    client->SetRemote(socket);
    m_nodes.Get(1)->AddApplication(client);
    client->SetStartTime(Seconds(1));
    auto server = CreateObject<PacketSocketServer>();
    server->SetLocal(socket);
    m_nodes.Get(0)->AddApplication(server);
    server->TraceConnectWithoutContext(
        "Rx",
        MakeCallback(&WifiFragmentUnderBlockAckRecipientTestCase::ApRx, this));

    m_sta->GetPhy()->TraceConnectWithoutContext(
        "PhyTxPsduBegin",
        MakeCallback(&WifiFragmentUnderBlockAckRecipientTestCase::StaTx, this));
    Simulator::Schedule(Seconds(2),
                        &WifiFragmentUnderBlockAckRecipientTestCase::SendFragments,
                        this);
}

void
WifiFragmentUnderBlockAckRecipientTestCase::DoRun()
{
    Simulator::Stop(Seconds(4));
    Simulator::Run();
    Simulator::Destroy();

    NS_TEST_EXPECT_MSG_EQ(m_agreementAfterFragments,
                          false,
                          "The recipient did not terminate the agreement on a fragment");
    NS_TEST_EXPECT_MSG_EQ(m_received,
                          9,
                          "Expected the 5 MSDUs, the fragmented one and the 3 after it");
}

/**
 * @ingroup wifi-test
 * @ingroup tests
 *
 * @brief Test suite for the @issueid{1199} fragmentation + Block Ack
 * double-dequeue regression. See WifiFragmentationBlockAckTestCase.
 */
class WifiFragmentationBlockAckTestSuite : public TestSuite
{
  public:
    WifiFragmentationBlockAckTestSuite();
};

WifiFragmentationBlockAckTestSuite::WifiFragmentationBlockAckTestSuite()
    : TestSuite("wifi-fragmentation-blockack", Type::UNIT)
{
    AddTestCase(new WifiFragmentationBlockAckTestCase, TestCase::Duration::QUICK);
    AddTestCase(new WifiFragmentUnderBlockAckRecipientTestCase, TestCase::Duration::QUICK);
}

/// the test suite
static WifiFragmentationBlockAckTestSuite g_wifiFragmentationBlockAckTestSuite;
