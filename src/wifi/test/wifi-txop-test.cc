/*
 * Copyright (c) 2020 Universita' degli Studi di Napoli Federico II
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Stefano Avallone <stavallo@unina.it>
 */

#include "ns3/ap-wifi-mac.h"
#include "ns3/boolean.h"
#include "ns3/config.h"
#include "ns3/he-phy.h"
#include "ns3/log.h"
#include "ns3/mobility-helper.h"
#include "ns3/multi-model-spectrum-channel.h"
#include "ns3/node-list.h"
#include "ns3/packet-socket-client.h"
#include "ns3/packet-socket-helper.h"
#include "ns3/packet-socket-server.h"
#include "ns3/packet.h"
#include "ns3/pointer.h"
#include "ns3/qos-txop.h"
#include "ns3/qos-utils.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/spectrum-wifi-helper.h"
#include "ns3/string.h"
#include "ns3/test.h"
#include "ns3/wifi-mac-header.h"
#include "ns3/wifi-net-device.h"
#include "ns3/wifi-ppdu.h"
#include "ns3/wifi-psdu.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("WifiTxopTest");

/**
 * @ingroup wifi-test
 * @ingroup tests
 *
 * @brief Test TXOP rules
 *
 * A BSS consisting of an AP and 3 non-AP STAs is considered in this test. Both non-HT (802.11a)
 * and HE devices are tested. Two TXOPs are simulated in this test:
 * - In the first TXOP, the AP sends a QoS data frame to each of the three STAs. The Ack in
 *   response to the initial frame is corrupted, hence the AP terminates the TXOP and tries again
 *   when a new TXOP is gained. In the new TXOP, the initial frame sent to STA 1 is successfully
 *   received, while the second frame to STA 2 is corrupted. It is checked that the AP performs
 *   PIFS recovery or invokes backoff depending on the value of the PifsRecovery attribute. All
 *   QoS data frames transmitted have a length/duration that does not exceed the length/duration
 *   based RTS/CTS threshold, hence RTS/CTS is never used. After that a QoS data frame has been
 *   sent to every STA, the AP sends another QoS data frame to one of them; the size/duration of
 *   this frame exceeds the threshold, but, given that the STA has already sent a response to the
 *   AP in this TXOP, RTS/CTS is only used if the ProtectedIfResponded is false.
 * - In the second TXOP, the AP sends a QoS data frame, in case of non-HT devices, or an A-MPDU
 *   consisting of 2 MPDUs, in case of HE devices, to each of the three STAs. All PSDUs transmitted
 *   have a length/duration that exceeds the length/duration based RTS/CTS threshold, hence RTS/CTS
 *   is used to protect every PSDU, unless the SingleRtsPerTxop attribute is set to true, in which
 *   case only the initial frame in the TXOP is protected by RTS/CTS.
 */
class WifiTxopTest : public TestCase
{
  public:
    /// Parameters for this test
    struct Params
    {
        bool nonHt;        //!< use 802.11a standard if true, 802.11ax standard otherwise
        bool pifsRecovery; //!< whether PIFS recovery is used after failure of a non-initial frame
        bool singleRtsPerTxop; //!< whether protection mechanism is used no more than once per TXOP
        bool lengthBasedRtsCtsThresh; //!< if true, use length based RTS/CTS threshold; if false,
                                      //!< use TX duration based RTS/CTS threshold
        bool protectedIfResponded;  //!< whether a station is assumed to be protected if replied to
                                    //!< a frame requiring acknowledgment
        bool protectSingleExchange; //!< whether the Duration/ID field in frames establishing
                                    //!< protection only covers the immediate frame exchange instead
                                    //!< of rest of the TXOP limit
        Time singleExchangeProtectionBuffer; //!< whether the NAV duration should be extended by a
                                             //!< PIFS after the frame exchange when protection only
                                             //!< covers the immediate frame exchange instead of
                                             //!< rest of the TXOP limit
    };

    /**
     * Constructor
     * @param params parameters for the Wi-Fi TXOP test
     */
    WifiTxopTest(const Params& params);

    /**
     * Function to trace packets received by the server application
     * @param context the context
     * @param p the packet
     * @param addr the address
     */
    void L7Receive(std::string context, Ptr<const Packet> p, const Address& addr);

    /**
     * Callback invoked when PHY receives a PSDU to transmit
     * @param context the context
     * @param psduMap the PSDU map
     * @param txVector the TX vector
     * @param txPowerW the tx power in Watts
     */
    void Transmit(std::string context,
                  WifiConstPsduMap psduMap,
                  WifiTxVector txVector,
                  double txPowerW);

    /**
     * Check correctness of transmitted frames
     */
    void CheckResults();

  private:
    void DoRun() override;

    /// Enumeration for traffic directions
    enum TrafficDirection : uint8_t
    {
        DOWNLINK = 0,
        UPLINK
    };

    /**
     * @param dir the traffic direction (downlink/uplink)
     * @param staId the index (starting at 0) of the non-AP STA generating/receiving packets
     * @param count the number of packets to generate
     * @param pktSize the size of the packets to generate
     * @return an application generating the given number packets of the given size from/to the
     *         AP to/from the given non-AP STA
     */
    Ptr<PacketSocketClient> GetApplication(TrafficDirection dir,
                                           std::size_t staId,
                                           std::size_t count,
                                           std::size_t pktSize) const;

    /// Information about transmitted frames
    struct FrameInfo
    {
        Time txStart;          ///< Frame start TX time
        Time txDuration;       ///< Frame TX duration
        uint32_t size;         ///< PSDU size in bytes
        WifiMacHeader header;  ///< Frame MAC header
        WifiTxVector txVector; ///< TX vector used to transmit the frame
    };

    uint16_t m_nStations;                   ///< number of stations
    NetDeviceContainer m_staDevices;        ///< container for stations' NetDevices
    NetDeviceContainer m_apDevices;         ///< container for AP's NetDevice
    std::vector<FrameInfo> m_txPsdus;       ///< transmitted PSDUs
    Time m_apTxopLimit;                     ///< TXOP limit for AP (AC BE)
    uint8_t m_staAifsn;                     ///< AIFSN for STAs (AC BE)
    uint32_t m_staCwMin;                    ///< CWmin for STAs (AC BE)
    uint32_t m_staCwMax;                    ///< CWmax for STAs (AC BE)
    Time m_staTxopLimit;                    ///< TXOP limit for STAs (AC BE)
    uint16_t m_received;                    ///< number of packets received by the stations
    bool m_nonHt;                           ///< @copydoc Params::nonHt
    std::size_t m_payloadSizeRtsOn;         ///< size in bytes of packets protected by RTS
    std::size_t m_payloadSizeRtsOff;        ///< size in bytes of packets not protected by RTS
    Time m_startTime;                       ///< time when data frame exchanges start
    WifiMode m_mode;                        ///< wifi mode used to transmit data frames
    bool m_pifsRecovery;                    ///< @copydoc Params::pifsRecovery
    bool m_singleRtsPerTxop;                ///< @copydoc Params::singleRtsPerTxop
    bool m_lengthBasedRtsCtsThresh;         ///< @copydoc Params::lengthBasedRtsCtsThresh
    bool m_protectedIfResponded;            ///< @copydoc Params::protectedIfResponded
    bool m_protectSingleExchange;           ///< @copydoc Params::protectSingleExchange
    Time m_singleExchangeProtectionSurplus; ///< @copydoc Params::singleExchangeProtectionBuffer
    Ptr<ListErrorModel> m_apErrorModel;     ///< error model to install on the AP
    Ptr<ListErrorModel> m_staErrorModel;    ///< error model to install on a STA
    bool m_apCorrupted;  ///< whether the frame to be corrupted by the AP has been corrupted
    bool m_staCorrupted; ///< whether the frame to be corrupted by the STA has been corrupted
    std::vector<PacketSocketAddress> m_dlSockets; ///< packet socket address for DL traffic
    std::vector<PacketSocketAddress> m_ulSockets; ///< packet socket address for UL traffic
};

WifiTxopTest::WifiTxopTest(const WifiTxopTest::Params& params)
    : TestCase("Check correct operation within TXOPs with parameters: nonHt=" +
               std::to_string(params.nonHt) +
               " pifsRecovery=" + std::to_string(params.pifsRecovery) +
               " singleRtsPerTxop=" + std::to_string(params.singleRtsPerTxop) +
               " lengthBasedRtsCtsThresh=" + std::to_string(params.lengthBasedRtsCtsThresh) +
               " protectedIfResponded=" + std::to_string(params.protectedIfResponded) +
               " protectSingleExchange=" + std::to_string(params.protectSingleExchange) +
               " singleExchangeProtectionBuffer=" +
               std::to_string(params.singleExchangeProtectionBuffer.GetMicroSeconds()) + "us"),
      m_nStations(3),
      m_apTxopLimit(MicroSeconds(4768)),
      m_staAifsn(4),
      m_staCwMin(63),
      m_staCwMax(511),
      m_staTxopLimit(MicroSeconds(3232)),
      m_received(0),
      m_nonHt(params.nonHt),
      m_payloadSizeRtsOn(m_nonHt ? 2000 : 540),
      m_payloadSizeRtsOff(500),
      m_startTime(MilliSeconds(m_nonHt ? 410 : 520)),
      m_mode(m_nonHt ? OfdmPhy::GetOfdmRate12Mbps() : HePhy::GetHeMcs0()),
      m_pifsRecovery(params.pifsRecovery),
      m_singleRtsPerTxop(params.singleRtsPerTxop),
      m_lengthBasedRtsCtsThresh(params.lengthBasedRtsCtsThresh),
      m_protectedIfResponded(params.protectedIfResponded),
      m_protectSingleExchange(params.protectSingleExchange),
      m_singleExchangeProtectionSurplus(params.singleExchangeProtectionBuffer),
      m_apErrorModel(CreateObject<ListErrorModel>()),
      m_staErrorModel(CreateObject<ListErrorModel>()),
      m_apCorrupted(false),
      m_staCorrupted(false)
{
}

void
WifiTxopTest::L7Receive(std::string context, Ptr<const Packet> p, const Address& addr)
{
    if (p->GetSize() >= m_payloadSizeRtsOff)
    {
        m_received++;
    }
}

void
WifiTxopTest::Transmit(std::string context,
                       WifiConstPsduMap psduMap,
                       WifiTxVector txVector,
                       double txPowerW)
{
    bool corrupted{false};

    // The AP does not correctly receive the Ack sent in response to the QoS
    // data frame sent to the first station
    if (const auto& hdr = psduMap.begin()->second->GetHeader(0);
        hdr.IsAck() && !m_apCorrupted && !m_txPsdus.empty() &&
        m_txPsdus.back().header.IsQosData() &&
        m_txPsdus.back().header.GetAddr1() == m_staDevices.Get(0)->GetAddress())
    {
        corrupted = m_apCorrupted = true;
        m_apErrorModel->SetList({psduMap.begin()->second->GetPacket()->GetUid()});
    }

    // The second station does not correctly receive the first QoS data frame sent by the AP
    if (const auto& hdr = psduMap.begin()->second->GetHeader(0);
        !m_txPsdus.empty() && hdr.IsQosData() &&
        hdr.GetAddr1() == m_staDevices.Get(1)->GetAddress())
    {
        if (!m_staCorrupted)
        {
            corrupted = m_staCorrupted = true;
        }
        if (corrupted)
        {
            m_staErrorModel->SetList({psduMap.begin()->second->GetPacket()->GetUid()});
        }
        else
        {
            m_staErrorModel->SetList({});
        }
    }

    // When the AP sends the first frame to the third station (which is not protected by RTS/CTS),
    // we generate another frame addressed to the second station whose size/duration exceeds the
    // threshold; however, RTS is not used because the second station has already responded to
    // another frame in the same TXOP
    if (const auto& hdr = psduMap.begin()->second->GetHeader(0);
        hdr.IsQosData() && hdr.GetAddr1() == m_staDevices.Get(2)->GetAddress() &&
        hdr.GetSequenceNumber() == (m_nonHt ? 0 : 1))
    {
        m_apDevices.Get(0)->GetNode()->AddApplication(
            GetApplication(DOWNLINK, 1, m_nonHt ? 1 : 2, m_payloadSizeRtsOn));
    }

    // Log all transmitted frames that are not beacon frames and have been transmitted
    // after the start time (so as to skip association requests/responses)
    if (!psduMap.begin()->second->GetHeader(0).IsBeacon() && Simulator::Now() >= m_startTime)
    {
        m_txPsdus.push_back({Simulator::Now(),
                             WifiPhy::CalculateTxDuration(psduMap, txVector, WIFI_PHY_BAND_5GHZ),
                             psduMap[SU_STA_ID]->GetSize(),
                             psduMap[SU_STA_ID]->GetHeader(0),
                             txVector});
    }

    // Print all the transmitted frames if the test is executed through test-runner
    NS_LOG_INFO(psduMap.begin()->second->GetHeader(0).GetTypeString()
                << " seq " << psduMap.begin()->second->GetHeader(0).GetSequenceNumber() << " to "
                << psduMap.begin()->second->GetAddr1() << " TX duration "
                << WifiPhy::CalculateTxDuration(psduMap, txVector, WIFI_PHY_BAND_5GHZ)
                << " duration/ID " << psduMap.begin()->second->GetHeader(0).GetDuration()
                << (corrupted ? " CORRUPTED" : "") << std::endl);
}

void
WifiTxopTest::DoRun()
{
    // LogComponentEnable("WifiTxopTest", LOG_LEVEL_ALL);
    NS_LOG_FUNCTION(this);

    RngSeedManager::SetSeed(1);
    RngSeedManager::SetRun(40);
    int64_t streamNumber = 100;

    NodeContainer wifiApNode;
    wifiApNode.Create(1);

    NodeContainer wifiStaNodes;
    wifiStaNodes.Create(m_nStations);

    auto spectrumChannel = CreateObject<MultiModelSpectrumChannel>();
    auto lossModel = CreateObject<FriisPropagationLossModel>();
    spectrumChannel->AddPropagationLossModel(lossModel);
    auto delayModel = CreateObject<ConstantSpeedPropagationDelayModel>();
    spectrumChannel->SetPropagationDelayModel(delayModel);

    SpectrumWifiPhyHelper phy;
    phy.SetChannel(spectrumChannel);
    // use default 20 MHz channel in 5 GHz band
    phy.Set("ChannelSettings", StringValue("{0, 20, BAND_5GHZ, 0}"));

    Config::SetDefault("ns3::QosFrameExchangeManager::PifsRecovery", BooleanValue(m_pifsRecovery));
    Config::SetDefault("ns3::QosFrameExchangeManager::ProtectSingleExchange",
                       BooleanValue(m_protectSingleExchange));
    Config::SetDefault("ns3::QosFrameExchangeManager::SingleExchangeProtectionSurplus",
                       TimeValue(m_singleExchangeProtectionSurplus));
    Config::SetDefault("ns3::WifiDefaultProtectionManager::SingleRtsPerTxop",
                       BooleanValue(m_singleRtsPerTxop));
    if (m_lengthBasedRtsCtsThresh)
    {
        Config::SetDefault("ns3::WifiRemoteStationManager::RtsCtsThreshold",
                           UintegerValue(m_payloadSizeRtsOn * (m_nonHt ? 1 : 2)));
    }
    else
    {
        Config::SetDefault("ns3::WifiRemoteStationManager::RtsCtsTxDurationThresh",
                           TimeValue(Seconds(m_payloadSizeRtsOn * (m_nonHt ? 1 : 2) * 8.0 /
                                             m_mode.GetDataRate(MHz_u{20}))));
    }
    Config::SetDefault("ns3::FrameExchangeManager::ProtectedIfResponded",
                       BooleanValue(m_protectedIfResponded));

    WifiHelper wifi;
    wifi.SetStandard(m_nonHt ? WIFI_STANDARD_80211a : WIFI_STANDARD_80211ax);
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                 "DataMode",
                                 WifiModeValue(m_mode),
                                 "ControlMode",
                                 StringValue("OfdmRate6Mbps"));

    WifiMacHelper mac;
    mac.SetType("ns3::StaWifiMac",
                "QosSupported",
                BooleanValue(true),
                "Ssid",
                SsidValue(Ssid("non-existent-ssid")));

    m_staDevices = wifi.Install(phy, mac, wifiStaNodes);
    streamNumber += WifiHelper::AssignStreams(m_staDevices, streamNumber);

    mac.SetType(
        "ns3::ApWifiMac",
        "QosSupported",
        BooleanValue(true),
        "Ssid",
        SsidValue(Ssid("wifi-txop-ssid")),
        "BeaconInterval",
        TimeValue(MicroSeconds(102400)),
        "EnableBeaconJitter",
        BooleanValue(false),
        "AifsnsForSta",
        StringValue(std::string("BE ") + std::to_string(m_staAifsn)),
        "CwMinsForSta",
        ApWifiMac::UintAccessParamsMapValue(
            ApWifiMac::UintAccessParamsMap{{AC_BE, std::vector<uint64_t>{m_staCwMin}}}),
        "CwMaxsForSta",
        StringValue(std::string("BE ") + std::to_string(m_staCwMax)),
        "TxopLimitsForSta",
        StringValue(std::string("BE ") + std::to_string(m_staTxopLimit.GetMicroSeconds()) + "us"));

    mac.SetEdca(AC_BE, "TxopLimits", AttributeContainerValue<TimeValue>(std::list{m_apTxopLimit}));

    m_apDevices = wifi.Install(phy, mac, wifiApNode);
    streamNumber += WifiHelper::AssignStreams(m_apDevices, streamNumber);

    // schedule association requests at different times. One station's SSID is
    // set to the correct value before initialization, so that such a station
    // starts the scanning procedure by looking for the correct SSID
    Ptr<WifiNetDevice> dev = DynamicCast<WifiNetDevice>(m_staDevices.Get(0));
    dev->GetMac()->SetSsid(Ssid("wifi-txop-ssid"));

    for (uint16_t i = 1; i < m_nStations; i++)
    {
        dev = DynamicCast<WifiNetDevice>(m_staDevices.Get(i));
        Simulator::Schedule(i * MicroSeconds(102400),
                            &WifiMac::SetSsid,
                            dev->GetMac(),
                            Ssid("wifi-txop-ssid"));
    }

    MobilityHelper mobility;
    Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator>();

    positionAlloc->Add(Vector(0.0, 0.0, 0.0));
    positionAlloc->Add(Vector(1.0, 0.0, 0.0));
    positionAlloc->Add(Vector(0.0, 1.0, 0.0));
    positionAlloc->Add(Vector(-1.0, 0.0, 0.0));
    mobility.SetPositionAllocator(positionAlloc);

    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(wifiApNode);
    mobility.Install(wifiStaNodes);

    PacketSocketHelper packetSocket;
    packetSocket.Install(wifiApNode);
    packetSocket.Install(wifiStaNodes);

    // install a packet socket server on all nodes
    for (auto nodeIt = NodeList::Begin(); nodeIt != NodeList::End(); ++nodeIt)
    {
        PacketSocketAddress srvAddr;
        auto device = DynamicCast<WifiNetDevice>((*nodeIt)->GetDevice(0));
        NS_TEST_ASSERT_MSG_NE(device, nullptr, "Expected a WifiNetDevice");
        srvAddr.SetSingleDevice(device->GetIfIndex());
        srvAddr.SetProtocol(1);

        auto server = CreateObject<PacketSocketServer>();
        server->SetLocal(srvAddr);
        (*nodeIt)->AddApplication(server);
        server->SetStartTime(Time{0}); // now
        server->SetStopTime(Seconds(1));
    }

    // set DL and UL packet sockets
    for (uint16_t i = 0; i < m_nStations; ++i)
    {
        m_dlSockets.emplace_back();
        m_dlSockets.back().SetSingleDevice(m_apDevices.Get(0)->GetIfIndex());
        m_dlSockets.back().SetPhysicalAddress(m_staDevices.Get(i)->GetAddress());
        m_dlSockets.back().SetProtocol(1);

        m_ulSockets.emplace_back();
        m_ulSockets.back().SetSingleDevice(m_staDevices.Get(i)->GetIfIndex());
        m_ulSockets.back().SetPhysicalAddress(m_apDevices.Get(0)->GetAddress());
        m_ulSockets.back().SetProtocol(1);
    }

    // DL frames
    for (uint16_t i = 0; i < m_nStations; i++)
    {
        if (!m_nonHt)
        {
            // Send one QoS data frame to establish Block Ack agreement (packet size is such that
            // this packet is not counted as a received packet)
            Simulator::Schedule(m_startTime - MilliSeconds(110 - i * 25),
                                &Node::AddApplication,
                                wifiApNode.Get(0),
                                GetApplication(DOWNLINK, i, 1, m_payloadSizeRtsOff - 1));
        }

        // Send one QoS data frame (not protected by RTS/CTS) to each station
        Simulator::Schedule(m_startTime,
                            &Node::AddApplication,
                            wifiApNode.Get(0),
                            GetApplication(DOWNLINK, i, 1, m_payloadSizeRtsOff));

        // Send one QoS data frame (protected by RTS/CTS) to each station
        Simulator::Schedule(m_startTime + MilliSeconds(110),
                            &Node::AddApplication,
                            wifiApNode.Get(0),
                            GetApplication(DOWNLINK, i, m_nonHt ? 1 : 2, m_payloadSizeRtsOn));
    }

    // install the error model on the AP
    dev = DynamicCast<WifiNetDevice>(m_apDevices.Get(0));
    dev->GetMac()->GetWifiPhy()->SetPostReceptionErrorModel(m_apErrorModel);

    // install the error model on the second station
    dev = DynamicCast<WifiNetDevice>(m_staDevices.Get(1));
    dev->GetMac()->GetWifiPhy()->SetPostReceptionErrorModel(m_staErrorModel);

    // UL Traffic (the first station sends one frame to the AP)
    {
        if (!m_nonHt)
        {
            // Send one QoS data frame to establish Block Ack agreement (packet size is such that
            // this packet is not counted as a received packet)
            Simulator::Schedule(m_startTime - MilliSeconds(35),
                                &Node::AddApplication,
                                wifiStaNodes.Get(0),
                                GetApplication(UPLINK, 0, 1, m_payloadSizeRtsOff - 1));
        }

        Simulator::Schedule(m_startTime + MilliSeconds(2),
                            &Node::AddApplication,
                            wifiStaNodes.Get(0),
                            GetApplication(UPLINK, 0, 1, m_payloadSizeRtsOff));
    }

    Config::Connect("/NodeList/*/ApplicationList/*/$ns3::PacketSocketServer/Rx",
                    MakeCallback(&WifiTxopTest::L7Receive, this));
    // Trace PSDUs passed to the PHY on all devices
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/PhyTxPsduBegin",
                    MakeCallback(&WifiTxopTest::Transmit, this));

    Simulator::Stop(Seconds(1));
    Simulator::Run();

    CheckResults();

    Simulator::Destroy();
}

Ptr<PacketSocketClient>
WifiTxopTest::GetApplication(TrafficDirection dir,
                             std::size_t staId,
                             std::size_t count,
                             std::size_t pktSize) const
{
    auto client = CreateObject<PacketSocketClient>();
    client->SetAttribute("PacketSize", UintegerValue(pktSize));
    client->SetAttribute("MaxPackets", UintegerValue(count));
    client->SetAttribute("Interval", TimeValue(MicroSeconds(0)));
    client->SetRemote(dir == DOWNLINK ? m_dlSockets.at(staId) : m_ulSockets.at(staId));
    client->SetStartTime(Time{0}); // now
    client->SetStopTime(Seconds(1));

    return client;
}

void
WifiTxopTest::CheckResults()
{
    // check that STAs used the access parameters advertised by the AP
    for (uint32_t i = 0; i < m_staDevices.GetN(); ++i)
    {
        auto staEdca = DynamicCast<WifiNetDevice>(m_staDevices.Get(i))->GetMac()->GetQosTxop(AC_BE);
        NS_TEST_EXPECT_MSG_EQ(staEdca->GetAifsn(SINGLE_LINK_OP_ID),
                              m_staAifsn,
                              "Unexpected AIFSN for STA " << i);
        NS_TEST_EXPECT_MSG_EQ(staEdca->GetMinCw(SINGLE_LINK_OP_ID),
                              m_staCwMin,
                              "Unexpected CWmin for STA " << i);
        NS_TEST_EXPECT_MSG_EQ(staEdca->GetMaxCw(SINGLE_LINK_OP_ID),
                              m_staCwMax,
                              "Unexpected CWmax for STA " << i);
        NS_TEST_EXPECT_MSG_EQ(staEdca->GetTxopLimit(SINGLE_LINK_OP_ID),
                              m_staTxopLimit,
                              "Unexpected TXOP limit for STA " << i);
    }

    const auto apDev = DynamicCast<WifiNetDevice>(m_apDevices.Get(0));

    NS_TEST_EXPECT_MSG_EQ(apDev->GetMac()->GetQosTxop(AC_BE)->GetTxopLimit(SINGLE_LINK_OP_ID),
                          m_apTxopLimit,
                          "Unexpected TXOP limit for AP");

    const auto aifsn = apDev->GetMac()->GetQosTxop(AC_BE)->GetAifsn(SINGLE_LINK_OP_ID);
    const auto cwMin = apDev->GetMac()->GetQosTxop(AC_BE)->GetMinCw(SINGLE_LINK_OP_ID);
    Time tEnd;                        // TX end for a frame
    Time tStart;                      // TX start for the next frame
    Time txopStart;                   // TXOP start time
    Time tolerance = NanoSeconds(50); // due to propagation delay
    Time sifs = apDev->GetPhy()->GetSifs();
    Time slot = apDev->GetPhy()->GetSlot();
    Time navEnd;
    TypeId::AttributeInformation info;
    WifiRemoteStationManager::GetTypeId().LookupAttributeByName("RtsCtsThreshold", &info);
    const uint32_t rtsCtsThreshold = DynamicCast<const UintegerValue>(info.initialValue)->Get();
    WifiRemoteStationManager::GetTypeId().LookupAttributeByName("RtsCtsTxDurationThresh", &info);
    const Time rtsCtsTxDurationThresh = DynamicCast<const TimeValue>(info.initialValue)->Get();

    // lambda to round Duration/ID (in microseconds) up to the next higher integer
    auto RoundDurationId = [](Time t) {
        return MicroSeconds(ceil(static_cast<double>(t.GetNanoSeconds()) / 1000));
    };

    /*
     * Verify the different behavior followed when an initial/non-initial frame of a TXOP
     * fails. Also, verify that a CF-end frame is sent if enough time remains in the TXOP.
     * The destination of failed frames is put in square brackets below.
     *
     *        |--NAV----till end TXOP-------->|
     *        |     |---NAV--till end TXOP--->|
     *        |     |          |-----------------------------NAV--------------------------------->|
     *        |     |          |     |----------------------------NAV---------------------------->|
     *        |     |          |     |     |---------------------------NAV----------------------->|
     *        |     |          |     |     |               |----------------NAV------------------>|
     *        |     |          |     |     |               |     |-------------NAV--------------->|
     *        |     |          |     |     |   Ack         |     |     |-----------NAV----------->|
     *   Start|     |     Start|     |     | Timeout       |     |     |     |--------NAV-------->|
     *   TXOP |     |     TXOP |     |     |   |-PIFS->    |     |     |     |     |-----NAV----->|
     *    |   |     | back |   |     |     |   |- or ->    |     |     |     |     |     |--NAV-->|
     *    ┌───┐ ┌───┐-off->┌───┐ ┌───┐ ┌───┐   |-back->┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌─────┐
     *    │QoS│ │Ack│      │QoS│ │Ack│ │QoS│   |-off ->│QoS│ │Ack│ │QoS│ │Ack│ │QoS│ │Ack│ │CFend│
     * ───┴───┴─┴───┴──────┴───┴─┴───┴─┴───┴───────────┴───┴─┴───┴─┴───┴─┴───┴─┴───┴─┴───┴─┴─────┴─
     * TA:  AP   STA1        AP   STA1   AP              AP   STA2   AP   STA3   AP   STA2    AP
     * RA: STA1  [AP]       STA1   AP  [STA2]           STA2   AP   STA3   AP   STA2   AP    all
     *
     * NOTE: If ProtectedIfResponded is false, the last QoS data frame is protected by RTS/CTS
     * NOTE: If ProtectSingleExchange is true, CF-END frames are not transmitted
     *       and NAV ends once DATA+Ack is completed
     */

    // We expect 25 frames to be transmitted if SingleRtsPerTxop is false and 22 frames (2 RTS
    // less, 2 CTS less, 1 more CF-End). If ProtectedIfResponded is false, there are 2 frames
    // (an RTS and a CTS) more. If ProtectSingleExchange is true, there are 2 (if
    // SingleRtsPerTxop is false) or 3 (if SingleRtsPerTxop is true) less frames (CF-END)
    NS_TEST_ASSERT_MSG_EQ(
        m_txPsdus.size(),
        (m_singleRtsPerTxop
             ? 21U + (m_protectedIfResponded ? 0U : 2U) + (m_protectSingleExchange ? 0U : 3U)
             : 25U + (m_protectedIfResponded ? 0U : 2U) + (m_protectSingleExchange ? 0U : 2U)),
        "Unexpected number of transmitted frames");

    // the first frame sent after 400ms is a QoS data frame sent by the AP to STA1 without RTS/CTS
    txopStart = m_txPsdus[0].txStart;

    NS_TEST_EXPECT_MSG_EQ(m_txPsdus[0].header.IsQosData(), true, "Expected a QoS data frame");
    NS_TEST_EXPECT_MSG_EQ(m_txPsdus[0].header.GetAddr1(),
                          DynamicCast<WifiNetDevice>(m_staDevices.Get(0))->GetMac()->GetAddress(),
                          "Expected a frame sent by the AP to the first station");
    if (m_lengthBasedRtsCtsThresh)
    {
        NS_TEST_EXPECT_MSG_LT(m_txPsdus[0].size,
                              rtsCtsThreshold,
                              "PSDU size expected not to exceed length based RTS/CTS threshold");
    }
    else
    {
        NS_TEST_EXPECT_MSG_LT(
            m_txPsdus[0].txDuration,
            rtsCtsTxDurationThresh,
            "PSDU duration expected not to exceed duration based RTS/CTS threshold");
    }
    NS_TEST_EXPECT_MSG_EQ(
        m_txPsdus[0].header.GetDuration(),
        (!m_protectSingleExchange
             ? RoundDurationId(m_apTxopLimit - m_txPsdus[0].txDuration)
             : sifs + m_txPsdus[1].txDuration + m_singleExchangeProtectionSurplus),
        (!m_protectSingleExchange
             ? "Duration/ID of the first frame must cover the whole TXOP"
             : "Duration/ID of the first frame must be set to remaining time to "
               "complete DATA+ACK sequence"));

    // a Normal Ack is sent by STA1
    tEnd = m_txPsdus[0].txStart + m_txPsdus[0].txDuration;
    tStart = m_txPsdus[1].txStart;

    NS_TEST_EXPECT_MSG_LT(tEnd + sifs, tStart, "Ack in response to the first frame sent too early");
    NS_TEST_EXPECT_MSG_LT(tStart,
                          tEnd + sifs + tolerance,
                          "Ack in response to the first frame sent too late");
    NS_TEST_EXPECT_MSG_EQ(m_txPsdus[1].header.IsAck(), true, "Expected a Normal Ack");
    NS_TEST_EXPECT_MSG_EQ(m_txPsdus[1].header.GetAddr1(),
                          apDev->GetMac()->GetAddress(),
                          "Expected a Normal Ack sent to the AP");
    NS_TEST_EXPECT_MSG_EQ(
        m_txPsdus[1].header.GetDuration(),
        RoundDurationId(m_txPsdus[0].header.GetDuration() - sifs - m_txPsdus[1].txDuration),
        "Duration/ID of the Ack must be derived from that of the first frame");

    // the AP receives a corrupted Ack in response to the frame it sent, which is the initial
    // frame of a TXOP. Hence, the TXOP is terminated and the AP retransmits the frame after
    // waiting for EIFS - DIFS + AIFS + backoff (see section 10.3.2.3.7 of 802.11-2020)
    txopStart = m_txPsdus[2].txStart;

    tEnd = m_txPsdus[1].txStart + m_txPsdus[1].txDuration;
    tStart = m_txPsdus[2].txStart;

    auto apPhy = apDev->GetPhy(SINGLE_LINK_OP_ID);
    auto eifsNoDifs = apPhy->GetSifs() + GetEstimatedAckTxTime(m_txPsdus[1].txVector);

    NS_TEST_EXPECT_MSG_GT_OR_EQ(
        tStart - tEnd,
        eifsNoDifs + sifs + aifsn * slot,
        "Less than AIFS elapsed between AckTimeout and the next TXOP start");
    NS_TEST_EXPECT_MSG_LT_OR_EQ(
        tStart - tEnd,
        eifsNoDifs + sifs + aifsn * slot + (2 * (cwMin + 1) - 1) * slot,
        "More than AIFS+BO elapsed between AckTimeout and the next TXOP start");
    NS_TEST_EXPECT_MSG_EQ(m_txPsdus[2].header.IsQosData(),
                          true,
                          "Expected to retransmit a QoS data frame");
    NS_TEST_EXPECT_MSG_EQ(m_txPsdus[2].header.GetAddr1(),
                          DynamicCast<WifiNetDevice>(m_staDevices.Get(0))->GetMac()->GetAddress(),
                          "Expected to retransmit a frame to the first station");
    if (m_lengthBasedRtsCtsThresh)
    {
        NS_TEST_EXPECT_MSG_LT(m_txPsdus[2].size,
                              rtsCtsThreshold,
                              "PSDU size expected not to exceed length based RTS/CTS threshold");
    }
    else
    {
        NS_TEST_EXPECT_MSG_LT(
            m_txPsdus[2].txDuration,
            rtsCtsTxDurationThresh,
            "PSDU duration expected not to exceed duration based RTS/CTS threshold");
    }
    NS_TEST_EXPECT_MSG_EQ(
        m_txPsdus[2].header.GetDuration(),
        (!m_protectSingleExchange
             ? RoundDurationId(m_apTxopLimit - m_txPsdus[2].txDuration)
             : sifs + m_txPsdus[3].txDuration + m_singleExchangeProtectionSurplus),
        (!m_protectSingleExchange
             ? "Duration/ID of the retransmitted frame must cover the whole TXOP"
             : "Duration/ID of the retransmitted frame must be set to remaining time to "
               "complete DATA+ACK sequence"));

    // a Normal Ack is then sent by STA1
    tEnd = m_txPsdus[2].txStart + m_txPsdus[2].txDuration;
    tStart = m_txPsdus[3].txStart;

    NS_TEST_EXPECT_MSG_LT(tEnd + sifs, tStart, "Ack in response to the first frame sent too early");
    NS_TEST_EXPECT_MSG_LT(tStart,
                          tEnd + sifs + tolerance,
                          "Ack in response to the first frame sent too late");
    NS_TEST_EXPECT_MSG_EQ(m_txPsdus[3].header.IsAck(), true, "Expected a Normal Ack");
    NS_TEST_EXPECT_MSG_EQ(m_txPsdus[3].header.GetAddr1(),
                          apDev->GetMac()->GetAddress(),
                          "Expected a Normal Ack sent to the AP");
    NS_TEST_EXPECT_MSG_EQ(
        m_txPsdus[3].header.GetDuration(),
        RoundDurationId(m_txPsdus[2].header.GetDuration() - sifs - m_txPsdus[3].txDuration),
        "Duration/ID of the Ack must be derived from that of the previous frame");

    // the AP sends a frame to STA2
    tEnd = m_txPsdus[3].txStart + m_txPsdus[3].txDuration;
    tStart = m_txPsdus[4].txStart;

    NS_TEST_EXPECT_MSG_LT(tEnd + sifs, tStart, "Second frame sent too early");
    NS_TEST_EXPECT_MSG_LT(tStart, tEnd + sifs + tolerance, "Second frame sent too late");
    NS_TEST_EXPECT_MSG_EQ(m_txPsdus[4].header.IsQosData(), true, "Expected a QoS data frame");
    NS_TEST_EXPECT_MSG_EQ(m_txPsdus[4].header.GetAddr1(),
                          DynamicCast<WifiNetDevice>(m_staDevices.Get(1))->GetMac()->GetAddress(),
                          "Expected a frame sent by the AP to the second station");
    if (m_lengthBasedRtsCtsThresh)
    {
        NS_TEST_EXPECT_MSG_LT(m_txPsdus[4].size,
                              rtsCtsThreshold,
                              "PSDU size expected not to exceed length based RTS/CTS threshold");
    }
    else
    {
        NS_TEST_EXPECT_MSG_LT(
            m_txPsdus[4].txDuration,
            rtsCtsTxDurationThresh,
            "PSDU duration expected not to exceed duration based RTS/CTS threshold");
    }
    NS_TEST_EXPECT_MSG_EQ(m_txPsdus[4].header.GetDuration(),
                          (!m_protectSingleExchange
                               ? RoundDurationId(m_apTxopLimit -
                                                 (m_txPsdus[4].txStart - txopStart) -
                                                 m_txPsdus[4].txDuration)
                               : sifs +
                                     m_txPsdus[6].txDuration + m_singleExchangeProtectionSurplus /* ACK is 2 frames later since this DATA is first corrupted */),
                          (!m_protectSingleExchange
                               ? "Duration/ID of the second frame does not cover the remaining TXOP"
                               : "Duration/ID of the second frame must be set to remaining time to "
                                 "complete DATA+ACK sequence"));

    // STA2 receives a corrupted frame and hence it does not send the Ack. When the AckTimeout
    // expires, the AP performs PIFS recovery or invoke backoff, without terminating the TXOP,
    // because a non-initial frame of the TXOP failed
    auto apStationManager = apDev->GetRemoteStationManager(SINGLE_LINK_OP_ID);
    auto staAddress = DynamicCast<WifiNetDevice>(m_staDevices.Get(1))->GetMac()->GetAddress();
    auto ackTxVector = apStationManager->GetAckTxVector(staAddress, m_txPsdus[4].txVector);
    tEnd = m_txPsdus[4].txStart + m_txPsdus[4].txDuration + sifs + slot +
           WifiPhy::CalculatePhyPreambleAndHeaderDuration(ackTxVector); // AckTimeout
    tStart = m_txPsdus[5].txStart;

    if (m_pifsRecovery)
    {
        NS_TEST_EXPECT_MSG_EQ(tEnd + sifs + slot,
                              tStart,
                              "Second frame must have been sent after a PIFS");
    }
    else
    {
        NS_TEST_EXPECT_MSG_GT_OR_EQ(
            tStart - tEnd,
            sifs + aifsn * slot,
            "Less than AIFS elapsed between AckTimeout and the next transmission");
        NS_TEST_EXPECT_MSG_LT_OR_EQ(
            tStart - tEnd,
            sifs + aifsn * slot + (2 * (cwMin + 1) - 1) * slot,
            "More than AIFS+BO elapsed between AckTimeout and the next TXOP start");
    }
    NS_TEST_EXPECT_MSG_EQ(m_txPsdus[5].header.IsQosData(), true, "Expected a QoS data frame");
    NS_TEST_EXPECT_MSG_EQ(m_txPsdus[5].header.GetAddr1(),
                          DynamicCast<WifiNetDevice>(m_staDevices.Get(1))->GetMac()->GetAddress(),
                          "Expected a frame sent by the AP to the second station");
    if (m_lengthBasedRtsCtsThresh)
    {
        NS_TEST_EXPECT_MSG_LT(m_txPsdus[5].size,
                              rtsCtsThreshold,
                              "PSDU size expected not to exceed length based RTS/CTS threshold");
    }
    else
    {
        NS_TEST_EXPECT_MSG_LT(
            m_txPsdus[5].txDuration,
            rtsCtsTxDurationThresh,
            "PSDU duration expected not to exceed duration based RTS/CTS threshold");
    }
    NS_TEST_EXPECT_MSG_EQ(
        m_txPsdus[5].header.GetDuration(),
        (!m_protectSingleExchange
             ? RoundDurationId(m_apTxopLimit - (m_txPsdus[5].txStart - txopStart) -
                               m_txPsdus[5].txDuration)
             : sifs + m_txPsdus[6].txDuration + m_singleExchangeProtectionSurplus),
        (!m_protectSingleExchange
             ? "Duration/ID of the second frame does not cover the remaining TXOP"
             : "Duration/ID of the second frame must be set to remaining time to "
               "complete DATA+ACK sequence"));

    // a Normal Ack is then sent by STA2
    tEnd = m_txPsdus[5].txStart + m_txPsdus[5].txDuration;
    tStart = m_txPsdus[6].txStart;

    NS_TEST_EXPECT_MSG_LT(tEnd + sifs,
                          tStart,
                          "Ack in response to the second frame sent too early");
    NS_TEST_EXPECT_MSG_LT(tStart,
                          tEnd + sifs + tolerance,
                          "Ack in response to the second frame sent too late");
    NS_TEST_EXPECT_MSG_EQ(m_txPsdus[6].header.IsAck(), true, "Expected a Normal Ack");
    NS_TEST_EXPECT_MSG_EQ(m_txPsdus[6].header.GetAddr1(),
                          apDev->GetMac()->GetAddress(),
                          "Expected a Normal Ack sent to the AP");
    NS_TEST_EXPECT_MSG_EQ(
        m_txPsdus[6].header.GetDuration(),
        RoundDurationId(m_txPsdus[5].header.GetDuration() - sifs - m_txPsdus[6].txDuration),
        "Duration/ID of the Ack must be derived from that of the previous frame");

    // the AP sends a frame to STA3
    tEnd = m_txPsdus[6].txStart + m_txPsdus[6].txDuration;
    tStart = m_txPsdus[7].txStart;

    NS_TEST_EXPECT_MSG_LT(tEnd + sifs, tStart, "Third frame sent too early");
    NS_TEST_EXPECT_MSG_LT(tStart, tEnd + sifs + tolerance, "Third frame sent too late");
    NS_TEST_EXPECT_MSG_EQ(m_txPsdus[7].header.IsQosData(), true, "Expected a QoS data frame");
    NS_TEST_EXPECT_MSG_EQ(m_txPsdus[7].header.GetAddr1(),
                          DynamicCast<WifiNetDevice>(m_staDevices.Get(2))->GetMac()->GetAddress(),
                          "Expected a frame sent by the AP to the third station");
    if (m_lengthBasedRtsCtsThresh)
    {
        NS_TEST_EXPECT_MSG_LT(m_txPsdus[7].size,
                              rtsCtsThreshold,
                              "PSDU size expected not to exceed length based RTS/CTS threshold");
    }
    else
    {
        NS_TEST_EXPECT_MSG_LT(
            m_txPsdus[7].txDuration,
            rtsCtsTxDurationThresh,
            "PSDU duration expected not to exceed duration based RTS/CTS threshold");
    }
    NS_TEST_EXPECT_MSG_EQ(
        m_txPsdus[7].header.GetDuration(),
        (!m_protectSingleExchange
             ? RoundDurationId(m_apTxopLimit - (m_txPsdus[7].txStart - txopStart) -
                               m_txPsdus[7].txDuration)
             : sifs + m_txPsdus[8].txDuration + m_singleExchangeProtectionSurplus),
        (!m_protectSingleExchange
             ? "Duration/ID of the third frame does not cover the remaining TXOP"
             : "Duration/ID of the third frame must be set to remaining time to "
               "complete DATA+ACK sequence"));

    // a Normal Ack is then sent by STA3
    tEnd = m_txPsdus[7].txStart + m_txPsdus[7].txDuration;
    tStart = m_txPsdus[8].txStart;

    NS_TEST_EXPECT_MSG_LT(tEnd + sifs, tStart, "Ack in response to the third frame sent too early");
    NS_TEST_EXPECT_MSG_LT(tStart,
                          tEnd + sifs + tolerance,
                          "Ack in response to the third frame sent too late");
    NS_TEST_EXPECT_MSG_EQ(m_txPsdus[8].header.IsAck(), true, "Expected a Normal Ack");
    NS_TEST_EXPECT_MSG_EQ(m_txPsdus[8].header.GetAddr1(),
                          apDev->GetMac()->GetAddress(),
                          "Expected a Normal Ack sent to the AP");
    NS_TEST_EXPECT_MSG_EQ(
        m_txPsdus[8].header.GetDuration(),
        RoundDurationId(m_txPsdus[7].header.GetDuration() - sifs - m_txPsdus[8].txDuration),
        "Duration/ID of the Ack must be derived from that of the previous frame");

    // the AP sends another frame to STA2, which is only protected if ProtectedIfResponded is true,
    // because its size/duration exceeds the threshold but STA2 has already responded to the AP in
    // this TXOP
    if (!m_protectedIfResponded)
    {
        tEnd = m_txPsdus[8].txStart + m_txPsdus[8].txDuration;
        tStart = m_txPsdus[9].txStart;

        NS_TEST_EXPECT_MSG_LT(tEnd + sifs, tStart, "RTS before fourth frame sent too early");
        NS_TEST_EXPECT_MSG_LT(tStart,
                              tEnd + sifs + tolerance,
                              "RTS before fourth frame sent too late");
        NS_TEST_EXPECT_MSG_EQ(m_txPsdus[9].header.IsRts(), true, "Expected an RTS frame");
        NS_TEST_EXPECT_MSG_EQ(
            m_txPsdus[9].header.GetAddr1(),
            DynamicCast<WifiNetDevice>(m_staDevices.Get(1))->GetMac()->GetAddress(),
            "Expected an RTS frame sent by the AP to the second station");
        NS_TEST_EXPECT_MSG_EQ(
            m_txPsdus[9].header.GetDuration(),
            (!m_protectSingleExchange
                 ? RoundDurationId(m_apTxopLimit - (m_txPsdus[9].txStart - txopStart) -
                                   m_txPsdus[9].txDuration)
                 : sifs + m_txPsdus[10].txDuration + sifs + m_txPsdus[11].txDuration + sifs +
                       m_txPsdus[12].txDuration + m_singleExchangeProtectionSurplus),
            (!m_protectSingleExchange
                 ? "Duration/ID of the RTS before the fourth frame does not cover the remaining "
                   "TXOP"
                 : "Duration/ID of the RTS before the fourth frame must be set to remaining time "
                   "to complete CTS+DATA+ACK sequence"));

        // a CTS is sent by STA2
        tEnd = m_txPsdus[9].txStart + m_txPsdus[9].txDuration;
        tStart = m_txPsdus[10].txStart;

        NS_TEST_EXPECT_MSG_LT(tEnd + sifs,
                              tStart,
                              "CTS in response to RTS before the fourth frame sent too early");
        NS_TEST_EXPECT_MSG_LT(tStart,
                              tEnd + sifs + tolerance,
                              "CTS in response to RTS before the fourth frame sent too late");
        NS_TEST_EXPECT_MSG_EQ(m_txPsdus[10].header.IsCts(), true, "Expected a CTS");
        NS_TEST_EXPECT_MSG_EQ(m_txPsdus[10].header.GetAddr1(),
                              apDev->GetMac()->GetAddress(),
                              "Expected a CTS frame sent to the AP");
        NS_TEST_EXPECT_MSG_EQ(
            m_txPsdus[10].header.GetDuration(),
            RoundDurationId(m_txPsdus[9].header.GetDuration() - sifs - m_txPsdus[10].txDuration),
            "Duration/ID of the CTS frame must be derived from that of the RTS frame");

        // the AP sends another frame to STA2
        tEnd = m_txPsdus[10].txStart + m_txPsdus[10].txDuration;
        tStart = m_txPsdus[11].txStart;

        // remove RTS and CTS so that indices are aligned with the ProtectedIfResponded true case
        m_txPsdus.erase(std::next(m_txPsdus.cbegin(), 9), std::next(m_txPsdus.cbegin(), 11));
    }
    else
    {
        // the AP sends another frame to STA2
        tEnd = m_txPsdus[8].txStart + m_txPsdus[8].txDuration;
        tStart = m_txPsdus[9].txStart;
    }

    NS_TEST_EXPECT_MSG_LT(tEnd + sifs, tStart, "Fourth frame sent too early");
    NS_TEST_EXPECT_MSG_LT(tStart, tEnd + sifs + tolerance, "Fourth frame sent too late");
    NS_TEST_EXPECT_MSG_EQ(m_txPsdus[9].header.IsQosData(), true, "Expected a QoS data frame");
    NS_TEST_EXPECT_MSG_EQ(m_txPsdus[9].header.GetAddr1(),
                          DynamicCast<WifiNetDevice>(m_staDevices.Get(1))->GetMac()->GetAddress(),
                          "Expected a frame sent by the AP to the second station");
    if (m_lengthBasedRtsCtsThresh)
    {
        NS_TEST_EXPECT_MSG_GT(m_txPsdus[9].size,
                              rtsCtsThreshold,
                              "PSDU size expected to exceed length based RTS/CTS threshold");
    }
    else
    {
        NS_TEST_EXPECT_MSG_GT(m_txPsdus[9].txDuration,
                              rtsCtsTxDurationThresh,
                              "PSDU duration expected to exceed duration based RTS/CTS threshold");
    }
    NS_TEST_EXPECT_MSG_EQ(
        m_txPsdus[9].header.GetDuration(),
        (!m_protectSingleExchange
             ? RoundDurationId(m_apTxopLimit - (m_txPsdus[9].txStart - txopStart) -
                               m_txPsdus[9].txDuration)
             : sifs + m_txPsdus[10].txDuration + m_singleExchangeProtectionSurplus),
        (!m_protectSingleExchange
             ? "Duration/ID of the fourth frame does not cover the remaining TXOP"
             : "Duration/ID of the fourth frame must be set to remaining time to "
               "complete DATA+(Block)ACK sequence"));

    std::string ack(m_nonHt ? "Normal Ack" : "Block Ack");

    // a Normal/Block Ack is then sent by STA1
    tEnd = m_txPsdus[9].txStart + m_txPsdus[9].txDuration;
    tStart = m_txPsdus[10].txStart;

    NS_TEST_EXPECT_MSG_LT(tEnd + sifs,
                          tStart,
                          ack << " in response to the fourth QoS data frame sent too early");
    NS_TEST_EXPECT_MSG_LT(tStart,
                          tEnd + sifs + tolerance,
                          ack << " in response to the fourth QoS data frame sent too late");
    NS_TEST_EXPECT_MSG_EQ(
        (m_nonHt ? m_txPsdus[10].header.IsAck() : m_txPsdus[10].header.IsBlockAck()),
        true,
        "Expected a " << ack);
    NS_TEST_EXPECT_MSG_EQ(m_txPsdus[10].header.GetAddr1(),
                          apDev->GetMac()->GetAddress(),
                          "Expected a " << ack << " sent to the AP");
    NS_TEST_EXPECT_MSG_EQ(
        m_txPsdus[10].header.GetDuration(),
        RoundDurationId(m_txPsdus[9].header.GetDuration() - sifs - m_txPsdus[10].txDuration),
        "Duration/ID of the " << ack << " must be derived from that of the previous frame");

    if (!m_protectSingleExchange)
    {
        // the TXOP limit is such that enough time for sending a CF-End frame remains
        tEnd = m_txPsdus[10].txStart + m_txPsdus[10].txDuration;
        tStart = m_txPsdus[11].txStart;

        NS_TEST_EXPECT_MSG_LT(tEnd + sifs, tStart, "CF-End sent too early");
        NS_TEST_EXPECT_MSG_LT(tStart, tEnd + sifs + tolerance, "CF-End sent too late");
        NS_TEST_EXPECT_MSG_EQ(m_txPsdus[11].header.IsCfEnd(), true, "Expected a CF-End frame");
        NS_TEST_EXPECT_MSG_EQ(m_txPsdus[11].header.GetDuration(),
                              MicroSeconds(0),
                              "Duration/ID must be set to 0 for CF-End frames");

        // remove CF-END so that indices are aligned with the protectSingleExchange false case
        m_txPsdus.erase(std::next(m_txPsdus.cbegin(), 11), std::next(m_txPsdus.cbegin(), 12));
    }

    // the CF-End frame resets the NAV on STA1, which can now transmit
    tEnd = m_txPsdus[10].txStart + m_txPsdus[10].txDuration;
    tStart = m_txPsdus[11].txStart;

    NS_TEST_EXPECT_MSG_GT_OR_EQ(tStart - tEnd,
                                sifs + m_staAifsn * slot,
                                "Less than AIFS elapsed between two TXOPs");
    NS_TEST_EXPECT_MSG_LT_OR_EQ(tStart - tEnd,
                                sifs + m_staAifsn * slot + m_staCwMin * slot + tolerance,
                                "More than AIFS+BO elapsed between two TXOPs");
    NS_TEST_EXPECT_MSG_EQ(m_txPsdus[11].header.IsQosData(), true, "Expected a QoS data frame");
    NS_TEST_EXPECT_MSG_EQ(m_txPsdus[11].header.GetAddr1(),
                          apDev->GetMac()->GetAddress(),
                          "Expected a frame sent by the first station to the AP");
    if (m_lengthBasedRtsCtsThresh)
    {
        NS_TEST_EXPECT_MSG_LT(m_txPsdus[11].size,
                              rtsCtsThreshold,
                              "PSDU size expected not to exceed length based RTS/CTS threshold");
    }
    else
    {
        NS_TEST_EXPECT_MSG_LT(
            m_txPsdus[11].txDuration,
            rtsCtsTxDurationThresh,
            "PSDU duration expected not to exceed duration based RTS/CTS threshold");
    }
    NS_TEST_EXPECT_MSG_EQ(
        m_txPsdus[11].header.GetDuration(),
        (!m_protectSingleExchange
             ? RoundDurationId(m_staTxopLimit - m_txPsdus[11].txDuration)
             : sifs + m_txPsdus[12].txDuration + m_singleExchangeProtectionSurplus),
        (!m_protectSingleExchange
             ? "Duration/ID of the frame sent by the first station does not "
               "cover the remaining TXOP"
             : "Duration/ID of the frame sent by the first station must be set "
               "to remaining time to "
               "complete DATA+ACK sequence"));

    // a Normal Ack is then sent by the AP
    tEnd = m_txPsdus[11].txStart + m_txPsdus[11].txDuration;
    tStart = m_txPsdus[12].txStart;

    NS_TEST_EXPECT_MSG_LT(tEnd + sifs, tStart, "Ack sent too early");
    NS_TEST_EXPECT_MSG_LT(tStart, tEnd + sifs + tolerance, "Ack sent too late");
    NS_TEST_EXPECT_MSG_EQ(m_txPsdus[12].header.IsAck(), true, "Expected a Normal Ack");
    NS_TEST_EXPECT_MSG_EQ(m_txPsdus[12].header.GetAddr1(),
                          DynamicCast<WifiNetDevice>(m_staDevices.Get(0))->GetMac()->GetAddress(),
                          "Expected a Normal Ack sent to the first station");
    NS_TEST_EXPECT_MSG_EQ(
        m_txPsdus[12].header.GetDuration(),
        RoundDurationId(m_txPsdus[11].header.GetDuration() - sifs - m_txPsdus[12].txDuration),
        "Duration/ID of the Ack must be derived from that of the previous frame");

    if (!m_protectSingleExchange)
    {
        // the TXOP limit is such that enough time for sending a CF-End frame remains
        tEnd = m_txPsdus[12].txStart + m_txPsdus[12].txDuration;
        tStart = m_txPsdus[13].txStart;

        NS_TEST_EXPECT_MSG_LT(tEnd + sifs, tStart, "CF-End sent too early");
        NS_TEST_EXPECT_MSG_LT(tStart, tEnd + sifs + tolerance, "CF-End sent too late");
        NS_TEST_EXPECT_MSG_EQ(m_txPsdus[13].header.IsCfEnd(), true, "Expected a CF-End frame");
        NS_TEST_EXPECT_MSG_EQ(m_txPsdus[13].header.GetDuration(),
                              MicroSeconds(0),
                              "Duration/ID must be set to 0 for CF-End frames");

        // remove CF-END so that indices are aligned with the protectSingleExchange false case
        m_txPsdus.erase(std::next(m_txPsdus.cbegin(), 13), std::next(m_txPsdus.cbegin(), 14));
    }

    /*
     * Verify that the Duration/ID of RTS/CTS frames is set correctly, that the TXOP holder is
     * kept and allows stations to ignore NAV properly and that the CF-End Frame is not sent if
     * not enough time remains. If SingleRtsPerTxop is set to true, only one RTS/CTS is sent.
     *
     *          |---------------------------------------------NAV---------------------------------->|
     *          | |-----------------------------------------NAV------------------------------->| |
     * |      |-------------------------------------NAV---------------------------->| |      | |
     * |---------------------------------NAV------------------------->| |      |      |      |
     * |-----------------------------NAV---------------------->| |      |      |      |      |
     * |-------------------------NAV------------------->| |      |      |      |      |      |
     * |---------------------NAV---------------->| |      |      |      |      |      |      |
     * |-----------------NAV------------->| |      |      |      |      |      |      |      |
     * |-------------NAV---------->| |      |      |      |      |      |      |      |      |
     * |---------NAV------->| |      |      |      |      |      |      |      |      |      |
     * |-----NAV---->| |      |      |      |      |      |      |      |      |      |      |
     * |-NAV->|
     *      |---|  |---|  |---|  |---|  |---|  |---|  |---|  |---|  |---|  |---|  |---|  |---|
     *      |RTS|  |CTS|  |QoS|  |Ack|  |RTS|  |CTS|  |QoS|  |Ack|  |RTS|  |CTS|  |QoS|  |Ack|
     * ----------------------------------------------------------------------------------------------------
     * From:  AP    STA1    AP    STA1    AP    STA2    AP    STA2    AP    STA3    AP    STA3
     *   To: STA1    AP    STA1    AP    STA2    AP    STA2    AP    STA3    AP    STA3    AP
     */

    // the first frame is an RTS frame sent by the AP to STA1
    txopStart = m_txPsdus[13].txStart;

    NS_TEST_EXPECT_MSG_EQ(m_txPsdus[13].header.IsRts(), true, "Expected an RTS frame");
    NS_TEST_EXPECT_MSG_EQ(m_txPsdus[13].header.GetAddr1(),
                          DynamicCast<WifiNetDevice>(m_staDevices.Get(0))->GetMac()->GetAddress(),
                          "Expected an RTS frame sent by the AP to the first station");
    NS_TEST_EXPECT_MSG_EQ(m_txPsdus[13].header.GetDuration(),
                          (!m_protectSingleExchange
                               ? RoundDurationId(m_apTxopLimit - m_txPsdus[13].txDuration)
                               : sifs + m_txPsdus[14].txDuration + sifs + m_txPsdus[15].txDuration +
                                     sifs + m_txPsdus[16].txDuration +
                                     m_singleExchangeProtectionSurplus),
                          (!m_protectSingleExchange
                               ? "Duration/ID of the first RTS frame must cover the whole TXOP"
                               : "Duration/ID of the first RTS frame must be set to remaining time "
                                 "to complete CTS+DATA+(Block)ACK sequence"));

    // a CTS is sent by STA1
    tEnd = m_txPsdus[13].txStart + m_txPsdus[13].txDuration;
    tStart = m_txPsdus[14].txStart;

    NS_TEST_EXPECT_MSG_LT(tEnd + sifs,
                          tStart,
                          "CTS in response to the first RTS frame sent too early");
    NS_TEST_EXPECT_MSG_LT(tStart,
                          tEnd + sifs + tolerance,
                          "CTS in response to the first RTS frame sent too late");
    NS_TEST_EXPECT_MSG_EQ(m_txPsdus[14].header.IsCts(), true, "Expected a CTS");
    NS_TEST_EXPECT_MSG_EQ(m_txPsdus[14].header.GetAddr1(),
                          apDev->GetMac()->GetAddress(),
                          "Expected a CTS frame sent to the AP");
    NS_TEST_EXPECT_MSG_EQ(
        m_txPsdus[14].header.GetDuration(),
        RoundDurationId(m_txPsdus[13].header.GetDuration() - sifs - m_txPsdus[14].txDuration),
        "Duration/ID of the CTS frame must be derived from that of the RTS frame");

    // the AP sends a frame to STA1
    tEnd = m_txPsdus[14].txStart + m_txPsdus[14].txDuration;
    tStart = m_txPsdus[15].txStart;

    NS_TEST_EXPECT_MSG_LT(tEnd + sifs, tStart, "First QoS data frame sent too early");
    NS_TEST_EXPECT_MSG_LT(tStart, tEnd + sifs + tolerance, "First QoS data frame sent too late");
    NS_TEST_EXPECT_MSG_EQ(m_txPsdus[15].header.IsQosData(), true, "Expected a QoS data frame");
    NS_TEST_EXPECT_MSG_EQ(m_txPsdus[15].header.GetAddr1(),
                          DynamicCast<WifiNetDevice>(m_staDevices.Get(0))->GetMac()->GetAddress(),
                          "Expected a frame sent by the AP to the first station");
    if (m_lengthBasedRtsCtsThresh)
    {
        NS_TEST_EXPECT_MSG_GT(m_txPsdus[15].size,
                              rtsCtsThreshold,
                              "PSDU size expected to exceed length based RTS/CTS threshold");
    }
    else
    {
        NS_TEST_EXPECT_MSG_GT(m_txPsdus[15].txDuration,
                              rtsCtsTxDurationThresh,
                              "PSDU duration expected to exceed duration based RTS/CTS threshold");
    }
    NS_TEST_EXPECT_MSG_EQ(
        m_txPsdus[15].header.GetDuration(),
        (!m_protectSingleExchange
             ? RoundDurationId(m_apTxopLimit - (m_txPsdus[15].txStart - txopStart) -
                               m_txPsdus[15].txDuration)
             : sifs + m_txPsdus[16].txDuration + m_singleExchangeProtectionSurplus),
        (!m_protectSingleExchange
             ? "Duration/ID of the first QoS data frame does not cover the remaining TXOP"
             : "Duration/ID of the first QoS data frame must be set to remaining time to "
               "complete DATA+(Block)ACK sequence"));

    // a Normal/Block Ack is then sent by STA1
    tEnd = m_txPsdus[15].txStart + m_txPsdus[15].txDuration;
    tStart = m_txPsdus[16].txStart;

    NS_TEST_EXPECT_MSG_LT(tEnd + sifs,
                          tStart,
                          ack << " in response to the first QoS data frame sent too early");
    NS_TEST_EXPECT_MSG_LT(tStart,
                          tEnd + sifs + tolerance,
                          ack << " in response to the first QoS data frame sent too late");
    NS_TEST_EXPECT_MSG_EQ(
        (m_nonHt ? m_txPsdus[16].header.IsAck() : m_txPsdus[16].header.IsBlockAck()),
        true,
        "Expected a " << ack);
    NS_TEST_EXPECT_MSG_EQ(m_txPsdus[16].header.GetAddr1(),
                          apDev->GetMac()->GetAddress(),
                          "Expected a " << ack << " sent to the AP");
    NS_TEST_EXPECT_MSG_EQ(
        m_txPsdus[16].header.GetDuration(),
        RoundDurationId(m_txPsdus[15].header.GetDuration() - sifs - m_txPsdus[16].txDuration),
        "Duration/ID of the " << ack << " must be derived from that of the previous frame");

    std::size_t idx = 16;

    if (!m_singleRtsPerTxop)
    {
        // An RTS frame is sent by the AP to STA2
        tEnd = m_txPsdus[idx].txStart + m_txPsdus[idx].txDuration;
        ++idx;
        tStart = m_txPsdus[idx].txStart;

        NS_TEST_EXPECT_MSG_LT(tEnd + sifs, tStart, "Second RTS frame sent too early");
        NS_TEST_EXPECT_MSG_LT(tStart, tEnd + sifs + tolerance, "Second RTS frame sent too late");
        NS_TEST_EXPECT_MSG_EQ(m_txPsdus[idx].header.IsRts(), true, "Expected an RTS frame");
        NS_TEST_EXPECT_MSG_EQ(
            m_txPsdus[idx].header.GetAddr1(),
            DynamicCast<WifiNetDevice>(m_staDevices.Get(1))->GetMac()->GetAddress(),
            "Expected an RTS frame sent by the AP to the second station");
        NS_TEST_EXPECT_MSG_EQ(
            m_txPsdus[idx].header.GetDuration(),
            (!m_protectSingleExchange
                 ? RoundDurationId(m_apTxopLimit - (m_txPsdus[idx].txStart - txopStart) -
                                   m_txPsdus[idx].txDuration)
                 : sifs + m_txPsdus[idx + 1].txDuration + sifs + m_txPsdus[idx + 2].txDuration +
                       sifs + m_txPsdus[idx + 3].txDuration + m_singleExchangeProtectionSurplus),
            (!m_protectSingleExchange
                 ? "Duration/ID of the second RTS frame must cover the whole TXOP"
                 : "Duration/ID of the second RTS frame must be set to remaining time to complete "
                   "CTS+DATA+(Block)ACK sequence"));

        // a CTS is sent by STA2 (which ignores the NAV)
        tEnd = m_txPsdus[idx].txStart + m_txPsdus[idx].txDuration;
        tStart = m_txPsdus[idx + 1].txStart;

        NS_TEST_EXPECT_MSG_LT(tEnd + sifs,
                              tStart,
                              "CTS in response to the second RTS frame sent too early");
        NS_TEST_EXPECT_MSG_LT(tStart,
                              tEnd + sifs + tolerance,
                              "CTS in response to the second RTS frame sent too late");
        NS_TEST_EXPECT_MSG_EQ(m_txPsdus[idx + 1].header.IsCts(), true, "Expected a CTS");
        NS_TEST_EXPECT_MSG_EQ(m_txPsdus[idx + 1].header.GetAddr1(),
                              apDev->GetMac()->GetAddress(),
                              "Expected a CTS frame sent to the AP");
        NS_TEST_EXPECT_MSG_EQ(
            m_txPsdus[idx + 1].header.GetDuration(),
            RoundDurationId(m_txPsdus[idx].header.GetDuration() - sifs -
                            m_txPsdus[idx + 1].txDuration),
            "Duration/ID of the CTS frame must be derived from that of the RTS frame");

        ++idx;
    }

    // the AP sends a frame to STA2
    tEnd = m_txPsdus[idx].txStart + m_txPsdus[idx].txDuration;
    ++idx;
    tStart = m_txPsdus[idx].txStart;

    NS_TEST_EXPECT_MSG_LT(tEnd + sifs, tStart, "Second QoS data frame sent too early");
    NS_TEST_EXPECT_MSG_LT(tStart, tEnd + sifs + tolerance, "Second QoS data frame sent too late");
    NS_TEST_EXPECT_MSG_EQ(m_txPsdus[idx].header.IsQosData(), true, "Expected a QoS data frame");
    NS_TEST_EXPECT_MSG_EQ(m_txPsdus[idx].header.GetAddr1(),
                          DynamicCast<WifiNetDevice>(m_staDevices.Get(1))->GetMac()->GetAddress(),
                          "Expected a frame sent by the AP to the second station");
    if (m_lengthBasedRtsCtsThresh)
    {
        NS_TEST_EXPECT_MSG_GT(m_txPsdus[idx].size,
                              rtsCtsThreshold,
                              "PSDU size expected to exceed length based RTS/CTS threshold");
    }
    else
    {
        NS_TEST_EXPECT_MSG_GT(m_txPsdus[idx].txDuration,
                              rtsCtsTxDurationThresh,
                              "PSDU duration expected to exceed duration based RTS/CTS threshold");
    }
    NS_TEST_EXPECT_MSG_EQ(
        m_txPsdus[idx].header.GetDuration(),
        (!m_protectSingleExchange
             ? RoundDurationId(m_apTxopLimit - (m_txPsdus[idx].txStart - txopStart) -
                               m_txPsdus[idx].txDuration)
             : sifs + m_txPsdus[idx + 1].txDuration + m_singleExchangeProtectionSurplus),
        (!m_protectSingleExchange
             ? "Duration/ID of the second QoS data frame does not cover the remaining TXOP"
             : "Duration/ID of the second QoS data frame must be set to remaining time to "
               "complete DATA+(Block)ACK sequence"));

    // a Normal/Block Ack is then sent by STA2
    tEnd = m_txPsdus[idx].txStart + m_txPsdus[idx].txDuration;
    tStart = m_txPsdus[idx + 1].txStart;

    NS_TEST_EXPECT_MSG_LT(tEnd + sifs,
                          tStart,
                          ack << " in response to the second QoS data frame sent too early");
    NS_TEST_EXPECT_MSG_LT(tStart,
                          tEnd + sifs + tolerance,
                          ack << " in response to the second QoS data frame sent too late");
    NS_TEST_EXPECT_MSG_EQ(
        (m_nonHt ? m_txPsdus[idx + 1].header.IsAck() : m_txPsdus[idx + 1].header.IsBlockAck()),
        true,
        "Expected a " << ack);
    NS_TEST_EXPECT_MSG_EQ(m_txPsdus[idx + 1].header.GetAddr1(),
                          apDev->GetMac()->GetAddress(),
                          "Expected a " << ack << " sent to the AP");
    NS_TEST_EXPECT_MSG_EQ(
        m_txPsdus[idx + 1].header.GetDuration(),
        RoundDurationId(m_txPsdus[idx].header.GetDuration() - sifs - m_txPsdus[idx + 1].txDuration),
        "Duration/ID of the " << ack << " must be derived from that of the previous frame");
    ++idx;

    if (!m_singleRtsPerTxop)
    {
        // An RTS frame is sent by the AP to STA3
        tEnd = m_txPsdus[idx].txStart + m_txPsdus[idx].txDuration;
        ++idx;
        tStart = m_txPsdus[idx].txStart;

        NS_TEST_EXPECT_MSG_LT(tEnd + sifs, tStart, "Third RTS frame sent too early");
        NS_TEST_EXPECT_MSG_LT(tStart, tEnd + sifs + tolerance, "Third RTS frame sent too late");
        NS_TEST_EXPECT_MSG_EQ(m_txPsdus[idx].header.IsRts(), true, "Expected an RTS frame");
        NS_TEST_EXPECT_MSG_EQ(
            m_txPsdus[idx].header.GetAddr1(),
            DynamicCast<WifiNetDevice>(m_staDevices.Get(2))->GetMac()->GetAddress(),
            "Expected an RTS frame sent by the AP to the third station");
        NS_TEST_EXPECT_MSG_EQ(
            m_txPsdus[idx].header.GetDuration(),
            (!m_protectSingleExchange
                 ? RoundDurationId(m_apTxopLimit - (m_txPsdus[idx].txStart - txopStart) -
                                   m_txPsdus[idx].txDuration)
                 : sifs + m_txPsdus[idx + 1].txDuration + sifs + m_txPsdus[idx + 2].txDuration +
                       sifs + m_txPsdus[idx + 3].txDuration + m_singleExchangeProtectionSurplus),
            (!m_protectSingleExchange
                 ? "Duration/ID of the third RTS frame must cover the whole TXOP"
                 : "Duration/ID of the third RTS frame must be set to remaining time to complete "
                   "CTS+DATA+(Block)ACK sequence"));

        // a CTS is sent by STA3 (which ignores the NAV)
        tEnd = m_txPsdus[idx].txStart + m_txPsdus[idx].txDuration;
        tStart = m_txPsdus[idx + 1].txStart;

        NS_TEST_EXPECT_MSG_LT(tEnd + sifs,
                              tStart,
                              "CTS in response to the third RTS frame sent too early");
        NS_TEST_EXPECT_MSG_LT(tStart,
                              tEnd + sifs + tolerance,
                              "CTS in response to the third RTS frame sent too late");
        NS_TEST_EXPECT_MSG_EQ(m_txPsdus[idx + 1].header.IsCts(), true, "Expected a CTS");
        NS_TEST_EXPECT_MSG_EQ(m_txPsdus[idx + 1].header.GetAddr1(),
                              apDev->GetMac()->GetAddress(),
                              "Expected a CTS frame sent to the AP");
        NS_TEST_EXPECT_MSG_EQ(
            m_txPsdus[idx + 1].header.GetDuration(),
            RoundDurationId(m_txPsdus[idx].header.GetDuration() - sifs -
                            m_txPsdus[idx + 1].txDuration),
            "Duration/ID of the CTS frame must be derived from that of the RTS frame");
        ++idx;
    }

    // the AP sends a frame to STA3
    tEnd = m_txPsdus[idx].txStart + m_txPsdus[idx].txDuration;
    ++idx;
    tStart = m_txPsdus[idx].txStart;

    NS_TEST_EXPECT_MSG_LT(tEnd + sifs, tStart, "Third QoS data frame sent too early");
    NS_TEST_EXPECT_MSG_LT(tStart, tEnd + sifs + tolerance, "Third QoS data frame sent too late");
    NS_TEST_EXPECT_MSG_EQ(m_txPsdus[idx].header.IsQosData(), true, "Expected a QoS data frame");
    NS_TEST_EXPECT_MSG_EQ(m_txPsdus[idx].header.GetAddr1(),
                          DynamicCast<WifiNetDevice>(m_staDevices.Get(2))->GetMac()->GetAddress(),
                          "Expected a frame sent by the AP to the third station");
    if (m_lengthBasedRtsCtsThresh)
    {
        NS_TEST_EXPECT_MSG_GT(m_txPsdus[idx].size,
                              rtsCtsThreshold,
                              "PSDU size expected to exceed length based RTS/CTS threshold");
    }
    else
    {
        NS_TEST_EXPECT_MSG_GT(m_txPsdus[idx].txDuration,
                              rtsCtsTxDurationThresh,
                              "PSDU duration expected to exceed duration based RTS/CTS threshold");
    }
    NS_TEST_EXPECT_MSG_EQ(
        m_txPsdus[idx].header.GetDuration(),
        (!m_protectSingleExchange
             ? RoundDurationId(m_apTxopLimit - (m_txPsdus[idx].txStart - txopStart) -
                               m_txPsdus[idx].txDuration)
             : sifs + m_txPsdus[idx + 1].txDuration + m_singleExchangeProtectionSurplus),
        (!m_protectSingleExchange
             ? "Duration/ID of the third QoS data frame does not cover the remaining TXOP"
             : "Duration/ID of the third QoS data frame must be set to remaining time to "
               "complete DATA+(Block)ACK sequence"));

    // a Normal/Block Ack is then sent by STA3
    tEnd = m_txPsdus[idx].txStart + m_txPsdus[idx].txDuration;
    tStart = m_txPsdus[idx + 1].txStart;

    NS_TEST_EXPECT_MSG_LT(tEnd + sifs,
                          tStart,
                          ack << " in response to the third QoS data frame sent too early");
    NS_TEST_EXPECT_MSG_LT(tStart,
                          tEnd + sifs + tolerance,
                          ack << " in response to the third QoS data frame sent too late");
    NS_TEST_EXPECT_MSG_EQ(
        (m_nonHt ? m_txPsdus[idx + 1].header.IsAck() : m_txPsdus[idx + 1].header.IsBlockAck()),
        true,
        "Expected a " << ack);
    NS_TEST_EXPECT_MSG_EQ(m_txPsdus[idx + 1].header.GetAddr1(),
                          apDev->GetMac()->GetAddress(),
                          "Expected a " << ack << " sent to the AP");
    NS_TEST_EXPECT_MSG_EQ(
        m_txPsdus[idx + 1].header.GetDuration(),
        RoundDurationId(m_txPsdus[idx].header.GetDuration() - sifs - m_txPsdus[idx + 1].txDuration),
        "Duration/ID of the " << ack << " must be derived from that of the previous frame");
    ++idx;

    // there is no time remaining for sending a CF-End frame if SingleRtsPerTxop is false. This is
    // verified by checking that 25 frames are transmitted (done at the beginning of this method)
    if (m_singleRtsPerTxop && !m_protectSingleExchange)
    {
        tEnd = m_txPsdus[idx].txStart + m_txPsdus[idx].txDuration;
        ++idx;
        tStart = m_txPsdus[idx].txStart;

        NS_TEST_EXPECT_MSG_LT(tEnd + sifs, tStart, "CF-End sent too early");
        NS_TEST_EXPECT_MSG_LT(tStart, tEnd + sifs + tolerance, "CF-End sent too late");
        NS_TEST_EXPECT_MSG_EQ(m_txPsdus[idx].header.IsCfEnd(), true, "Expected a CF-End frame");
        NS_TEST_EXPECT_MSG_EQ(m_txPsdus[idx].header.GetDuration(),
                              MicroSeconds(0),
                              "Duration/ID must be set to 0 for CF-End frames");
    }

    // Expected received packets:
    // - 4 DL packets (without RTS/CTS) if non-HT, 5 DL packets (without RTS/CTS) if HE
    // - 1 UL packet
    // - 3 DL packets (with RTS/CTS) if non-HT, 6 DL packets (with RTS/CTS) if HE
    NS_TEST_EXPECT_MSG_EQ(m_received, (m_nonHt ? 8 : 12), "Unexpected number of packets received");
}

/**
 * @ingroup wifi-test
 * @ingroup tests
 *
 * @brief wifi TXOP Test Suite
 */
class WifiTxopTestSuite : public TestSuite
{
  public:
    WifiTxopTestSuite();
};

WifiTxopTestSuite::WifiTxopTestSuite()
    : TestSuite("wifi-txop", Type::UNIT)
{
    for (const auto nonHt : {true, false})
    {
        for (const auto pifsRecovery : {true, false})
        {
            for (const auto singleRtsPerTxop : {true, false})
            {
                for (const auto lengthBasedRtsCtsThresh : {true, false})
                {
                    for (const auto protectedIfResponded : {true, false})
                    {
                        for (const auto protectSingleExchange : {false, true})
                        {
                            for (const auto& singleExchangeProtectionBuffer :
                                 {Time(), MicroSeconds(25) /* PIFS */})
                            {
                                if (!protectSingleExchange &&
                                    singleExchangeProtectionBuffer.IsStrictlyPositive())
                                {
                                    continue;
                                }
                                AddTestCase(new WifiTxopTest({
                                                .nonHt = nonHt,
                                                .pifsRecovery = pifsRecovery,
                                                .singleRtsPerTxop = singleRtsPerTxop,
                                                .lengthBasedRtsCtsThresh = lengthBasedRtsCtsThresh,
                                                .protectedIfResponded = protectedIfResponded,
                                                .protectSingleExchange = protectSingleExchange,
                                                .singleExchangeProtectionBuffer =
                                                    singleExchangeProtectionBuffer,
                                            }),
                                            TestCase::Duration::QUICK);
                            }
                        }
                    }
                }
            }
        }
    }
}

static WifiTxopTestSuite g_wifiTxopTestSuite; ///< the test suite
