/*
 *  Copyright (c) 2007,2008, 2009 INRIA, UDcast
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Mohamed Amine Ismail <amine.ismail@sophia.inria.fr>
 *                              <amine.ismail@udcast.com>
 */

#include "simple-ofdm-wimax-phy.h"

#include "simple-ofdm-wimax-channel.h"
#include "wimax-channel.h"
#include "wimax-mac-header.h"
#include "wimax-net-device.h"

#include "ns3/double.h"
#include "ns3/node.h"
#include "ns3/packet-burst.h"
#include "ns3/packet.h"
#include "ns3/simulator.h"
#include "ns3/string.h"
#include "ns3/trace-source-accessor.h"
#include "ns3/uinteger.h"

#include <cmath>
#include <string>
#include <vector>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("SimpleOfdmWimaxPhy");

NS_OBJECT_ENSURE_REGISTERED(SimpleOfdmWimaxPhy);

TypeId
SimpleOfdmWimaxPhy::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::SimpleOfdmWimaxPhy")
            .SetParent<WimaxPhy>()
            .SetGroupName("Wimax")

            .AddConstructor<SimpleOfdmWimaxPhy>()

            .AddAttribute(
                "NoiseFigure",
                "Loss (dB) in the Signal-to-Noise-Ratio due to non-idealities in the receiver.",
                DoubleValue(5),
                MakeDoubleAccessor(&SimpleOfdmWimaxPhy::SetNoiseFigure,
                                   &SimpleOfdmWimaxPhy::GetNoiseFigure),
                MakeDoubleChecker<double>())

            .AddAttribute("TxPower",
                          "Transmission power (dB).",
                          DoubleValue(30),
                          MakeDoubleAccessor(&SimpleOfdmWimaxPhy::SetTxPower,
                                             &SimpleOfdmWimaxPhy::GetTxPower),
                          MakeDoubleChecker<double>())

            .AddAttribute("G",
                          "This is the ratio of CP time to useful time.",
                          DoubleValue(0.25),
                          MakeDoubleAccessor(&SimpleOfdmWimaxPhy::DoSetGValue,
                                             &SimpleOfdmWimaxPhy::DoGetGValue),
                          MakeDoubleChecker<double>())

            .AddAttribute(
                "TxGain",
                "Transmission gain (dB).",
                DoubleValue(0),
                MakeDoubleAccessor(&SimpleOfdmWimaxPhy::SetTxGain, &SimpleOfdmWimaxPhy::GetTxGain),
                MakeDoubleChecker<double>())

            .AddAttribute(
                "RxGain",
                "Reception gain (dB).",
                DoubleValue(0),
                MakeDoubleAccessor(&SimpleOfdmWimaxPhy::SetRxGain, &SimpleOfdmWimaxPhy::GetRxGain),
                MakeDoubleChecker<double>())

            .AddAttribute("Nfft",
                          "FFT size",
                          UintegerValue(256),
                          MakeUintegerAccessor(&SimpleOfdmWimaxPhy::DoSetNfft,
                                               &SimpleOfdmWimaxPhy::DoGetNfft),
                          MakeUintegerChecker<uint16_t>(256, 1024))

            .AddAttribute("TraceFilePath",
                          "Path to the directory containing SNR to block error rate files",
                          StringValue(""),
                          MakeStringAccessor(&SimpleOfdmWimaxPhy::GetTraceFilePath,
                                             &SimpleOfdmWimaxPhy::SetTraceFilePath),
                          MakeStringChecker())

            .AddTraceSource("Rx",
                            "Receive trace",
                            MakeTraceSourceAccessor(&SimpleOfdmWimaxPhy::m_traceRx),
                            "ns3::PacketBurst::TracedCallback")
            .AddTraceSource("Tx",
                            "Transmit trace",
                            MakeTraceSourceAccessor(&SimpleOfdmWimaxPhy::m_traceTx),
                            "ns3::PacketBurst::TracedCallback")

            .AddTraceSource(
                "PhyTxBegin",
                "Trace source indicating a packet has begun transmitting over the channel medium",
                MakeTraceSourceAccessor(&SimpleOfdmWimaxPhy::m_phyTxBeginTrace),
                "ns3::PacketBurst::TracedCallback")

            .AddTraceSource(
                "PhyTxEnd",
                "Trace source indicating a packet has been completely transmitted over the channel",
                MakeTraceSourceAccessor(&SimpleOfdmWimaxPhy::m_phyTxEndTrace),
                "ns3::PacketBurst::TracedCallback")

            .AddTraceSource("PhyTxDrop",
                            "Trace source indicating a packet has been dropped by the device "
                            "during transmission",
                            MakeTraceSourceAccessor(&SimpleOfdmWimaxPhy::m_phyTxDropTrace),
                            "ns3::PacketBurst::TracedCallback")

            .AddTraceSource("PhyRxBegin",
                            "Trace source indicating a packet has begun being received from the "
                            "channel medium by the device",
                            MakeTraceSourceAccessor(&SimpleOfdmWimaxPhy::m_phyRxBeginTrace),
                            "ns3::PacketBurst::TracedCallback")

            .AddTraceSource("PhyRxEnd",
                            "Trace source indicating a packet has been completely received from "
                            "the channel medium by the device",
                            MakeTraceSourceAccessor(&SimpleOfdmWimaxPhy::m_phyRxEndTrace),
                            "ns3::PacketBurst::TracedCallback")

            .AddTraceSource(
                "PhyRxDrop",
                "Trace source indicating a packet has been dropped by the device during reception",
                MakeTraceSourceAccessor(&SimpleOfdmWimaxPhy::m_phyRxDropTrace),
                "ns3::PacketBurst::TracedCallback");
    return tid;
}

void
SimpleOfdmWimaxPhy::InitSimpleOfdmWimaxPhy()
{
    m_fecBlockSize = 0;
    m_nrFecBlocksSent = 0;
    m_dataRateBpsk12 = 0;
    m_dataRateQpsk12 = 0;
    m_dataRateQpsk34 = 0;
    m_dataRateQam16_12 = 0;

    m_dataRateQam16_34 = 0;
    m_dataRateQam64_23 = 0;
    m_dataRateQam64_34 = 0;

    m_nrBlocks = 0;
    m_blockSize = 0;
    m_paddingBits = 0;
    m_rxGain = 0;
    m_txGain = 0;
    m_nfft = 256;
    m_g = 1.0 / 4;
    SetNrCarriers(192);
    m_fecBlocks = new std::list<Bvec>;
    m_receivedFecBlocks = new std::list<Bvec>;
    m_currentBurstSize = 0;
    m_noiseFigure = 5;      // dB
    m_txPower = 30;         // dBm
    SetBandwidth(10000000); // 10Mhz
    m_nbErroneousBlock = 0;
    m_nrReceivedFecBlocks = 0;
    m_snrToBlockErrorRateManager = new SNRToBlockErrorRateManager();
}

SimpleOfdmWimaxPhy::SimpleOfdmWimaxPhy()
{
    m_URNG = CreateObject<UniformRandomVariable>();

    InitSimpleOfdmWimaxPhy();
    m_snrToBlockErrorRateManager->SetTraceFilePath((char*)"");
    m_snrToBlockErrorRateManager->LoadTraces();
}

SimpleOfdmWimaxPhy::SimpleOfdmWimaxPhy(char* tracesPath)
{
    InitSimpleOfdmWimaxPhy();
    m_snrToBlockErrorRateManager->SetTraceFilePath(tracesPath);
    m_snrToBlockErrorRateManager->LoadTraces();
}

SimpleOfdmWimaxPhy::~SimpleOfdmWimaxPhy()
{
}

void
SimpleOfdmWimaxPhy::ActivateLoss(bool loss)
{
    m_snrToBlockErrorRateManager->ActivateLoss(loss);
}

void
SimpleOfdmWimaxPhy::SetSNRToBlockErrorRateTracesPath(char* tracesPath)
{
    m_snrToBlockErrorRateManager->SetTraceFilePath(tracesPath);
    m_snrToBlockErrorRateManager->ReLoadTraces();
}

uint32_t
SimpleOfdmWimaxPhy::GetBandwidth() const
{
    return WimaxPhy::GetChannelBandwidth();
}

void
SimpleOfdmWimaxPhy::SetBandwidth(uint32_t BW)
{
    WimaxPhy::SetChannelBandwidth(BW);
}

double
SimpleOfdmWimaxPhy::GetTxPower() const
{
    return m_txPower;
}

void
SimpleOfdmWimaxPhy::SetTxPower(double txPower)
{
    m_txPower = txPower;
}

double
SimpleOfdmWimaxPhy::GetNoiseFigure() const
{
    return m_noiseFigure;
}

void
SimpleOfdmWimaxPhy::SetNoiseFigure(double noiseFigure)
{
    m_noiseFigure = noiseFigure;
}

void
SimpleOfdmWimaxPhy::DoDispose()
{
    delete m_receivedFecBlocks;
    delete m_fecBlocks;
    m_receivedFecBlocks = nullptr;
    m_fecBlocks = nullptr;
    delete m_snrToBlockErrorRateManager;
    WimaxPhy::DoDispose();
}

void
SimpleOfdmWimaxPhy::DoAttach(Ptr<WimaxChannel> channel)
{
    GetChannel()->Attach(this);
}

void
SimpleOfdmWimaxPhy::Send(SendParams* params)
{
    auto o_params = dynamic_cast<OfdmSendParams*>(params);
    NS_ASSERT(o_params != nullptr);
    Send(o_params->GetBurst(),
         (WimaxPhy::ModulationType)o_params->GetModulationType(),
         o_params->GetDirection());
}

WimaxPhy::PhyType
SimpleOfdmWimaxPhy::GetPhyType() const
{
    return WimaxPhy::simpleOfdmWimaxPhy;
}

void
SimpleOfdmWimaxPhy::Send(Ptr<PacketBurst> burst,
                         WimaxPhy::ModulationType modulationType,
                         uint8_t direction)
{
    if (GetState() != PHY_STATE_TX)
    {
        m_currentBurstSize = burst->GetSize();
        m_nrFecBlocksSent = 0;
        m_currentBurst = burst;
        SetBlockParameters(burst->GetSize(), modulationType);
        NotifyTxBegin(m_currentBurst);
        StartSendDummyFecBlock(true, modulationType, direction);
        m_traceTx(burst);
    }
}

void
SimpleOfdmWimaxPhy::StartSendDummyFecBlock(bool isFirstBlock,
                                           WimaxPhy::ModulationType modulationType,
                                           uint8_t direction)
{
    SetState(PHY_STATE_TX);
    bool isLastFecBlock = false;
    if (isFirstBlock)
    {
        m_blockTime = GetBlockTransmissionTime(modulationType);
    }

    SimpleOfdmWimaxChannel* channel =
        dynamic_cast<SimpleOfdmWimaxChannel*>(PeekPointer(GetChannel()));
    NS_ASSERT(channel != nullptr);

    isLastFecBlock = (m_nrRemainingBlocksToSend == 1);
    channel->Send(m_blockTime,
                  m_currentBurstSize,
                  this,
                  isFirstBlock,
                  isLastFecBlock,
                  GetTxFrequency(),
                  modulationType,
                  direction,
                  m_txPower,
                  m_currentBurst);

    m_nrRemainingBlocksToSend--;
    Simulator::Schedule(m_blockTime,
                        &SimpleOfdmWimaxPhy::EndSendFecBlock,
                        this,
                        modulationType,
                        direction);
}

void
SimpleOfdmWimaxPhy::EndSendFecBlock(WimaxPhy::ModulationType modulationType, uint8_t direction)
{
    m_nrFecBlocksSent++;
    SetState(PHY_STATE_IDLE);

    if (m_nrFecBlocksSent * m_blockSize == m_currentBurstSize * 8 + m_paddingBits)
    {
        // this is the last FEC block of the burst
        NS_ASSERT_MSG(m_nrRemainingBlocksToSend == 0, "Error while sending a burst");
        NotifyTxEnd(m_currentBurst);
    }
    else
    {
        StartSendDummyFecBlock(false, modulationType, direction);
    }
}

void
SimpleOfdmWimaxPhy::EndSend()
{
    SetState(PHY_STATE_IDLE);
}

void
SimpleOfdmWimaxPhy::StartReceive(uint32_t burstSize,
                                 bool isFirstBlock,
                                 uint64_t frequency,
                                 WimaxPhy::ModulationType modulationType,
                                 uint8_t direction,
                                 double rxPower,
                                 Ptr<PacketBurst> burst)
{
    bool drop = false;
    double Nwb = -114 + m_noiseFigure + 10 * std::log(GetBandwidth() / 1000000000.0) / 2.303;
    double SNR = rxPower - Nwb;

    SNRToBlockErrorRateRecord* record =
        m_snrToBlockErrorRateManager->GetSNRToBlockErrorRateRecord(SNR, modulationType);
    double I1 = record->GetI1();
    double I2 = record->GetI2();

    double blockErrorRate = m_URNG->GetValue(I1, I2);

    double rand = m_URNG->GetValue(0.0, 1.0);

    if (rand < blockErrorRate)
    {
        drop = true;
    }
    if (rand > blockErrorRate)
    {
        drop = false;
    }

    if (blockErrorRate == 1.0)
    {
        drop = true;
    }
    if (blockErrorRate == 0.0)
    {
        drop = false;
    }
    delete record;

    NS_LOG_INFO("PHY: Receive rxPower=" << rxPower << ", Nwb=" << Nwb << ", SNR=" << SNR
                                        << ", Modulation=" << modulationType << ", BlockErrorRate="
                                        << blockErrorRate << ", drop=" << std::boolalpha << drop);

    switch (GetState())
    {
    case PHY_STATE_SCANNING:
        if (frequency == GetScanningFrequency())
        {
            Simulator::Cancel(GetChnlSrchTimeoutEvent());
            SetScanningCallback();
            SetSimplex(frequency);
            SetState(PHY_STATE_IDLE);
        }
        break;
    case PHY_STATE_IDLE:
        if (frequency == GetRxFrequency())
        {
            if (isFirstBlock)
            {
                NotifyRxBegin(burst);
                m_receivedFecBlocks->clear();
                m_nrReceivedFecBlocks = 0;
                SetBlockParameters(burstSize, modulationType);
                m_blockTime = GetBlockTransmissionTime(modulationType);
            }

            Simulator::Schedule(m_blockTime,
                                &SimpleOfdmWimaxPhy::EndReceiveFecBlock,
                                this,
                                burstSize,
                                modulationType,
                                direction,
                                drop,
                                burst);

            SetState(PHY_STATE_RX);
        }
        break;
    case PHY_STATE_RX:
        // drop
        break;
    case PHY_STATE_TX:
        if (IsDuplex() && frequency == GetRxFrequency())
        {
        }
        break;
    }
}

void
SimpleOfdmWimaxPhy::EndReceiveFecBlock(uint32_t burstSize,
                                       WimaxPhy::ModulationType modulationType,
                                       uint8_t direction,
                                       bool drop,
                                       Ptr<PacketBurst> burst)
{
    SetState(PHY_STATE_IDLE);
    m_nrReceivedFecBlocks++;

    if (drop)
    {
        m_nbErroneousBlock++;
    }

    if ((uint32_t)m_nrReceivedFecBlocks * m_blockSize == burstSize * 8 + m_paddingBits)
    {
        NotifyRxEnd(burst);
        if (m_nbErroneousBlock == 0)
        {
            Simulator::Schedule(Seconds(0), &SimpleOfdmWimaxPhy::EndReceive, this, burst);
        }
        else
        {
            NotifyRxDrop(burst);
        }
        m_nbErroneousBlock = 0;
        m_nrReceivedFecBlocks = 0;
    }
}

void
SimpleOfdmWimaxPhy::EndReceive(Ptr<const PacketBurst> burst)
{
    Ptr<PacketBurst> b = burst->Copy();
    GetReceiveCallback()(b);
    m_traceRx(burst);
}

Bvec
SimpleOfdmWimaxPhy::ConvertBurstToBits(Ptr<const PacketBurst> burst)
{
    Bvec buffer(burst->GetSize() * 8, false);

    std::list<Ptr<Packet>> packets = burst->GetPackets();

    uint32_t j = 0;
    for (auto iter = packets.begin(); iter != packets.end(); ++iter)
    {
        Ptr<Packet> packet = *iter;
        auto pstart = (uint8_t*)std::malloc(packet->GetSize());
        std::memset(pstart, 0, packet->GetSize());
        packet->CopyData(pstart, packet->GetSize());
        Bvec temp(8);
        temp.resize(0, false);
        temp.resize(8, false);
        for (uint32_t i = 0; i < packet->GetSize(); i++)
        {
            for (uint8_t l = 0; l < 8; l++)
            {
                temp[l] = (bool)((((uint8_t)pstart[i]) >> (7 - l)) & 0x01);
                buffer.at(j * 8 + l) = temp[l];
            }
            j++;
        }
        std::free(pstart);
    }

    return buffer;
}

/*
 Converts back the bit buffer (Bvec) to the actual burst.
 Actually creates byte buffer from the Bvec and resets the buffer
 of each packet in the copy of the original burst stored before transmitting.
 By doing this it preserves the metadata and tags in the packet.
 Function could also be named DeserializeBurst because actually it
 copying to the burst's byte buffer.
 */
Ptr<PacketBurst>
SimpleOfdmWimaxPhy::ConvertBitsToBurst(Bvec buffer)
{
    const auto bufferSize = buffer.size() / 8;
    std::vector<uint8_t> bytes(bufferSize, 0);
    int32_t j = 0;
    // recreating byte buffer from bit buffer (Bvec)
    for (std::size_t i = 0; i < buffer.size(); i += 8)
    {
        uint8_t temp = 0;
        for (std::size_t l = 0; l < 8; l++)
        {
            bool bin = buffer.at(i + l);
            temp |= (bin << (7 - l));
        }

        bytes[j] = temp;
        j++;
    }
    uint16_t pos = 0;
    Ptr<PacketBurst> RecvBurst = CreateObject<PacketBurst>();
    while (pos < bufferSize)
    {
        uint16_t packetSize = 0;
        // Get the header type: first bit
        uint8_t ht = (bytes[pos] >> 7) & 0x01;
        if (ht == 1)
        {
            // BW request header. Size is always 8 bytes
            packetSize = 6;
        }
        else
        {
            // Read the size
            uint8_t Len_MSB = bytes[pos + 1] & 0x07;
            packetSize = (uint16_t)((uint16_t)(Len_MSB << 8) | (uint16_t)(bytes[pos + 2]));
            if (packetSize == 0)
            {
                break; // padding
            }
        }

        Ptr<Packet> p = Create<Packet>(&bytes[pos], packetSize);
        RecvBurst->AddPacket(p);
        pos += packetSize;
    }
    return RecvBurst;
}

void
SimpleOfdmWimaxPhy::CreateFecBlocks(const Bvec& buffer, WimaxPhy::ModulationType modulationType)
{
    Bvec fecBlock(m_blockSize);
    for (uint32_t i = 0, j = m_nrBlocks; j > 0; i += m_blockSize, j--)
    {
        if (j == 1 && m_paddingBits > 0) // last block can be smaller than block size
        {
            fecBlock = Bvec(buffer.begin() + i, buffer.end());
            fecBlock.resize(m_blockSize, false);
        }
        else
        {
            fecBlock = Bvec(buffer.begin() + i, buffer.begin() + i + m_blockSize);
        }

        m_fecBlocks->push_back(fecBlock);
    }
}

Bvec
SimpleOfdmWimaxPhy::RecreateBuffer()
{
    Bvec buffer(m_blockSize * (unsigned long)m_nrBlocks);
    Bvec block(m_blockSize);
    uint32_t i = 0;
    for (uint32_t j = 0; j < m_nrBlocks; j++)
    {
        Bvec tmpRecFecBlock = m_receivedFecBlocks->front();
        buffer.insert(buffer.begin() + i, tmpRecFecBlock.begin(), tmpRecFecBlock.end());
        m_receivedFecBlocks->pop_front();
        i += m_blockSize;
    }
    return buffer;
}

void
SimpleOfdmWimaxPhy::DoSetDataRates()
{
    m_dataRateBpsk12 = CalculateDataRate(MODULATION_TYPE_BPSK_12);    // 6912000 bps
    m_dataRateQpsk12 = CalculateDataRate(MODULATION_TYPE_QPSK_12);    // 13824000
    m_dataRateQpsk34 = CalculateDataRate(MODULATION_TYPE_QPSK_34);    // 20736000
    m_dataRateQam16_12 = CalculateDataRate(MODULATION_TYPE_QAM16_12); // 27648000
    m_dataRateQam16_34 = CalculateDataRate(MODULATION_TYPE_QAM16_34); // 41472000
    m_dataRateQam64_23 = CalculateDataRate(MODULATION_TYPE_QAM64_23); // 55224000
    m_dataRateQam64_34 = CalculateDataRate(MODULATION_TYPE_QAM64_34); // 62208000
}

void
SimpleOfdmWimaxPhy::GetModulationFecParams(WimaxPhy::ModulationType modulationType,
                                           uint8_t& bitsPerSymbol,
                                           double& fecCode) const
{
    switch (modulationType)
    {
    case MODULATION_TYPE_BPSK_12:
        bitsPerSymbol = 1;
        fecCode = 1.0 / 2;
        break;
    case MODULATION_TYPE_QPSK_12:
        bitsPerSymbol = 2;
        fecCode = 1.0 / 2;
        break;
    case MODULATION_TYPE_QPSK_34:
        bitsPerSymbol = 2;
        fecCode = 3.0 / 4;
        break;
    case MODULATION_TYPE_QAM16_12:
        bitsPerSymbol = 4;
        fecCode = 1.0 / 2;
        break;
    case MODULATION_TYPE_QAM16_34:
        bitsPerSymbol = 4;
        fecCode = 3.0 / 4;
        break;
    case MODULATION_TYPE_QAM64_23:
        bitsPerSymbol = 6;
        fecCode = 2.0 / 3;
        break;
    case MODULATION_TYPE_QAM64_34:
        bitsPerSymbol = 6;
        fecCode = 0.75;
        break;
    }
}

uint32_t
SimpleOfdmWimaxPhy::CalculateDataRate(WimaxPhy::ModulationType modulationType) const
{
    uint8_t bitsPerSymbol = 0;
    double fecCode = 0;
    GetModulationFecParams(modulationType, bitsPerSymbol, fecCode);
    double symbolsPerSecond = 1 / GetSymbolDuration().GetSeconds();
    auto bitsTransmittedPerSymbol = (uint16_t)(bitsPerSymbol * GetNrCarriers() * fecCode);
    // 96, 192, 288, 384, 576, 767 and 864 bits per symbol for the seven modulations, respectively

    return (uint32_t)(symbolsPerSecond * bitsTransmittedPerSymbol);
}

uint32_t
SimpleOfdmWimaxPhy::DoGetDataRate(WimaxPhy::ModulationType modulationType) const
{
    switch (modulationType)
    {
    case MODULATION_TYPE_BPSK_12:
        return m_dataRateBpsk12;
    case MODULATION_TYPE_QPSK_12:
        return m_dataRateQpsk12;
    case MODULATION_TYPE_QPSK_34:
        return m_dataRateQpsk34;
    case MODULATION_TYPE_QAM16_12:
        return m_dataRateQam16_12;
    case MODULATION_TYPE_QAM16_34:
        return m_dataRateQam16_34;
    case MODULATION_TYPE_QAM64_23:
        return m_dataRateQam64_23;
    case MODULATION_TYPE_QAM64_34:
        return m_dataRateQam64_34;
    }
    NS_FATAL_ERROR("Invalid modulation type");
    return 0;
}

Time
SimpleOfdmWimaxPhy::GetBlockTransmissionTime(WimaxPhy::ModulationType modulationType) const
{
    return Seconds((double)GetFecBlockSize(modulationType) / DoGetDataRate(modulationType));
}

Time
SimpleOfdmWimaxPhy::DoGetTransmissionTime(uint32_t size,
                                          WimaxPhy::ModulationType modulationType) const
{
    /*adding 3 extra nano second to cope with the loss of precision problem.
     the time is internally stored in a 64 bit hence a floating-point time would loss
     precision, e.g., 0.00001388888888888889 seconds will become 13888888888 femtoseconds.*/
    return Seconds(DoGetNrSymbols(size, modulationType) * GetSymbolDuration().GetSeconds()) +
           NanoSeconds(3);
}

uint64_t
SimpleOfdmWimaxPhy::DoGetNrSymbols(uint32_t size, WimaxPhy::ModulationType modulationType) const
{
    Time transmissionTime =
        Seconds((double)(GetNrBlocks(size, modulationType) * GetFecBlockSize(modulationType)) /
                DoGetDataRate(modulationType));
    return (uint64_t)std::ceil(transmissionTime.GetSeconds() / GetSymbolDuration().GetSeconds());
}

uint64_t
SimpleOfdmWimaxPhy::DoGetNrBytes(uint32_t symbols, WimaxPhy::ModulationType modulationType) const
{
    Time transmissionTime = Seconds(symbols * GetSymbolDuration().GetSeconds());
    return (uint64_t)std::floor((transmissionTime.GetSeconds() * DoGetDataRate(modulationType)) /
                                8);
}

uint32_t
SimpleOfdmWimaxPhy::GetFecBlockSize(WimaxPhy::ModulationType modulationType) const
{
    uint32_t blockSize = 0;
    switch (modulationType)
    {
    case MODULATION_TYPE_BPSK_12:
        blockSize = 12;
        break;
    case MODULATION_TYPE_QPSK_12:
        blockSize = 24;
        break;
    case MODULATION_TYPE_QPSK_34:
        blockSize = 36;
        break;
    case MODULATION_TYPE_QAM16_12:
        blockSize = 48;
        break;
    case MODULATION_TYPE_QAM16_34:
        blockSize = 72;
        break;
    case MODULATION_TYPE_QAM64_23:
        blockSize = 96;
        break;
    case MODULATION_TYPE_QAM64_34:
        blockSize = 108;
        break;
    default:
        NS_FATAL_ERROR("Invalid modulation type");
        break;
    }
    return blockSize * 8; // in bits
}

// Channel coding block size, Table 215, page 434
uint32_t
SimpleOfdmWimaxPhy::GetCodedFecBlockSize(WimaxPhy::ModulationType modulationType) const
{
    uint32_t blockSize = 0;
    switch (modulationType)
    {
    case MODULATION_TYPE_BPSK_12:
        blockSize = 24;
        break;
    case MODULATION_TYPE_QPSK_12:
    case MODULATION_TYPE_QPSK_34:
        blockSize = 48;
        break;
    case MODULATION_TYPE_QAM16_12:
    case MODULATION_TYPE_QAM16_34:
        blockSize = 96;
        break;
    case MODULATION_TYPE_QAM64_23:
    case MODULATION_TYPE_QAM64_34:
        blockSize = 144;
        break;
    default:
        NS_FATAL_ERROR("Invalid modulation type");
        break;
    }
    return blockSize * 8; // in bits
}

void
SimpleOfdmWimaxPhy::SetBlockParameters(uint32_t burstSize, WimaxPhy::ModulationType modulationType)
{
    m_blockSize = GetFecBlockSize(modulationType);
    m_nrBlocks = GetNrBlocks(burstSize, modulationType);
    m_paddingBits = (m_nrBlocks * m_blockSize) - (burstSize * 8);
    m_nrRemainingBlocksToSend = m_nrBlocks;
    NS_ASSERT_MSG(static_cast<uint32_t>(m_nrBlocks * m_blockSize) >= (burstSize * 8),
                  "Size of padding bytes < 0");
}

uint16_t
SimpleOfdmWimaxPhy::DoGetTtg() const
{
    // assumed equal to 2 symbols
    return 2 * GetPsPerSymbol();
}

uint16_t
SimpleOfdmWimaxPhy::DoGetRtg() const
{
    // assumed equal to 2 symbols
    return 2 * GetPsPerSymbol();
}

uint8_t
SimpleOfdmWimaxPhy::DoGetFrameDurationCode() const
{
    uint16_t duration = 0;
    duration = (uint16_t)(GetFrameDuration().GetSeconds() * 10000);
    uint8_t retval = 0;
    switch (duration)
    {
    case 25: {
        retval = FRAME_DURATION_2_POINT_5_MS;
        break;
    }
    case 40: {
        retval = FRAME_DURATION_4_MS;
        break;
    }
    case 50: {
        retval = FRAME_DURATION_5_MS;
        break;
    }
    case 80: {
        retval = FRAME_DURATION_8_MS;
        break;
    }
    case 100: {
        retval = FRAME_DURATION_10_MS;
        break;
    }
    case 125: {
        retval = FRAME_DURATION_12_POINT_5_MS;
        break;
    }
    case 200: {
        retval = FRAME_DURATION_20_MS;
        break;
    }
    default: {
        NS_FATAL_ERROR("Invalid frame duration = " << duration);
        retval = 0;
    }
    }
    return retval;
}

Time
SimpleOfdmWimaxPhy::DoGetFrameDuration(uint8_t frameDurationCode) const
{
    switch (frameDurationCode)
    {
    case FRAME_DURATION_2_POINT_5_MS:
        return Seconds(2.5);
    case FRAME_DURATION_4_MS:
        return Seconds(4);
    case FRAME_DURATION_5_MS:
        return Seconds(5);
    case FRAME_DURATION_8_MS:
        return Seconds(8);
    case FRAME_DURATION_10_MS:
        return Seconds(10);
    case FRAME_DURATION_12_POINT_5_MS:
        return Seconds(12.5);
    case FRAME_DURATION_20_MS:
        return Seconds(20);
    default:
        NS_FATAL_ERROR("Invalid modulation type");
    }
    return Seconds(0);
}

/*
 Returns number of blocks (FEC blocks) the burst will be split in.
 The size of the block is specific for each modulation type.
 */
uint16_t
SimpleOfdmWimaxPhy::GetNrBlocks(uint32_t burstSize, WimaxPhy::ModulationType modulationType) const
{
    uint32_t blockSize = GetFecBlockSize(modulationType);
    uint16_t nrBlocks = (burstSize * 8) / blockSize;

    if ((burstSize * 8) % blockSize > 0)
    {
        nrBlocks += 1;
    }

    return nrBlocks;
}

/*---------------------PHY parameters functions-----------------------*/

void
SimpleOfdmWimaxPhy::DoSetPhyParameters()
{
    /*Calculations as per section 8.3.2.
     Currently assuming license-exempt 5 GHz band. For channel bandwidth 20 MHz (Table B.28, page
     812) and frame duration 10 ms (Table 232, page 460) i.e, 100 frames per second, sampling
     frequency is 23040000, symbol (OFDM symbol) duration is 1.388888888888889e-05 seconds, PS
     duration is 1.7361111111111112e-07 seconds. Hence PSs per frame is 57600, symbols per frame is
     720 and PSs per symbol is 80. Note that defining these parameters (symbol and PS duration) as
     Time may not result in exactly these values therefore lrint has been used (otherwise should be
     defined as double). For licensed bands set channel bandwidth according to Table B.26, page
     810.*/

    double samplingFrequency = DoGetSamplingFrequency();
    Time psDuration = Seconds(4.0 / samplingFrequency);

    SetPsDuration(psDuration);
    uint16_t psPerFrame = (uint16_t)(GetFrameDuration().GetSeconds() / psDuration.GetSeconds());
    SetPsPerFrame(psPerFrame);
    double subcarrierSpacing = samplingFrequency / DoGetNfft();
    double tb = 1.0 / subcarrierSpacing;    // Tb (useful symbol time)
    double tg = DoGetGValue() * tb;         // Tg (cyclic prefix time)
    Time symbolDuration = Seconds(tb + tg); // OFDM Symbol Time
    SetSymbolDuration(symbolDuration);
    uint16_t psPerSymbol = lrint(symbolDuration.GetSeconds() / psDuration.GetSeconds());
    SetPsPerSymbol(psPerSymbol);
    uint32_t symbolsPerFrame = lrint(GetFrameDuration().GetSeconds() / symbolDuration.GetSeconds());
    SetSymbolsPerFrame(symbolsPerFrame);
}

void
SimpleOfdmWimaxPhy::DoSetNfft(uint16_t nfft)
{
    m_nfft = nfft;
}

uint16_t
SimpleOfdmWimaxPhy::DoGetNfft() const
{
    return m_nfft;
}

double
SimpleOfdmWimaxPhy::DoGetSamplingFactor() const
{
    // sampling factor (n), see Table 213, page 429

    uint32_t channelBandwidth = GetChannelBandwidth();

    if (channelBandwidth % 1750000 == 0)
    {
        return 8.0 / 7;
    }
    else if (channelBandwidth % 1500000 == 0)
    {
        return 86.0 / 75;
    }
    else if (channelBandwidth % 1250000 == 0)
    {
        return 144.0 / 125;
    }
    else if (channelBandwidth % 2750000 == 0)
    {
        return 316.0 / 275;
    }
    else if (channelBandwidth % 2000000 == 0)
    {
        return 57.0 / 50;
    }
    else
    {
        NS_LOG_DEBUG("Oops may be wrong channel bandwidth for OFDM PHY!");
        NS_FATAL_ERROR("wrong channel bandwidth for OFDM PHY");
    }

    return 8.0 / 7;
}

double
SimpleOfdmWimaxPhy::DoGetSamplingFrequency() const
{
    // sampling frequency (Fs), see 8.3.2.2

    return (DoGetSamplingFactor() * GetChannelBandwidth() / 8000) * 8000;
}

double
SimpleOfdmWimaxPhy::DoGetGValue() const
{
    return m_g;
}

void
SimpleOfdmWimaxPhy::DoSetGValue(double g)
{
    m_g = g;
}

void
SimpleOfdmWimaxPhy::SetTxGain(double txGain)
{
    m_txGain = txGain;
}

void
SimpleOfdmWimaxPhy::SetRxGain(double txRain)
{
    m_rxGain = txRain;
}

double
SimpleOfdmWimaxPhy::GetTxGain() const
{
    return m_txGain;
}

double
SimpleOfdmWimaxPhy::GetRxGain() const
{
    return m_rxGain;
}

std::string
SimpleOfdmWimaxPhy::GetTraceFilePath() const
{
    return (m_snrToBlockErrorRateManager->GetTraceFilePath());
}

void
SimpleOfdmWimaxPhy::SetTraceFilePath(std::string path)
{
    m_snrToBlockErrorRateManager->SetTraceFilePath((char*)path.c_str());
    m_snrToBlockErrorRateManager->LoadTraces();
}

void
SimpleOfdmWimaxPhy::NotifyTxBegin(Ptr<PacketBurst> burst)
{
    m_phyTxBeginTrace(burst);
}

void
SimpleOfdmWimaxPhy::NotifyTxEnd(Ptr<PacketBurst> burst)
{
    m_phyTxEndTrace(burst);
}

void
SimpleOfdmWimaxPhy::NotifyTxDrop(Ptr<PacketBurst> burst)
{
    m_phyTxDropTrace(burst);
}

void
SimpleOfdmWimaxPhy::NotifyRxBegin(Ptr<PacketBurst> burst)
{
    m_phyRxBeginTrace(burst);
}

void
SimpleOfdmWimaxPhy::NotifyRxEnd(Ptr<PacketBurst> burst)
{
    m_phyRxEndTrace(burst);
}

void
SimpleOfdmWimaxPhy::NotifyRxDrop(Ptr<PacketBurst> burst)
{
    m_phyRxDropTrace(burst);
}

int64_t
SimpleOfdmWimaxPhy::AssignStreams(int64_t stream)
{
    NS_LOG_FUNCTION(this << stream);
    m_URNG->SetStream(stream);
    return 1;
}

} // namespace ns3
