/*
 * Copyright (c) 2026 Centre Tecnològic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
 */

#include "tcp-error-model.h"
#include "tcp-general-test.h"

#include "ns3/boolean.h"
#include "ns3/log.h"
#include "ns3/node.h"
#include "ns3/string.h"
#include "ns3/tcp-header.h"
#include "ns3/tcp-option-sack.h"
#include "ns3/tcp-option-winscale.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/test.h"
#include "ns3/uinteger.h"

#include <algorithm>
#include <map>
#include <string>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TcpLinuxRcvWindowTestSuite");

/**
 * @ingroup internet-test
 *
 * @brief Check the windows advertised with the LinuxRcvWindow attribute.
 *
 * RcvBufSize 65536 gives a 131072-byte buffer (Linux doubles SO_RCVBUF);
 * segments are 1448 bytes. Expected, as Linux:
 * - SYN+ACK: half the buffer rounded down to whole segments, 65160 bytes,
 *   window scale 2 from net.core.rmem_max 212992 (tcp_select_initial_window()).
 * - Established bound: half the buffer minus one segment, 64088 bytes
 *   (tcp_init_buffer_space()).
 * - After the first segment (1448 bytes in a 2784-byte skb): data/memory
 *   ratio 133/256, bound 131072 * 133 / 256 = 68096; window
 *   (131072 - 2784) * 133 / 256 = 66649, rounded up to the scale unit: 66652
 *   (tcp_grow_window(), __tcp_select_window()).
 * - After a loss, the ACK of the retransmission (sent before the application
 *   reads the delivered out-of-order data) advertises less than before the
 *   loss, but never shrinks the right edge (tcp_select_window()).
 * Without the attribute, the window is RcvBufSize.
 */
class TcpLinuxRcvWindowTest : public TcpGeneralTest
{
  public:
    /**
     * @brief Constructor.
     * @param name Test description.
     * @param linuxWindow Value of the LinuxRcvWindow attribute.
     */
    TcpLinuxRcvWindowTest(const std::string& name, bool linuxWindow)
        : TcpGeneralTest(name),
          m_linux(linuxWindow)
    {
    }

  protected:
    void ConfigureProperties() override;
    void ConfigureEnvironment() override;
    Ptr<TcpSocketMsgBase> CreateReceiverSocket(Ptr<Node> node) override;
    Ptr<ErrorModel> CreateReceiverErrorModel() override;
    void Rx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override;
    void FinalChecks() override;

  private:
    static constexpr uint32_t SEGMENT = 1448;             //!< Data bytes of a segment
    static constexpr uint32_t LOST_SEQ = 1 + 5 * SEGMENT; //!< Lost segment

    bool m_linux;                    //!< Value of the LinuxRcvWindow attribute
    uint8_t m_wscale{0};             //!< Window scale of the receiver
    uint32_t m_synAckWindow{0};      //!< Window field of the SYN+ACK
    std::vector<uint32_t> m_acks;    //!< Acknowledgment numbers, relative
    std::vector<uint32_t> m_windows; //!< Windows advertised, scaled
};

void
TcpLinuxRcvWindowTest::ConfigureEnvironment()
{
    TcpGeneralTest::ConfigureEnvironment();
    SetAppPktSize(SEGMENT);
    SetAppPktCount(30);
    SetAppPktInterval(MicroSeconds(10));
    SetPropagationDelay(MilliSeconds(50));
}

void
TcpLinuxRcvWindowTest::ConfigureProperties()
{
    TcpGeneralTest::ConfigureProperties();
    SetInitialCwnd(SENDER, 10);
    SetSegmentSize(SENDER, SEGMENT + TcpSocketBase::GetSegmentOptionsSize());
    SetSegmentSize(RECEIVER, SEGMENT + TcpSocketBase::GetSegmentOptionsSize());
    SetRcvBufSize(RECEIVER, 65536);
}

Ptr<TcpSocketMsgBase>
TcpLinuxRcvWindowTest::CreateReceiverSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateReceiverSocket(node);
    socket->SetAttribute("LinuxRcvWindow", BooleanValue(m_linux));
    return socket;
}

Ptr<ErrorModel>
TcpLinuxRcvWindowTest::CreateReceiverErrorModel()
{
    Ptr<TcpSeqErrorModel> errorModel = CreateObject<TcpSeqErrorModel>();
    errorModel->AddSeqToKill(SequenceNumber32(LOST_SEQ));
    return errorModel;
}

void
TcpLinuxRcvWindowTest::Rx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who)
{
    if (who != SENDER)
    {
        return;
    }
    if (h.GetFlags() & TcpHeader::SYN)
    {
        m_synAckWindow = h.GetWindowSize();
        if (h.HasOption(TcpOption::WINSCALE))
        {
            m_wscale =
                DynamicCast<const TcpOptionWinScale>(h.GetOption(TcpOption::WINSCALE))->GetScale();
        }
        return;
    }
    m_acks.push_back(ToRelativeSeq(SENDER, h.GetAckNumber()).GetValue());
    m_windows.push_back(static_cast<uint32_t>(h.GetWindowSize()) << m_wscale);
    NS_LOG_DEBUG("ACK " << m_acks.back() << " window " << m_windows.back());
}

void
TcpLinuxRcvWindowTest::FinalChecks()
{
    // The ACK of the handshake carries no data
    NS_TEST_ASSERT_MSG_GT(m_acks.size(), 10, "Too few ACKs");
    std::size_t first = 0;
    while (first < m_acks.size() && m_acks[first] <= 1)
    {
        ++first;
    }
    NS_TEST_ASSERT_MSG_LT(first, m_acks.size(), "No ACK of data");
    std::size_t holeFill = first;
    while (holeFill < m_acks.size() && m_acks[holeFill] <= LOST_SEQ)
    {
        ++holeFill;
    }
    NS_TEST_ASSERT_MSG_LT(holeFill, m_acks.size(), "The lost segment was not recovered");
    uint32_t edgeBefore = m_acks[holeFill - 1] + m_windows[holeFill - 1];
    uint32_t edgeAfter = m_acks[holeFill] + m_windows[holeFill];

    if (!m_linux)
    {
        NS_TEST_ASSERT_MSG_EQ(m_windows[first], 65536, "Unexpected window without the attribute");
        NS_TEST_ASSERT_MSG_EQ(m_windows[holeFill], 65536, "Unexpected window after the loss");
        return;
    }
    NS_TEST_ASSERT_MSG_EQ(m_synAckWindow, 65160, "Unexpected window in the SYN+ACK");
    NS_TEST_ASSERT_MSG_EQ(+m_wscale, 2, "Unexpected window scale");
    NS_TEST_ASSERT_MSG_EQ(m_windows[first], 66652, "Unexpected window of the first ACK");
    NS_TEST_ASSERT_MSG_LT(m_windows[holeFill],
                          m_windows[holeFill - 1],
                          "The data delivered at once did not reduce the window");
    NS_TEST_ASSERT_MSG_GT_OR_EQ(edgeAfter, edgeBefore, "The window shrank");
}

/**
 * @ingroup internet-test
 *
 * @brief Check the quick ACKs left once the window stops growing.
 *
 * With QuickAck, LinuxRcvWindow and a 1 MiB buffer, the receiver gets pairs
 * of segments (the second pushed) and starts with 16 quick ACKs
 * (tcp_incr_quickack()). As tcp_grow_window(), each segment that grows the
 * window ORs the count with one, so its ACK does not consume the even count.
 * Expected: once the window stops growing all 16 quick ACKs remain, giving 16
 * per-segment ACKs plus one more because the last quick ACK falls on the
 * first segment of a pair, whose pushed second segment is ACKed at once.
 */
class TcpLinuxRcvWindowQuickAckTest : public TcpGeneralTest
{
  public:
    /**
     * @brief Constructor.
     * @param name Test description.
     */
    TcpLinuxRcvWindowQuickAckTest(const std::string& name)
        : TcpGeneralTest(name)
    {
    }

  protected:
    void ConfigureProperties() override;
    void ConfigureEnvironment() override;
    Ptr<TcpSocketMsgBase> CreateReceiverSocket(Ptr<Node> node) override;
    void Rx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override;
    void FinalChecks() override;

  private:
    static constexpr uint32_t SEGMENT = 1448; //!< Data bytes of a segment

    uint8_t m_wscale{0};             //!< Window scale of the receiver
    std::vector<uint32_t> m_acks;    //!< Acknowledgment numbers, relative
    std::vector<uint32_t> m_windows; //!< Windows advertised, scaled
};

void
TcpLinuxRcvWindowQuickAckTest::ConfigureEnvironment()
{
    TcpGeneralTest::ConfigureEnvironment();
    SetAppPktSize(2 * SEGMENT);
    SetAppPktCount(40);
    SetAppPktInterval(MilliSeconds(2));
    SetPropagationDelay(MilliSeconds(5));
}

void
TcpLinuxRcvWindowQuickAckTest::ConfigureProperties()
{
    TcpGeneralTest::ConfigureProperties();
    SetInitialCwnd(SENDER, 10);
    SetSegmentSize(SENDER, SEGMENT + TcpSocketBase::GetSegmentOptionsSize());
    SetSegmentSize(RECEIVER, SEGMENT + TcpSocketBase::GetSegmentOptionsSize());
    SetRcvBufSize(RECEIVER, 1 << 20);
}

Ptr<TcpSocketMsgBase>
TcpLinuxRcvWindowQuickAckTest::CreateReceiverSocket(Ptr<Node> node)
{
    Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateReceiverSocket(node);
    socket->SetAttribute("LinuxRcvWindow", BooleanValue(true));
    socket->SetAttribute("QuickAck", BooleanValue(true));
    return socket;
}

void
TcpLinuxRcvWindowQuickAckTest::Rx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who)
{
    if (who != SENDER)
    {
        return;
    }
    if (h.GetFlags() & TcpHeader::SYN)
    {
        if (h.HasOption(TcpOption::WINSCALE))
        {
            m_wscale =
                DynamicCast<const TcpOptionWinScale>(h.GetOption(TcpOption::WINSCALE))->GetScale();
        }
        return;
    }
    uint32_t ack = ToRelativeSeq(SENDER, h.GetAckNumber()).GetValue();
    if (ack <= 1 || (!m_acks.empty() && ack == m_acks.back()))
    {
        return;
    }
    m_acks.push_back(ack);
    m_windows.push_back(static_cast<uint32_t>(h.GetWindowSize()) << m_wscale);
    NS_LOG_DEBUG("ACK " << ack << " window " << m_windows.back());
}

void
TcpLinuxRcvWindowQuickAckTest::FinalChecks()
{
    NS_TEST_ASSERT_MSG_GT(m_acks.size(), 30, "Too few ACKs");
    // The ACKs that the window grew for: the window advertised rose
    std::size_t growing = 1;
    while (growing < m_acks.size() && m_windows[growing] > m_windows[growing - 1])
    {
        ++growing;
    }
    std::size_t single = 0;
    uint32_t previous = 1;
    while (single < m_acks.size() && m_acks[single] - previous == SEGMENT)
    {
        previous = m_acks[single];
        ++single;
    }
    NS_TEST_ASSERT_MSG_LT(growing, single, "The window grew on every segment acknowledged");
    NS_TEST_ASSERT_MSG_EQ(single - growing,
                          16 + 1,
                          "The quick ACKs left once the window stopped growing differ");
    NS_TEST_ASSERT_MSG_LT(single, m_acks.size(), "The ACKs were never delayed");
    NS_TEST_ASSERT_MSG_EQ(m_acks[single] - previous, 2 * SEGMENT, "Unexpected delayed ACK");
}

/**
 * @ingroup internet-test
 *
 * @brief Check the window of the duplicate ACKs with the LinuxRcvWindow attribute.
 *
 * A segment is lost while the receive window still grows. As
 * tcp_data_queue_ofo(), out-of-order segments grow the window
 * (tcp_grow_window()) only with SACK. Without SACK the duplicate ACKs must
 * keep the same window, or the sender would not count them as duplicates
 * (RFC 5681, Section 2) and would not fast retransmit; with SACK the window
 * keeps growing.
 */
class TcpLinuxRcvWindowDupAckTest : public TcpGeneralTest
{
  public:
    /**
     * @brief Constructor.
     * @param name Test description.
     * @param sack Value of the Sack attribute of the receiver.
     */
    TcpLinuxRcvWindowDupAckTest(const std::string& name, bool sack)
        : TcpGeneralTest(name),
          m_sack(sack)
    {
    }

  protected:
    void ConfigureEnvironment() override
    {
        TcpGeneralTest::ConfigureEnvironment();
        SetAppPktSize(SEGMENT);
        SetAppPktCount(40);
        SetAppPktInterval(MicroSeconds(10));
        SetPropagationDelay(MilliSeconds(50));
    }

    void ConfigureProperties() override
    {
        TcpGeneralTest::ConfigureProperties();
        SetInitialCwnd(SENDER, 40);
        SetSegmentSize(SENDER, SEGMENT + TcpSocketBase::GetSegmentOptionsSize());
        SetSegmentSize(RECEIVER, SEGMENT + TcpSocketBase::GetSegmentOptionsSize());
        SetRcvBufSize(RECEIVER, 1048576);
    }

    Ptr<TcpSocketMsgBase> CreateReceiverSocket(Ptr<Node> node) override
    {
        Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateReceiverSocket(node);
        socket->SetAttribute("LinuxRcvWindow", BooleanValue(true));
        socket->SetAttribute("Sack", BooleanValue(m_sack));
        return socket;
    }

    Ptr<ErrorModel> CreateReceiverErrorModel() override
    {
        Ptr<TcpSeqErrorModel> errorModel = CreateObject<TcpSeqErrorModel>();
        errorModel->AddSeqToKill(SequenceNumber32(LOST_SEQ));
        return errorModel;
    }

    void Rx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override
    {
        if (who != SENDER || (h.GetFlags() & TcpHeader::SYN))
        {
            return;
        }
        if (ToRelativeSeq(SENDER, h.GetAckNumber()).GetValue() == LOST_SEQ)
        {
            m_dupAckWindows.push_back(h.GetWindowSize());
            NS_LOG_DEBUG("ACK of the hole, window field " << m_dupAckWindows.back());
        }
    }

    void FinalChecks() override
    {
        NS_TEST_ASSERT_MSG_GT(m_dupAckWindows.size(), 10, "Too few duplicate ACKs");
        auto [lowest, highest] =
            std::minmax_element(m_dupAckWindows.begin() + 1, m_dupAckWindows.end());
        if (m_sack)
        {
            NS_TEST_ASSERT_MSG_GT(*highest, *lowest, "The window did not grow with SACK");
        }
        else
        {
            NS_TEST_ASSERT_MSG_EQ(*highest, *lowest, "The duplicate ACKs changed the window");
        }
    }

  private:
    static constexpr uint32_t SEGMENT = 1448;             //!< Data bytes of a segment
    static constexpr uint32_t LOST_SEQ = 1 + 5 * SEGMENT; //!< Lost segment

    bool m_sack;                           //!< Value of the Sack attribute of the receiver
    std::vector<uint16_t> m_dupAckWindows; //!< Window fields of the ACKs of the hole
};

/**
 * @ingroup internet-test
 *
 * @brief Check the windows given by the Linux skb memory accounting.
 *
 * RcvBufSize 131072 (262144-byte buffer, window scale 3); 1448-byte segments
 * 1 ms apart, each quick-ACKed before the application reads. Expected values
 * match Linux 6.12 under DCE:
 * - Bound: half the buffer minus the listener's route advmss of 1460, 129612
 *   bytes (tcp_init_buffer_space()); the first segment adds two segments:
 *   132508, rounded up to the scale unit: 132512.
 * - With a 2784-byte skb held (1248-byte head + 1536-byte virtio-net receive
 *   buffer) the window reaches 134752.
 * - Every 21 buffers one takes the end of a 32 KiB page and is 2048 bytes;
 *   with RcvBufferOffset 1 (one frame before the SYN, like the router
 *   solicitation DCE delivers) this is the 18th segment, whose ACK
 *   advertises 134480.
 */
class TcpLinuxRcvWindowSkbTest : public TcpGeneralTest
{
  public:
    /**
     * @brief Constructor.
     * @param name Test description.
     */
    TcpLinuxRcvWindowSkbTest(const std::string& name)
        : TcpGeneralTest(name)
    {
    }

  protected:
    void ConfigureEnvironment() override
    {
        TcpGeneralTest::ConfigureEnvironment();
        SetAppPktSize(SEGMENT);
        SetAppPktCount(18);
        SetAppPktInterval(MilliSeconds(1));
        SetPropagationDelay(MilliSeconds(5));
    }

    void ConfigureProperties() override
    {
        TcpGeneralTest::ConfigureProperties();
        SetInitialCwnd(SENDER, 20);
        SetSegmentSize(SENDER, SEGMENT + TcpSocketBase::GetSegmentOptionsSize());
        SetSegmentSize(RECEIVER, SEGMENT + TcpSocketBase::GetSegmentOptionsSize());
        SetRcvBufSize(RECEIVER, 131072);
    }

    Ptr<TcpSocketMsgBase> CreateReceiverSocket(Ptr<Node> node) override
    {
        Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateReceiverSocket(node);
        socket->SetAttribute("LinuxRcvWindow", BooleanValue(true));
        socket->SetAttribute("QuickAck", BooleanValue(true));
        socket->SetAttribute("RcvBufferOffset", UintegerValue(1));
        return socket;
    }

    void Rx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override
    {
        if (who != SENDER)
        {
            return;
        }
        if (h.GetFlags() & TcpHeader::SYN)
        {
            if (h.HasOption(TcpOption::WINSCALE))
            {
                m_wscale = DynamicCast<const TcpOptionWinScale>(h.GetOption(TcpOption::WINSCALE))
                               ->GetScale();
            }
            return;
        }
        uint32_t ack = ToRelativeSeq(SENDER, h.GetAckNumber()).GetValue();
        if (ack > 1 && ack <= 1 + 18 * SEGMENT)
        {
            m_windows[ack] = static_cast<uint32_t>(h.GetWindowSize()) << m_wscale;
        }
    }

    void FinalChecks() override
    {
        NS_TEST_ASSERT_MSG_EQ(+m_wscale, 3, "Unexpected window scale");
        NS_TEST_ASSERT_MSG_EQ(m_windows.size(), 18, "A segment was not acknowledged at once");
        for (uint32_t i = 1; i <= 18; ++i)
        {
            uint32_t expected = i == 1 ? 132512 : (i == 18 ? 134480 : 134752);
            NS_TEST_ASSERT_MSG_EQ(m_windows[1 + i * SEGMENT],
                                  expected,
                                  "Unexpected window of the ACK of segment " << i);
        }
    }

  private:
    static constexpr uint32_t SEGMENT = 1448; //!< Data bytes of a segment

    uint8_t m_wscale{0};                    //!< Window scale of the receiver
    std::map<uint32_t, uint32_t> m_windows; //!< Window of each ACK number, relative
};

/**
 * @ingroup internet-test
 *
 * @brief Check the out-of-order data dropped under TCP memory pressure.
 *
 * Segment 100 of a 1500-segment burst is lost; the receiver (SACK, 4 MiB
 * buffer) holds the rest out of order, about 17 per 27360-byte skb. Reserved
 * pages are charged to the protocol per MiB (Linux per-CPU reserve), so with
 * LinuxTcpMem "100 200 300" (4096-byte pages) the first MiB causes memory
 * pressure, where a socket may not exceed the hard limit shared by the
 * node's two sockets (listener and connection). As tcp_try_rmem_schedule(),
 * the out-of-order queue is pruned down to the incoming segment, which is
 * dropped. Expected: SACK blocks stop growing at about 1 MiB; without a
 * limit they cover all data the window admits.
 */
class TcpLinuxRcvWindowMemTest : public TcpGeneralTest
{
  public:
    /**
     * @brief Constructor.
     * @param name Test description.
     * @param tcpMem Value of the LinuxTcpMem attribute of the receiver.
     */
    TcpLinuxRcvWindowMemTest(const std::string& name, const std::string& tcpMem)
        : TcpGeneralTest(name),
          m_tcpMem(tcpMem)
    {
    }

  protected:
    void ConfigureEnvironment() override
    {
        TcpGeneralTest::ConfigureEnvironment();
        SetAppPktSize(SEGMENT);
        SetAppPktCount(1500);
        SetAppPktInterval(MicroSeconds(10));
        SetPropagationDelay(MilliSeconds(50));
    }

    void ConfigureProperties() override
    {
        TcpGeneralTest::ConfigureProperties();
        SetInitialCwnd(SENDER, 1500);
        SetSegmentSize(SENDER, SEGMENT + TcpSocketBase::GetSegmentOptionsSize());
        SetSegmentSize(RECEIVER, SEGMENT + TcpSocketBase::GetSegmentOptionsSize());
        SetRcvBufSize(RECEIVER, 2097152);
    }

    Ptr<TcpSocketMsgBase> CreateSenderSocket(Ptr<Node> node) override
    {
        Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateSenderSocket(node);
        socket->SetAttribute("SndBufSize", UintegerValue(4194304));
        return socket;
    }

    Ptr<TcpSocketMsgBase> CreateReceiverSocket(Ptr<Node> node) override
    {
        Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateReceiverSocket(node);
        socket->SetAttribute("LinuxRcvWindow", BooleanValue(true));
        socket->SetAttribute("LinuxTcpMem", StringValue(m_tcpMem));
        return socket;
    }

    Ptr<ErrorModel> CreateReceiverErrorModel() override
    {
        Ptr<TcpSeqErrorModel> errorModel = CreateObject<TcpSeqErrorModel>();
        errorModel->AddSeqToKill(SequenceNumber32(LOST_SEQ));
        return errorModel;
    }

    void Rx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override
    {
        if (who != SENDER || !h.HasOption(TcpOption::SACK))
        {
            return;
        }
        auto sack = DynamicCast<const TcpOptionSack>(h.GetOption(TcpOption::SACK));
        for (const auto& block : sack->GetSackList())
        {
            m_maxSackEnd = std::max(m_maxSackEnd, ToRelativeSeq(SENDER, block.second).GetValue());
        }
    }

    void FinalChecks() override
    {
        const uint32_t bound = 1 + 800 * SEGMENT;
        if (m_tcpMem.empty())
        {
            NS_TEST_ASSERT_MSG_GT(m_maxSackEnd, bound, "Out-of-order data dropped");
        }
        else
        {
            NS_TEST_ASSERT_MSG_GT(m_maxSackEnd, LOST_SEQ + SEGMENT, "No out-of-order data held");
            NS_TEST_ASSERT_MSG_LT(m_maxSackEnd, bound, "Out-of-order data not dropped");
        }
    }

  private:
    static constexpr uint32_t SEGMENT = 1448;              //!< Data bytes of a segment
    static constexpr uint32_t LOST_SEQ = 1 + 99 * SEGMENT; //!< Lost segment

    std::string m_tcpMem;     //!< LinuxTcpMem attribute of the receiver
    uint32_t m_maxSackEnd{0}; //!< Highest right edge of the SACK blocks
};

/**
 * @ingroup internet-test
 *
 * @brief Check the layout of the SACK option (AlignSackOption attribute).
 *
 * A segment is lost and the receiver SACKs the following ones, with
 * timestamps. Expected options length for n SACK blocks: 12 + 8 n bytes by
 * default; 16 + 8 n with AlignSackOption, as Linux puts two NOPs before the
 * SACK option.
 */
class TcpAlignSackOptionTest : public TcpGeneralTest
{
  public:
    /**
     * @brief Constructor.
     * @param name Test description.
     * @param align Value of the AlignSackOption attribute of the receiver.
     */
    TcpAlignSackOptionTest(const std::string& name, bool align)
        : TcpGeneralTest(name),
          m_align(align)
    {
    }

  protected:
    void ConfigureEnvironment() override
    {
        TcpGeneralTest::ConfigureEnvironment();
        SetAppPktSize(SEGMENT);
        SetAppPktCount(20);
        SetAppPktInterval(MicroSeconds(10));
        SetPropagationDelay(MilliSeconds(50));
    }

    void ConfigureProperties() override
    {
        TcpGeneralTest::ConfigureProperties();
        SetInitialCwnd(SENDER, 20);
        SetSegmentSize(SENDER, SEGMENT + TcpSocketBase::GetSegmentOptionsSize());
        SetSegmentSize(RECEIVER, SEGMENT + TcpSocketBase::GetSegmentOptionsSize());
    }

    Ptr<TcpSocketMsgBase> CreateReceiverSocket(Ptr<Node> node) override
    {
        Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateReceiverSocket(node);
        socket->SetAttribute("AlignSackOption", BooleanValue(m_align));
        return socket;
    }

    Ptr<ErrorModel> CreateReceiverErrorModel() override
    {
        Ptr<TcpSeqErrorModel> errorModel = CreateObject<TcpSeqErrorModel>();
        errorModel->AddSeqToKill(SequenceNumber32(1 + 5 * SEGMENT));
        return errorModel;
    }

    void Rx(const Ptr<const Packet> p, const TcpHeader& h, SocketWho who) override
    {
        if (who != SENDER || !h.HasOption(TcpOption::SACK))
        {
            return;
        }
        auto sack = DynamicCast<const TcpOptionSack>(h.GetOption(TcpOption::SACK));
        uint32_t blocks = sack->GetSackList().size();
        uint32_t expected = (m_align ? 16 : 12) + 8 * blocks;
        NS_TEST_ASSERT_MSG_EQ(h.GetOptionLength(), expected, "Unexpected length of the options");
        ++m_sackAcks;
    }

    void FinalChecks() override
    {
        NS_TEST_ASSERT_MSG_GT(m_sackAcks, 0, "No ACK with SACK");
    }

  private:
    static constexpr uint32_t SEGMENT = 1448; //!< Data bytes of a segment

    bool m_align;           //!< AlignSackOption attribute of the receiver
    uint32_t m_sackAcks{0}; //!< ACKs with SACK
};

/**
 * @ingroup internet-test
 *
 * @brief TestSuite for the LinuxRcvWindow attribute
 */
class TcpLinuxRcvWindowTestSuite : public TestSuite
{
  public:
    TcpLinuxRcvWindowTestSuite()
        : TestSuite("tcp-linux-rcv-window", Type::UNIT)
    {
        AddTestCase(new TcpLinuxRcvWindowTest("Window derived from the buffer as Linux", true),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpLinuxRcvWindowTest("Window of RcvBufSize by default", false),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpLinuxRcvWindowQuickAckTest("Quick ACKs kept while the window grows"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpLinuxRcvWindowDupAckTest("Duplicate ACKs keep the window", false),
                    TestCase::Duration::QUICK);
        AddTestCase(
            new TcpLinuxRcvWindowDupAckTest("Out-of-order data grows the window with SACK", true),
            TestCase::Duration::QUICK);
        AddTestCase(new TcpLinuxRcvWindowSkbTest("Windows of the skbs of Linux"),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpAlignSackOptionTest("SACK option not aligned", false),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpAlignSackOptionTest("SACK option aligned as Linux", true),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpLinuxRcvWindowMemTest("Out-of-order data kept", ""),
                    TestCase::Duration::QUICK);
        AddTestCase(new TcpLinuxRcvWindowMemTest("Out-of-order data dropped under memory pressure",
                                                 "100 200 300"),
                    TestCase::Duration::QUICK);
    }
};

static TcpLinuxRcvWindowTestSuite g_tcpLinuxRcvWindowTestSuite; //!< static var for test init
