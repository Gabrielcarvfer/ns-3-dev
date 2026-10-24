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
 * The receiver has RcvBufSize 65536, hence a buffer of 131072 bytes as Linux
 * doubles the size given to SO_RCVBUF, and receives segments of 1448 bytes.
 * As Linux (tcp_select_initial_window()), the SYN+ACK offers half the buffer
 * rounded down to whole segments, 65160 bytes, with a window scale of 2
 * (from net.core.rmem_max, 212992). Once established, the window is bound to
 * half the buffer minus one segment, 64088 bytes (tcp_init_buffer_space()).
 * The first segment, of 1448 bytes in an skb of 2784 bytes, sets the data to
 * memory ratio to 133/256 and the bound of the window to 131072 * 133 / 256 =
 * 68096 bytes; the window then grows to what the memory left with the
 * segment held allows, (131072 - 2784) * 133 / 256 = 66649 bytes, rounded up
 * to the scale unit: 66652 bytes (tcp_grow_window(), __tcp_select_window()).
 *
 * A segment is then lost. The ACK of its retransmission, which delivers the
 * segments held out of order, advertises what the memory still taken by them
 * allows, since Linux acknowledges before the application reads: less than
 * the window of before the loss, but not less than what the right edge
 * already advertised allows (tcp_select_window() never shrinks the window).
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
 * With the QuickAck and LinuxRcvWindow attributes and a buffer of 1 MiB, the
 * receiver of pairs of segments, the second one pushed, starts with 16 quick
 * ACKs (tcp_incr_quickack(), the window over two segments being more than
 * 16), and its window grows on each segment for a while. As Linux
 * (tcp_grow_window()), each segment that grows the window or-es the count
 * with one, so that the ACK it triggers does not consume the even count
 * left: once the window stops growing, the 16 quick ACKs remain, and the
 * segments are acknowledged one by one 16 more times, and once more as the
 * last quick ACK falls on the first segment of a pair, whose pushed second
 * segment is acknowledged at once, before the ACKs cover the pairs.
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
 * A segment of a burst is lost while the window of the receiver still grows.
 * As tcp_data_queue_ofo() of Linux, the out-of-order segments that follow
 * grow the window (tcp_grow_window()) only with SACK: without it, the
 * duplicate ACKs must carry the same window, or the sender would not count
 * them as duplicate ACKs (RFC 5681, Section 2), and would not retransmit
 * fast. With SACK, the window keeps growing.
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
 * @brief Check the windows that the memory of the skbs of Linux gives.
 *
 * The receiver has RcvBufSize 131072 (a buffer of 262144 bytes, a window
 * scale of 3), and receives segments of 1448 bytes, 1 ms apart, each
 * acknowledged at once (quick ACKs), before the application reads it. Its
 * window is bound to half the buffer less the 1460-byte advmss of the route
 * that its listener used (tcp_init_buffer_space()), 129612 bytes, to which
 * the first segment adds two segments: 132508 bytes, rounded up to the unit
 * of the window scale, 132512 (as Linux 6.12 under DCE). With an skb of
 * 2784 bytes held (an skb head of 1248 bytes and a receive buffer of 1536
 * of virtio-net), the window then reaches 134752 bytes, but the receive
 * buffer of the segment that takes the end of a 32 KiB page, 21 buffers
 * apart, is 2048 bytes: with RcvBufferOffset 1 (a frame before the SYN, as
 * the router solicitation that DCE delivers), that of the 18th segment,
 * whose ACK advertises 134480 bytes.
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
 * The 100th segment of a burst of 1500 is lost, and the receiver, with SACK
 * and a buffer of 4 MiB, holds the others out of order, 17 to an skb of
 * 27360 bytes or so. The pages that the socket reserves reach the memory of
 * the protocol by the MiB (the per-CPU reserve of Linux): with LinuxTcpMem
 * "100 200 300" (pages of 4096 bytes), the first MiB puts the protocol in
 * memory pressure, in which a socket may not use more than the hard limit
 * shared by the two sockets of the node (the listener and the connection).
 * As tcp_try_rmem_schedule() of Linux, the out-of-order queue is pruned
 * down to the incoming segment, which, the last one, is dropped: the SACK
 * blocks stop growing at some 1 MiB of data. Without a limit, they cover
 * all the data that the window lets in.
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
 * A segment is lost, and the receiver reports the segments that follow it
 * with SACK, with the timestamps. By default, the 10 bytes of the timestamps
 * option and the SACK option make 12 + 8 n bytes of options, n being the
 * number of SACK blocks. With AlignSackOption, as Linux, two NOP options
 * precede the SACK option, which makes 16 + 8 n bytes.
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
