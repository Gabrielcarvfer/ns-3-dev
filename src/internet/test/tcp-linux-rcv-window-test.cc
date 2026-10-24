/*
 * Copyright (c) 2026 Centre Tecnològic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
 */

#include "tcp-error-model.h"
#include "tcp-general-test.h"

#include "ns3/log.h"
#include "ns3/node.h"
#include "ns3/tcp-header.h"
#include "ns3/tcp-option-winscale.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/test.h"

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
    }
};

static TcpLinuxRcvWindowTestSuite g_tcpLinuxRcvWindowTestSuite; //!< static var for test init
