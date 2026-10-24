/*
 * Copyright (c) 2007 Georgia Tech Research Corporation
 * Copyright (c) 2010 Adrian Sai-wah Tam
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Adrian Sai-wah Tam <adrian.sw.tam@gmail.com>
 */

#define NS_LOG_APPEND_CONTEXT                                                                      \
    if (m_node)                                                                                    \
    {                                                                                              \
        ns3LogStream << " [node " << m_node->GetId() << "] ";                                      \
    }

#include "tcp-socket-base.h"

#include "ipv4-end-point.h"
#include "ipv4-route.h"
#include "ipv4-routing-protocol.h"
#include "ipv4-source-route-tag.h"
#include "ipv4.h"
#include "ipv6-end-point.h"
#include "ipv6-l3-protocol.h"
#include "ipv6-route.h"
#include "ipv6-routing-protocol.h"
#include "rtt-estimator.h"
#include "tcp-congestion-ops.h"
#include "tcp-header.h"
#include "tcp-l4-protocol.h"
#include "tcp-option-rfc793.h"
#include "tcp-option-sack-permitted.h"
#include "tcp-option-sack.h"
#include "tcp-option-ts.h"
#include "tcp-option-winscale.h"
#include "tcp-rate-ops.h"
#include "tcp-recovery-ops.h"
#include "tcp-rx-buffer.h"
#include "tcp-tx-buffer.h"

#include "ns3/abort.h"
#include "ns3/data-rate.h"
#include "ns3/double.h"
#include "ns3/hash.h"
#include "ns3/inet-socket-address.h"
#include "ns3/inet6-socket-address.h"
#include "ns3/log.h"
#include "ns3/node.h"
#include "ns3/object.h"
#include "ns3/packet.h"
#include "ns3/pointer.h"
#include "ns3/simulation-singleton.h"
#include "ns3/simulator.h"
#include "ns3/string.h"
#include "ns3/trace-source-accessor.h"
#include "ns3/uinteger.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace
{

/**
 * @brief map TcpPacketType and EcnMode to boolean value to check whether ECN-marking is allowed or
 * not
 */
const std::map<std::pair<ns3::TcpSocketBase::TcpPacketType_t, ns3::TcpSocketState::EcnMode_t>, bool>
    ECN_RESTRICTION_MAP{
        {{ns3::TcpSocketBase::SYN, ns3::TcpSocketState::ClassicEcn}, false},
        {{ns3::TcpSocketBase::SYN_ACK, ns3::TcpSocketState::ClassicEcn}, false},
        {{ns3::TcpSocketBase::PURE_ACK, ns3::TcpSocketState::ClassicEcn}, false},
        {{ns3::TcpSocketBase::WINDOW_PROBE, ns3::TcpSocketState::ClassicEcn}, false},
        {{ns3::TcpSocketBase::FIN, ns3::TcpSocketState::ClassicEcn}, false},
        {{ns3::TcpSocketBase::RST, ns3::TcpSocketState::ClassicEcn}, false},
        {{ns3::TcpSocketBase::RE_XMT, ns3::TcpSocketState::ClassicEcn}, false},
        {{ns3::TcpSocketBase::DATA, ns3::TcpSocketState::ClassicEcn}, true},

        {{ns3::TcpSocketBase::SYN, ns3::TcpSocketState::DctcpEcn}, true},
        {{ns3::TcpSocketBase::SYN_ACK, ns3::TcpSocketState::DctcpEcn}, true},
        {{ns3::TcpSocketBase::PURE_ACK, ns3::TcpSocketState::DctcpEcn}, true},
        {{ns3::TcpSocketBase::WINDOW_PROBE, ns3::TcpSocketState::DctcpEcn}, true},
        {{ns3::TcpSocketBase::FIN, ns3::TcpSocketState::DctcpEcn}, true},
        {{ns3::TcpSocketBase::RST, ns3::TcpSocketState::DctcpEcn}, true},
        {{ns3::TcpSocketBase::RE_XMT, ns3::TcpSocketState::DctcpEcn}, true},
        {{ns3::TcpSocketBase::DATA, ns3::TcpSocketState::DctcpEcn}, true},
};

/// Memory of the sk_buff structure of Linux 6.12 (SKB_DATA_ALIGN(sizeof(struct sk_buff)))
constexpr uint32_t LINUX_SKB_STRUCT_SIZE = 224;

/// Page size of Linux, the unit of its memory accounting
constexpr int64_t LINUX_PAGE_SIZE = 4096;

/// Pages a CPU reserves before the memory of the protocol counts them, as SK_MEMORY_PCPU_RESERVE
constexpr int64_t LINUX_MEM_PCPU_RESERVE = (1 << 20) / LINUX_PAGE_SIZE;

/// Memory a socket may always use under memory pressure, as tcp_rmem[0] of Linux
constexpr int64_t LINUX_TCP_RMEM0 = 4096;

/// Bound on the number of quick ACKs, as TCP_MAX_QUICKACKS of Linux
constexpr uint32_t MAX_QUICK_ACKS = 16;

/// Bound on the number of quick ACKs upon a CE mark, as in __tcp_ecn_check_ce() of Linux
constexpr uint32_t CE_QUICK_ACKS = 2;

/// ACKs of out-of-order data sent at once at a given RCV.NXT, as TCP_FASTRETRANS_THRESH of Linux
constexpr uint32_t SACK_COMPRESS_DUPACKS = 3;

/// Bound on the number of ACKs merged by SACK compression, as tcp_comp_sack_nr of Linux
constexpr uint32_t SACK_COMPRESS_MAX = 44;

/// Index of the SACK block whose change sends the ACK held back by SACK
/// compression (TCP_SACK_BLOCKS_EXPECTED in Linux)
constexpr uint32_t SACK_COMPRESS_FLUSH_BLOCK = 2;

/// Minimum of the adaptive delayed ACK timeout in ms, as TCP_ATO_MIN and TCP_DELACK_MIN of Linux
constexpr int64_t ATO_MIN_MS = 40;

/// Bound on the adaptive delayed ACK timeout above the minimum in ms, as HZ / 2 in
/// tcp_send_delayed_ack() of Linux
constexpr int64_t ATO_MAX_MS = 500;
} // namespace

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("TcpSocketBase");

NS_OBJECT_ENSURE_REGISTERED(TcpSocketBase);

TypeId
TcpSocketBase::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::TcpSocketBase")
            .SetParent<TcpSocket>()
            .SetGroupName("Internet")
            .AddConstructor<TcpSocketBase>()
            //    .AddAttribute ("TcpState", "State in TCP state machine",
            //                   TypeId::ATTR_GET,
            //                   EnumValue (CLOSED),
            //                   MakeEnumAccessor (&TcpSocketBase::m_state),
            //                   MakeEnumChecker (CLOSED, "Closed"))
            .AddAttribute("KeepAliveInterval",
                          "Time between the keep-alives which are not answered, at least one "
                          "millisecond. The keep-alives are turned on, and their idle time set, "
                          "by the KeepAlive and KeepAliveTime attributes of ns3::TcpSocket.",
                          TimeValue(Seconds(75)),
                          MakeTimeAccessor(&TcpSocketBase::m_keepAliveInterval),
                          MakeTimeChecker(MilliSeconds(1)))
            .AddAttribute("KeepAliveRetries",
                          "Number of unanswered keep-alives before the connection is dropped. "
                          "At least two, since a single unanswered keep-alive must not be "
                          "taken for a dead connection (RFC 9293, Section 3.8.4, MUST-29). The "
                          "keep-alives are turned on, and their idle time set, by the KeepAlive "
                          "and KeepAliveTime attributes of ns3::TcpSocket.",
                          UintegerValue(9),
                          MakeUintegerAccessor(&TcpSocketBase::m_keepAliveRetries),
                          MakeUintegerChecker<uint32_t>(2))
            .AddAttribute("MaxSegLifetime",
                          "Maximum segment lifetime in seconds, use for TIME_WAIT state transition "
                          "to CLOSED state",
                          DoubleValue(120), /* RFC793 says MSL=2 minutes*/
                          MakeDoubleAccessor(&TcpSocketBase::m_msl),
                          MakeDoubleChecker<double>(0))
            .AddAttribute("MaxWindowSize",
                          "Max size of advertised window",
                          UintegerValue(65535),
                          MakeUintegerAccessor(&TcpSocketBase::m_maxWinSize),
                          MakeUintegerChecker<uint16_t>())
            .AddAttribute("IcmpCallback",
                          "Callback invoked whenever an icmp error is received on this socket.",
                          CallbackValue(),
                          MakeCallbackAccessor(&TcpSocketBase::m_icmpCallback),
                          MakeCallbackChecker())
            .AddAttribute("IcmpCallback6",
                          "Callback invoked whenever an icmpv6 error is received on this socket.",
                          CallbackValue(),
                          MakeCallbackAccessor(&TcpSocketBase::m_icmpCallback6),
                          MakeCallbackChecker())
            .AddAttribute("WindowScaling",
                          "Enable or disable Window Scaling option",
                          BooleanValue(true),
                          MakeBooleanAccessor(&TcpSocketBase::m_winScalingEnabled),
                          MakeBooleanChecker())
            .AddAttribute("Sack",
                          "Enable or disable Sack option",
                          BooleanValue(true),
                          MakeBooleanAccessor(&TcpSocketBase::m_sackEnabled),
                          MakeBooleanChecker())
            .AddAttribute("Timestamp",
                          "Enable or disable Timestamp option",
                          BooleanValue(true),
                          MakeBooleanAccessor(&TcpSocketBase::m_timestampEnabled),
                          MakeBooleanChecker())
            .AddAttribute("RandomTimestampOffset",
                          "Add a per-connection offset, derived from a hash of the connection "
                          "identifying parameters and of a per-node secret, to the timestamp "
                          "clock (RFC 7323, Sections 5.4 and 7.1)",
                          BooleanValue(false),
                          MakeBooleanAccessor(&TcpSocketBase::m_randomTsOffset),
                          MakeBooleanChecker())
            .AddAttribute("Fack",
                          "Enable or disable FACK option",
                          BooleanValue(false),
                          MakeBooleanAccessor(&TcpSocketBase::m_fackEnabled),
                          MakeBooleanChecker())
            .AddAttribute(
                "MinRto",
                "Minimum retransmit timeout value",
                TimeValue(Seconds(1)), // RFC 6298 says min RTO=1 sec, but Linux uses 200ms.
                // See http://www.postel.org/pipermail/end2end-interest/2004-November/004402.html
                MakeTimeAccessor(&TcpSocketBase::SetMinRto, &TcpSocketBase::GetMinRto),
                MakeTimeChecker())
            .AddAttribute(
                "ClockGranularity",
                "Clock Granularity used in RTO calculations",
                TimeValue(MilliSeconds(1)), // RFC6298 suggest to use fine clock granularity
                MakeTimeAccessor(&TcpSocketBase::SetClockGranularity,
                                 &TcpSocketBase::GetClockGranularity),
                MakeTimeChecker())
            .AddAttribute("RtoTimerGranularity",
                          "Period of the clock tick driving the retransmission timer, as the "
                          "jiffy of Linux (10 ms with CONFIG_HZ=100), or zero for a timer "
                          "expiring exactly after the RTO. When non-zero, the RTO is rounded "
                          "up to whole ticks, counted from the last tick (a multiple of the "
                          "period since the simulation start), and the timer expires one tick "
                          "later, rounded up to the granularity of the level of the Linux timer "
                          "wheel the timeout falls in (one tick below 63 ticks, eight below "
                          "504, and so on), so that it never expires early. With "
                          "AdaptiveDelAck, the delayed ACK timer counts the same ticks",
                          TimeValue(Seconds(0)),
                          MakeTimeAccessor(&TcpSocketBase::m_rtoTimerGranularity),
                          MakeTimeChecker(Seconds(0)))
            .AddAttribute("TxBuffer",
                          "TCP Tx buffer",
                          PointerValue(),
                          MakePointerAccessor(&TcpSocketBase::GetTxBuffer),
                          MakePointerChecker<TcpTxBuffer>())
            .AddAttribute("RxBuffer",
                          "TCP Rx buffer",
                          PointerValue(),
                          MakePointerAccessor(&TcpSocketBase::GetRxBuffer),
                          MakePointerChecker<TcpRxBuffer>())
            .AddAttribute("CongestionOps",
                          "Pointer to TcpCongestionOps object",
                          PointerValue(),
                          MakePointerAccessor(&TcpSocketBase::m_congestionControl),
                          MakePointerChecker<TcpCongestionOps>())
            .AddAttribute("RecoveryOps",
                          "Pointer to TcpRecoveryOps object",
                          PointerValue(),
                          MakePointerAccessor(&TcpSocketBase::m_recoveryOps),
                          MakePointerChecker<TcpRecoveryOps>())
            .AddAttribute(
                "ReTxThreshold",
                "Threshold for fast retransmit",
                UintegerValue(3),
                MakeUintegerAccessor(&TcpSocketBase::SetRetxThresh, &TcpSocketBase::GetRetxThresh),
                MakeUintegerChecker<uint32_t>())
            .AddAttribute("QuickAck",
                          "Acknowledge every data segment in the quick ACK mode of Linux: at "
                          "the start of the connection, after an idle period longer than the "
                          "RTO, after duplicate data and after a CE mark; and acknowledge at "
                          "once a pushed segment once the application has read it",
                          BooleanValue(false),
                          MakeBooleanAccessor(&TcpSocketBase::m_quickAck),
                          MakeBooleanChecker())
            .AddAttribute("SackCompression",
                          "Delay the ACKs of out-of-order data as Linux (SACK compression): "
                          "after three immediate ACKs at a given RCV.NXT, the ACKs wait for "
                          "SackCompressionDelay or 5% of the RTT, whichever is shorter. This "
                          "departs from the immediate ACK of RFC 5681, Section 4.2",
                          BooleanValue(false),
                          MakeBooleanAccessor(&TcpSocketBase::m_sackCompression),
                          MakeBooleanChecker())
            .AddAttribute("SackCompressionDelay",
                          "Bound on the delay of an ACK held back by SACK compression, as "
                          "tcp_comp_sack_delay_ns of Linux",
                          TimeValue(MilliSeconds(1)),
                          MakeTimeAccessor(&TcpSocketBase::m_sackCompressionDelay),
                          MakeTimeChecker(Time(0)))
            .AddAttribute("SackCompressionSlack",
                          "Slack of the timer of an ACK held back by SACK compression, as "
                          "tcp_comp_sack_slack_ns of Linux: the timer, a range timer in Linux, "
                          "expires at the end of its range when no other timer is due earlier",
                          TimeValue(MicroSeconds(100)),
                          MakeTimeAccessor(&TcpSocketBase::m_sackCompressionSlack),
                          MakeTimeChecker(Time(0)))
            .AddAttribute("AdaptiveDelAck",
                          "Adapt the delayed ACK timeout to the arrival interval of the data "
                          "as Linux (ato), from 40 ms up to DelAckTimeout",
                          BooleanValue(false),
                          MakeBooleanAccessor(&TcpSocketBase::m_adaptiveDelAck),
                          MakeBooleanChecker())
            .AddAttribute("FillReceiveWindow",
                          "Fill the receive window of the peer as Linux: a burst that the "
                          "right edge of the window cuts ends with a segment shorter than "
                          "the MSS. This departs from the sender silly window syndrome "
                          "avoidance of RFC 9293, Section 3.8.6.2.1",
                          BooleanValue(false),
                          MakeBooleanAccessor(&TcpSocketBase::m_fillRcvWindow),
                          MakeBooleanChecker())
            .AddAttribute("NonSackUndo",
                          "Without SACK, as Linux does (tcp_try_undo_recovery()): a fast "
                          "recovery or a timeout recovery whose cumulative ACK reaches exactly "
                          "the recovery point is held until data above the recovery point "
                          "is acknowledged, and then its window reduction is undone, restoring "
                          "the cWnd and ssThresh of before the loss. This departs from "
                          "RFC 5681 and RFC 6582, which keep the reduced window",
                          BooleanValue(false),
                          MakeBooleanAccessor(&TcpSocketBase::m_nonSackUndo),
                          MakeBooleanChecker())
            .AddAttribute("RecoveryReentry",
                          "With SACK, start a new fast recovery on the ACK which ends one when "
                          "data is still deemed lost, as Linux (tcp_time_to_recover() in "
                          "tcp_fastretrans_alert()), rather than on the next duplicate ACK "
                          "(RFC 6675, Section 5)",
                          BooleanValue(false),
                          MakeBooleanAccessor(&TcpSocketBase::m_recoveryReentry),
                          MakeBooleanChecker())
            .AddAttribute("PartialAckInWindow",
                          "Without SACK, as Linux (tcp_newreno_mark_lost()): a partial ACK "
                          "of a fast recovery marks the first unacknowledged segment lost, "
                          "and it is retransmitted only when the congestion window allows. "
                          "This departs from RFC 6582, Section 3.2, step 3, which "
                          "retransmits it at once",
                          BooleanValue(false),
                          MakeBooleanAccessor(&TcpSocketBase::m_partialAckInWindow),
                          MakeBooleanChecker())
            .AddAttribute("KeepRecoverySsThresh",
                          "On a retransmission timeout during a fast recovery (or a CWR "
                          "episode) which has not reached its recovery point, keep the slow "
                          "start threshold that the recovery set instead of computing it "
                          "again, as Linux tcp_enter_loss(). This departs from RFC 5681, "
                          "Section 3.1, equation (4)",
                          BooleanValue(false),
                          MakeBooleanAccessor(&TcpSocketBase::m_keepRecoverySsThresh),
                          MakeBooleanChecker())
            .AddAttribute("SackRestartsRto",
                          "Restart the retransmission timer on an ACK which does not advance "
                          "SND.UNA but SACKs a segment, never retransmitted, sent before the "
                          "latest transmission of the first unacknowledged segment, as Linux "
                          "(FLAG_SET_XMIT_TIMER in tcp_clean_rtx_queue()). This departs from "
                          "RFC 6298, Section 5.3, which restarts the timer only when new data "
                          "is acknowledged",
                          BooleanValue(false),
                          MakeBooleanAccessor(&TcpSocketBase::m_sackRestartsRto),
                          MakeBooleanChecker())
            .AddAttribute("LinuxCaRtt",
                          "Sample the RTT as Linux. The congestion control gets, as the RTT "
                          "sample of an ACK, the time since the transmission of the newest "
                          "segment it acknowledges for the first time and that was never "
                          "retransmitted, or of the newest one it newly SACKs (ca_rtt_us), "
                          "rather than the sample of the timestamps when the acknowledged data "
                          "was retransmitted. The RTT estimator gets the time since the "
                          "transmission of the oldest segment (or SYN) newly acknowledged, "
                          "unless the ACK covers retransmitted data (seq_rtt_us), the "
                          "timestamps being used only then, rather than the timestamps "
                          "whenever they are enabled (RFC 7323, Section 4)",
                          BooleanValue(false),
                          MakeBooleanAccessor(&TcpSocketBase::m_linuxCaRtt),
                          MakeBooleanChecker())
            .AddAttribute("LinuxPktsAcked",
                          "Tell the congestion control of each ACK once, before the ACK "
                          "changes the congestion state (from CA_OPEN to CA_DISORDER or "
                          "CA_RECOVERY, or out of a recovery), with the segments it "
                          "acknowledges cumulatively, none for a duplicate ACK, as Linux "
                          "calls pkts_acked() from tcp_clean_rtx_queue(). Otherwise "
                          "PktsAcked() is called by the state machine, with a segment per "
                          "duplicate ACK and the newly SACKed segments counted",
                          BooleanValue(false),
                          MakeBooleanAccessor(&TcpSocketBase::m_linuxPktsAcked),
                          MakeBooleanChecker())
            .AddAttribute("SackRttSamples",
                          "Take an RTT sample for the RTO estimator from an ACK which does "
                          "not advance SND.UNA but SACKs segments never retransmitted, from "
                          "the oldest of them, as Linux (sack_rtt_us in "
                          "tcp_clean_rtx_queue()). This departs from RFC 7323, Section 4.1, "
                          "whose samples come only from the ACKs advancing SND.UNA",
                          BooleanValue(false),
                          MakeBooleanAccessor(&TcpSocketBase::m_sackRttSamples),
                          MakeBooleanChecker())
            .AddAttribute("RetransmitRestartsRto",
                          "Restart the retransmission timer when the first unacknowledged "
                          "segment is retransmitted, as Linux (tcp_xmit_retransmit_queue()). "
                          "This departs from RFC 6298, Section 5.1, which starts the timer on "
                          "a transmission only when it is not running",
                          BooleanValue(false),
                          MakeBooleanAccessor(&TcpSocketBase::m_retransmitRestartsRto),
                          MakeBooleanChecker())
            .AddAttribute("LinuxRcvWindow",
                          "Derive the advertised window from the memory left in the receive "
                          "buffer as Linux does (tcp_select_window()), with a buffer of twice "
                          "RcvBufSize, as Linux doubles the size given to SO_RCVBUF, and grow "
                          "it from its initial value as segments arrive (rcv_ssthresh)",
                          BooleanValue(false),
                          MakeBooleanAccessor(&TcpSocketBase::m_linuxRcvWindow),
                          MakeBooleanChecker())
            .AddAttribute("RcvHeadTruesize",
                          "With LinuxRcvWindow, the memory of the skb that Linux builds for a "
                          "received frame (sk_buff and head), all of it when the frame is "
                          "copied into the head (see RcvCopyBreak). The default is the one of "
                          "the virtio-net driver of Linux 6.12 (napi_alloc_skb() of 128 bytes)",
                          UintegerValue(1248),
                          MakeUintegerAccessor(&TcpSocketBase::m_rcvHeadTruesize),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("RcvFragTruesize",
                          "With LinuxRcvWindow, the memory of the receive buffer of the "
                          "driver that holds, as a page fragment of the skb, a frame larger "
                          "than RcvCopyBreak (1536 bytes for the mergeable buffers of "
                          "virtio-net with a 1500-byte MTU)",
                          UintegerValue(1536),
                          MakeUintegerAccessor(&TcpSocketBase::m_rcvFragTruesize),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("RcvCopyBreak",
                          "With LinuxRcvWindow, the largest IP datagram that the driver copies "
                          "into the head of the skb (the 638 bytes of tailroom of the head "
                          "of virtio-net, less the 14 bytes of the Ethernet header), its "
                          "receive buffer being freed",
                          UintegerValue(624),
                          MakeUintegerAccessor(&TcpSocketBase::m_rcvCopyBreak),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("RcvFragPageSize",
                          "With LinuxRcvWindow, the size of the pages the receive buffers are "
                          "carved from, one per segment received (skb_page_frag_refill()), "
                          "the last buffer of a page taking the room too small for another "
                          "one, or zero for buffers all of RcvFragTruesize",
                          UintegerValue(32768),
                          MakeUintegerAccessor(&TcpSocketBase::m_rcvFragPageSize),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute(
                "LinuxTcpMem",
                "With LinuxRcvWindow, the limits of the TCP memory of the node, in "
                "pages of 4096 bytes, as net.ipv4.tcp_mem (\"min pressure max\", "
                "or separated by commas), "
                "against which the receive memory of the socket is reserved, the "
                "socket being alone on the node; empty for no limit. Linux derives "
                "them from the memory of the system (738 984 1476 in the Linux "
                "kernel library of DCE)",
                StringValue(""),
                MakeStringAccessor(&TcpSocketBase::SetLinuxTcpMem, &TcpSocketBase::GetLinuxTcpMem),
                MakeStringChecker())
            .AddAttribute("RcvBufferOffset",
                          "With LinuxRcvWindow, the receive buffers of the driver taken by "
                          "the frames that the node received before the SYN of the "
                          "connection, counted from the first buffer of a page",
                          UintegerValue(0),
                          MakeUintegerAccessor(&TcpSocketBase::m_rcvBufferOffset),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("ForcedPush",
                          "Set PSH as Linux, which lays out the data written in skbs "
                          "(tcp_sendmsg_locked()): on the segment ending the skb that holds "
                          "the end of each application write, on the segment ending the skb "
                          "filled past half the largest window of the peer since the last PSH "
                          "(forced_push()), and on the last segment of each burst of new data "
                          "that Linux sends as one GSO packet (__tcp_transmit_skb())",
                          BooleanValue(false),
                          MakeBooleanAccessor(&TcpSocketBase::m_forcedPush),
                          MakeBooleanChecker())
            .AddAttribute("PruneOverhead",
                          "Memory overhead of the buffered data relative to its size, as "
                          "the truesize of Linux. If positive, the receiver accounts the "
                          "data it holds at this cost against twice RcvBufSize and, over it, "
                          "drops out-of-order data and clamps its window as Linux "
                          "(tcp_prune_queue()), reneging on its SACKs; 0 disables it. "
                          "Ignored with LinuxRcvWindow, which accounts the memory of the "
                          "skbs of Linux instead",
                          DoubleValue(0),
                          MakeDoubleAccessor(&TcpSocketBase::m_pruneOverhead),
                          MakeDoubleChecker<double>(0))
            .AddAttribute("AlignSackOption",
                          "Precede the SACK option with two NOP options, as Linux does "
                          "(tcp_options_write()), aligning the SACK blocks on 32 bits, as the "
                          "TCP header of the example of RFC 2018 (Section 3), at the cost of "
                          "two bytes, and four with the timestamps, of each ACK with SACK",
                          BooleanValue(false),
                          MakeBooleanAccessor(&TcpSocketBase::m_alignSackOption),
                          MakeBooleanChecker())
            .AddAttribute("SackRenegingTimeout",
                          "On SACK reneging by the receiver, a cumulative ACK reaching SACKed "
                          "data, expire the retransmission timer after half the smoothed RTT "
                          "(at least 10 ms), as Linux (tcp_check_sack_reneging()), rather "
                          "than mark the segment lost",
                          BooleanValue(false),
                          MakeBooleanAccessor(&TcpSocketBase::SetSackRenegingTimeout),
                          MakeBooleanChecker())
            .AddAttribute("TsoDefer",
                          "Defer, as Linux (tcp_tso_should_defer()), the transmission of new "
                          "data that the windows allow only in part, to send a larger burst "
                          "on the next ACK: outside of loss recovery, within 1 ms of the last "
                          "transmission, while less than a GSO packet and a third of the "
                          "window can be sent and the oldest outstanding segment was sent at "
                          "least half an RTT ago. With Nagle's algorithm (TcpNoDelay false), "
                          "hold a short last segment only while a short segment sent is not "
                          "acknowledged, as Linux (tcp_nagle_check(), Minshall's variant)",
                          BooleanValue(false),
                          MakeBooleanAccessor(&TcpSocketBase::m_tsoDefer),
                          MakeBooleanChecker())
            .AddAttribute("SndSkbOverhead",
                          "With ForcedPush or TsoDefer, if positive, account the send buffer "
                          "as Linux: each skb of the data written (as tcp_sendmsg_locked() "
                          "fills them) or sent costs this many bytes beyond its data "
                          "(truesize), a write takes no new skb once the cost reaches "
                          "SndBufSize, returning the data it could queue, and the application "
                          "may write again only once the free buffer is half the cost "
                          "(sk_stream_is_writeable()); 0 disables it",
                          UintegerValue(0),
                          MakeUintegerAccessor(&TcpSocketBase::m_sndSkbOverhead),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("LinuxCwndLimited",
                          "Grow the congestion window only while it limits the transmissions, "
                          "as Linux (tcp_is_cwnd_limited()): while the flight filled it or a "
                          "TSO deferral was due to it in the window of data last sent, or, in "
                          "slow start, below twice the largest flight of that window "
                          "(tcp_cwnd_validate())",
                          BooleanValue(false),
                          MakeBooleanAccessor(&TcpSocketBase::m_linuxCwndLimited),
                          MakeBooleanChecker())
            .AddAttribute("ForwardRetransmit",
                          "In SACK loss recovery, when no segment is deemed lost and no new "
                          "data can be sent, retransmit the first unSACKed segment below the "
                          "highest SACK (RFC 6675, NextSeg() rule (3)). Linux, which removed "
                          "its forward retransmission (tcp_xmit_retransmit_queue()), "
                          "retransmits only the segments deemed lost, those above the "
                          "highest SACK included",
                          BooleanValue(true),
                          MakeBooleanAccessor(&TcpSocketBase::m_forwardRetransmit),
                          MakeBooleanChecker())
            .AddAttribute("LimitedTransmit",
                          "Enable limited transmit",
                          BooleanValue(true),
                          MakeBooleanAccessor(&TcpSocketBase::m_limitedTx),
                          MakeBooleanChecker())
            .AddAttribute("UseEcn",
                          "Parameter to set ECN functionality",
                          EnumValue(TcpSocketState::Off),
                          MakeEnumAccessor<TcpSocketState::UseEcn_t>(&TcpSocketBase::SetUseEcn),
                          MakeEnumChecker(TcpSocketState::Off,
                                          "Off",
                                          TcpSocketState::On,
                                          "On",
                                          TcpSocketState::AcceptOnly,
                                          "AcceptOnly"))
            .AddAttribute("UseAbe",
                          "Parameter to set ABE functionality",
                          BooleanValue(false),
                          MakeBooleanAccessor(&TcpSocketBase::m_useAbe),
                          MakeBooleanChecker())
            .AddTraceSource("RTO",
                            "Retransmission timeout",
                            MakeTraceSourceAccessor(&TcpSocketBase::m_rto),
                            "ns3::TracedValueCallback::Time")
            .AddTraceSource("RTT",
                            "Smoothed RTT",
                            MakeTraceSourceAccessor(&TcpSocketBase::m_srttTrace),
                            "ns3::TracedValueCallback::Time")
            .AddTraceSource("LastRTT",
                            "RTT of the last (S)ACKed packet",
                            MakeTraceSourceAccessor(&TcpSocketBase::m_lastRttTrace),
                            "ns3::TracedValueCallback::Time")
            .AddTraceSource("NextTxSequence",
                            "Next sequence number to send (SND.NXT)",
                            MakeTraceSourceAccessor(&TcpSocketBase::m_nextTxSequenceTrace),
                            "ns3::SequenceNumber32TracedValueCallback")
            .AddTraceSource("HighestSequence",
                            "Highest sequence number ever sent in socket's life time",
                            MakeTraceSourceAccessor(&TcpSocketBase::m_highTxMarkTrace),
                            "ns3::TracedValueCallback::SequenceNumber32")
            .AddTraceSource("State",
                            "TCP state",
                            MakeTraceSourceAccessor(&TcpSocketBase::m_state),
                            "ns3::TcpStatesTracedValueCallback")
            .AddTraceSource("CongState",
                            "TCP Congestion machine state",
                            MakeTraceSourceAccessor(&TcpSocketBase::m_congStateTrace),
                            "ns3::TcpSocketState::TcpCongStatesTracedValueCallback")
            .AddTraceSource("EcnState",
                            "Trace ECN state change of socket",
                            MakeTraceSourceAccessor(&TcpSocketBase::m_ecnStateTrace),
                            "ns3::TcpSocketState::EcnStatesTracedValueCallback")
            .AddTraceSource("AdvWND",
                            "Advertised Window Size",
                            MakeTraceSourceAccessor(&TcpSocketBase::m_advWnd),
                            "ns3::TracedValueCallback::Uint32")
            .AddTraceSource("RWND",
                            "Remote side's flow control window",
                            MakeTraceSourceAccessor(&TcpSocketBase::m_rWnd),
                            "ns3::TracedValueCallback::Uint32")
            .AddTraceSource("BytesInFlight",
                            "Socket estimation of bytes in flight",
                            MakeTraceSourceAccessor(&TcpSocketBase::m_bytesInFlightTrace),
                            "ns3::TracedValueCallback::Uint32")
            .AddTraceSource("FackAwnd",
                            "Socket estimation of bytes in flight by FACK",
                            MakeTraceSourceAccessor(&TcpSocketBase::m_fackAwndTrace),
                            "ns3::TracedValueCallback::Uint32")
            .AddTraceSource("HighestRxSequence",
                            "Highest sequence number received from peer",
                            MakeTraceSourceAccessor(&TcpSocketBase::m_highRxMark),
                            "ns3::TracedValueCallback::SequenceNumber32")
            .AddTraceSource("HighestRxAck",
                            "Highest ack received from peer",
                            MakeTraceSourceAccessor(&TcpSocketBase::m_highRxAckMark),
                            "ns3::TracedValueCallback::SequenceNumber32")
            .AddTraceSource("PacingRate",
                            "The current TCP pacing rate",
                            MakeTraceSourceAccessor(&TcpSocketBase::m_pacingRateTrace),
                            "ns3::TracedValueCallback::DataRate")
            .AddTraceSource("CongestionWindow",
                            "The TCP connection's congestion window",
                            MakeTraceSourceAccessor(&TcpSocketBase::m_cWndTrace),
                            "ns3::TracedValueCallback::Uint32")
            .AddTraceSource("CongestionWindowInflated",
                            "The TCP connection's congestion window inflates as in older RFC",
                            MakeTraceSourceAccessor(&TcpSocketBase::m_cWndInflTrace),
                            "ns3::TracedValueCallback::Uint32")
            .AddTraceSource("SlowStartThreshold",
                            "TCP slow start threshold (bytes)",
                            MakeTraceSourceAccessor(&TcpSocketBase::m_ssThTrace),
                            "ns3::TracedValueCallback::Uint32")
            .AddTraceSource("Tx",
                            "Send tcp packet to IP protocol",
                            MakeTraceSourceAccessor(&TcpSocketBase::m_txTrace),
                            "ns3::TcpSocketBase::TcpTxRxTracedCallback")
            .AddTraceSource("Retransmission",
                            "Notification of a TCP retransmission",
                            MakeTraceSourceAccessor(&TcpSocketBase::m_retransmissionTrace),
                            "ns3::TcpSocketBase::RetransmissionCallback")
            .AddTraceSource("Rx",
                            "Receive tcp packet from IP protocol",
                            MakeTraceSourceAccessor(&TcpSocketBase::m_rxTrace),
                            "ns3::TcpSocketBase::TcpTxRxTracedCallback")
            .AddTraceSource("EcnEchoSeq",
                            "Sequence of last received ECN Echo",
                            MakeTraceSourceAccessor(&TcpSocketBase::m_ecnEchoSeq),
                            "ns3::SequenceNumber32TracedValueCallback")
            .AddTraceSource("EcnCeSeq",
                            "Sequence of last received CE",
                            MakeTraceSourceAccessor(&TcpSocketBase::m_ecnCESeq),
                            "ns3::SequenceNumber32TracedValueCallback")
            .AddTraceSource("EcnCwrSeq",
                            "Sequence of last received CWR",
                            MakeTraceSourceAccessor(&TcpSocketBase::m_ecnCWRSeq),
                            "ns3::SequenceNumber32TracedValueCallback");
    return tid;
}

TcpSocketBase::TcpSocketBase()
    : TcpSocket()
{
    NS_LOG_FUNCTION(this);

    m_txBuffer = CreateObject<TcpTxBuffer>();
    m_txBuffer->SetRWndCallback(MakeCallback(&TcpSocketBase::GetRWnd, this));
    m_tcb = CreateObject<TcpSocketState>();
    m_rateOps = CreateObject<TcpRateLinux>();

    m_sndFack = 0;
    m_outstandingRetransBytes = 0;

    m_tcb->m_rxBuffer = CreateObject<TcpRxBuffer>();

    m_tcb->m_pacingRate = m_tcb->m_maxPacingRate;
    m_pacingTimer.SetFunction(&TcpSocketBase::NotifyPacingPerformed, this);

    m_tcb->m_sendEmptyPacketCallback = MakeCallback(&TcpSocketBase::SendEmptyPacket, this);

    bool ok;

    ok = m_tcb->TraceConnectWithoutContext(
        "PacingRate",
        MakeCallback(&TcpSocketBase::UpdatePacingRateTrace, this));
    NS_ASSERT_MSG(ok, "Could not connect trace source PacingRate");

    ok = m_tcb->TraceConnectWithoutContext("CongestionWindow",
                                           MakeCallback(&TcpSocketBase::UpdateCwnd, this));
    NS_ASSERT_MSG(ok, "Could not connect trace source CongestionWindow");

    ok = m_tcb->TraceConnectWithoutContext("CongestionWindowInflated",
                                           MakeCallback(&TcpSocketBase::UpdateCwndInfl, this));
    NS_ASSERT_MSG(ok, "Could not connect trace source CongestionWindowInflated");

    ok = m_tcb->TraceConnectWithoutContext("SlowStartThreshold",
                                           MakeCallback(&TcpSocketBase::UpdateSsThresh, this));
    NS_ASSERT_MSG(ok, "Could not connect trace source SlowStartThreshold");

    ok = m_tcb->TraceConnectWithoutContext("CongState",
                                           MakeCallback(&TcpSocketBase::UpdateCongState, this));
    NS_ASSERT_MSG(ok, "Could not connect trace source CongState");

    ok = m_tcb->TraceConnectWithoutContext("EcnState",
                                           MakeCallback(&TcpSocketBase::UpdateEcnState, this));
    NS_ASSERT_MSG(ok, "Could not connect trace source EcnState");

    ok =
        m_tcb->TraceConnectWithoutContext("NextTxSequence",
                                          MakeCallback(&TcpSocketBase::UpdateNextTxSequence, this));
    NS_ASSERT_MSG(ok, "Could not connect trace source NextTxSequence");

    ok = m_tcb->TraceConnectWithoutContext("HighestSequence",
                                           MakeCallback(&TcpSocketBase::UpdateHighTxMark, this));
    NS_ASSERT_MSG(ok, "Could not connect trace source HighestSequence");

    ok = m_tcb->TraceConnectWithoutContext("BytesInFlight",
                                           MakeCallback(&TcpSocketBase::UpdateBytesInFlight, this));
    NS_ASSERT_MSG(ok, "Could not connect trace source BytesInFlight");

    ok = m_tcb->TraceConnectWithoutContext("FackAwnd",
                                           MakeCallback(&TcpSocketBase::UpdateFackAwnd, this));
    NS_ASSERT_MSG(ok, "Could not connect trace source FackAwnd");

    ok = m_tcb->TraceConnectWithoutContext("RTT", MakeCallback(&TcpSocketBase::UpdateRtt, this));
    NS_ASSERT_MSG(ok, "Could not connect trace source RTT");

    ok = m_tcb->TraceConnectWithoutContext("LastRTT",
                                           MakeCallback(&TcpSocketBase::UpdateLastRtt, this));
    NS_ASSERT_MSG(ok, "Could not connect trace source LastRTT");
}

void
TcpSocketBase::NotifyConstructionCompleted()
{
    NS_LOG_FUNCTION(this);

    SetUseAbe(m_useAbe);
}

TcpSocketBase::TcpSocketBase(const TcpSocketBase& sock)
    : TcpSocket(sock),
      // copy object::m_tid and socket::callbacks
      m_dupAckCount(sock.m_dupAckCount),
      m_delAckCount(0),
      m_delAckMaxCount(sock.m_delAckMaxCount),
      m_quickAck(sock.m_quickAck),
      m_sackCompression(sock.m_sackCompression),
      m_sackCompressionDelay(sock.m_sackCompressionDelay),
      m_sackCompressionSlack(sock.m_sackCompressionSlack),
      m_keepRecoverySsThresh(sock.m_keepRecoverySsThresh),
      m_sackRestartsRto(sock.m_sackRestartsRto),
      m_sackRttSamples(sock.m_sackRttSamples),
      m_retransmitRestartsRto(sock.m_retransmitRestartsRto),
      m_adaptiveDelAck(sock.m_adaptiveDelAck),
      m_linuxRcvWindow(sock.m_linuxRcvWindow),
      m_rcvHeadTruesize(sock.m_rcvHeadTruesize),
      m_rcvFragTruesize(sock.m_rcvFragTruesize),
      m_rcvCopyBreak(sock.m_rcvCopyBreak),
      m_rcvFragPageSize(sock.m_rcvFragPageSize),
      m_rcvBufferOffset(sock.m_rcvBufferOffset),
      m_lxRxPackets(sock.m_lxRxPackets),
      m_lxTcpMem(sock.m_lxTcpMem),
      m_noDelay(sock.m_noDelay),
      m_fillRcvWindow(sock.m_fillRcvWindow),
      m_maxSndWnd(sock.m_maxSndWnd),
      m_forcedPush(sock.m_forcedPush),
      m_tsoDefer(sock.m_tsoDefer),
      m_linuxCwndLimited(sock.m_linuxCwndLimited),
      m_sndSkbOverhead(sock.m_sndSkbOverhead),
      m_pruneOverhead(sock.m_pruneOverhead),
      m_alignSackOption(sock.m_alignSackOption),
      m_sackRenegingTimeout(sock.m_sackRenegingTimeout),
      m_synCount(sock.m_synCount),
      m_synRetries(sock.m_synRetries),
      m_dataRetrCount(sock.m_dataRetrCount),
      m_dataRetries(sock.m_dataRetries),
      m_rto(sock.m_rto),
      m_minRto(sock.m_minRto),
      m_clockGranularity(sock.m_clockGranularity),
      m_rtoTimerGranularity(sock.m_rtoTimerGranularity),
      m_delAckTimeout(sock.m_delAckTimeout),
      m_persistTimeout(sock.m_persistTimeout),
      m_sourceRoute(sock.m_sourceRoute),
      m_appSourceRoute(sock.m_appSourceRoute),
      m_keepAlive(sock.m_keepAlive),
      m_keepAliveTime(sock.m_keepAliveTime),
      m_keepAliveInterval(sock.m_keepAliveInterval),
      m_keepAliveRetries(sock.m_keepAliveRetries),
      m_cnTimeout(sock.m_cnTimeout),
      m_endPoint(nullptr),
      m_endPoint6(nullptr),
      m_node(sock.m_node),
      m_tcp(sock.m_tcp),
      m_state(sock.m_state),
      m_errno(sock.m_errno),
      m_closeNotified(sock.m_closeNotified),
      m_closeOnEmpty(sock.m_closeOnEmpty),
      m_shutdownSend(sock.m_shutdownSend),
      m_shutdownRecv(sock.m_shutdownRecv),
      m_connected(sock.m_connected),
      m_msl(sock.m_msl),
      m_maxWinSize(sock.m_maxWinSize),
      m_bytesAckedNotProcessed(sock.m_bytesAckedNotProcessed),
      m_rWnd(sock.m_rWnd),
      m_advRightEdge(sock.m_advRightEdge),
      m_advRightEdgeArmed(sock.m_advRightEdgeArmed),
      m_highRxMark(sock.m_highRxMark),
      m_highRxAckMark(sock.m_highRxAckMark),
      m_sackEnabled(sock.m_sackEnabled),
      m_winScalingEnabled(sock.m_winScalingEnabled),
      m_rcvWindShift(sock.m_rcvWindShift),
      m_sndWindShift(sock.m_sndWindShift),
      m_timestampEnabled(sock.m_timestampEnabled),
      m_timestampToEcho(sock.m_timestampToEcho),
      m_randomTsOffset(sock.m_randomTsOffset),
      m_fackEnabled(sock.m_fackEnabled),
      m_recover(sock.m_recover),
      m_recoverActive(sock.m_recoverActive),
      m_retxThresh(sock.m_retxThresh),
      m_limitedTx(sock.m_limitedTx),
      m_forwardRetransmit(sock.m_forwardRetransmit),
      m_nonSackUndo(sock.m_nonSackUndo),
      m_partialAckInWindow(sock.m_partialAckInWindow),
      m_recoveryReentry(sock.m_recoveryReentry),
      m_linuxCaRtt(sock.m_linuxCaRtt),
      m_linuxPktsAcked(sock.m_linuxPktsAcked),
      m_activeOpen(sock.m_activeOpen),
      m_isn(sock.m_isn),
      m_sndUrgentPoint(sock.m_sndUrgentPoint),
      m_rcvUrgentPoint(sock.m_rcvUrgentPoint),
      m_sndUrgentArmed(sock.m_sndUrgentArmed),
      m_rcvUrgentArmed(sock.m_rcvUrgentArmed),
      m_advertisedMss(sock.m_advertisedMss),
      m_segmentSizeAdjusted(sock.m_segmentSizeAdjusted),
      m_txTrace(sock.m_txTrace),
      m_rxTrace(sock.m_rxTrace),
      m_pacingTimer(Timer::CANCEL_ON_DESTROY),
      m_ecnEchoSeq(sock.m_ecnEchoSeq),
      m_ecnCESeq(sock.m_ecnCESeq),
      m_ecnCWRSeq(sock.m_ecnCWRSeq),
      m_cwrPending(sock.m_cwrPending)
{
    NS_LOG_FUNCTION(this);
    NS_LOG_LOGIC("Invoked the copy constructor");
    // Copy the rtt estimator if it is set
    if (sock.m_rtt)
    {
        m_rtt = sock.m_rtt->Copy();
    }
    // Reset all callbacks to null
    Callback<void, Ptr<Socket>> vPS = MakeNullCallback<void, Ptr<Socket>>();
    Callback<void, Ptr<Socket>, const Address&> vPSA =
        MakeNullCallback<void, Ptr<Socket>, const Address&>();
    Callback<void, Ptr<Socket>, uint32_t> vPSUI = MakeNullCallback<void, Ptr<Socket>, uint32_t>();
    SetConnectCallback(vPS, vPS);
    SetDataSentCallback(vPSUI);
    SetSendCallback(vPSUI);
    SetRecvCallback(vPS);
    m_txBuffer = CopyObject(sock.m_txBuffer);
    m_txBuffer->SetRWndCallback(MakeCallback(&TcpSocketBase::GetRWnd, this));
    m_tcb = CopyObject(sock.m_tcb);
    m_tcb->m_rxBuffer = CopyObject(sock.m_tcb->m_rxBuffer);

    m_tcb->m_pacingRate = m_tcb->m_maxPacingRate;
    m_pacingTimer.SetFunction(&TcpSocketBase::NotifyPacingPerformed, this);

    m_rateOps = CreateObject<TcpRateLinux>();

    if (sock.m_congestionControl)
    {
        m_congestionControl = sock.m_congestionControl->Fork();
        m_congestionControl->Init(m_tcb);
        m_congestionControl->SetRateOps(m_rateOps);
    }

    if (sock.m_recoveryOps)
    {
        m_recoveryOps = sock.m_recoveryOps->Fork();
    }

    if (m_tcb->m_sendEmptyPacketCallback.IsNull())
    {
        m_tcb->m_sendEmptyPacketCallback = MakeCallback(&TcpSocketBase::SendEmptyPacket, this);
    }

    m_sndFack = sock.m_sndFack;
    m_outstandingRetransBytes = sock.m_outstandingRetransBytes;

    bool ok;

    ok = m_tcb->TraceConnectWithoutContext(
        "PacingRate",
        MakeCallback(&TcpSocketBase::UpdatePacingRateTrace, this));
    NS_ASSERT_MSG(ok, "Could not connect trace source PacingRate");

    ok = m_tcb->TraceConnectWithoutContext("CongestionWindow",
                                           MakeCallback(&TcpSocketBase::UpdateCwnd, this));
    NS_ASSERT_MSG(ok, "Could not connect trace source CongestionWindow");

    ok = m_tcb->TraceConnectWithoutContext("CongestionWindowInflated",
                                           MakeCallback(&TcpSocketBase::UpdateCwndInfl, this));
    NS_ASSERT_MSG(ok, "Could not connect trace source CongestionWindowInflated");

    ok = m_tcb->TraceConnectWithoutContext("SlowStartThreshold",
                                           MakeCallback(&TcpSocketBase::UpdateSsThresh, this));
    NS_ASSERT_MSG(ok, "Could not connect trace source SlowStartThreshold");

    ok = m_tcb->TraceConnectWithoutContext("CongState",
                                           MakeCallback(&TcpSocketBase::UpdateCongState, this));
    NS_ASSERT_MSG(ok, "Could not connect trace source CongState");

    ok = m_tcb->TraceConnectWithoutContext("EcnState",
                                           MakeCallback(&TcpSocketBase::UpdateEcnState, this));
    NS_ASSERT_MSG(ok, "Could not connect trace source EcnState");

    ok =
        m_tcb->TraceConnectWithoutContext("NextTxSequence",
                                          MakeCallback(&TcpSocketBase::UpdateNextTxSequence, this));
    NS_ASSERT_MSG(ok, "Could not connect trace source NextTxSequence");

    ok = m_tcb->TraceConnectWithoutContext("HighestSequence",
                                           MakeCallback(&TcpSocketBase::UpdateHighTxMark, this));
    NS_ASSERT_MSG(ok, "Could not connect trace source HighestSequence");
    ok = m_tcb->TraceConnectWithoutContext("BytesInFlight",
                                           MakeCallback(&TcpSocketBase::UpdateBytesInFlight, this));
    NS_ASSERT_MSG(ok, "Could not connect trace source BytesInFlight");

    ok = m_tcb->TraceConnectWithoutContext("FackAwnd",
                                           MakeCallback(&TcpSocketBase::UpdateFackAwnd, this));

    ok = m_tcb->TraceConnectWithoutContext("RTT", MakeCallback(&TcpSocketBase::UpdateRtt, this));
    NS_ASSERT_MSG(ok, "Could not connect trace source RTT");

    ok = m_tcb->TraceConnectWithoutContext("LastRTT",
                                           MakeCallback(&TcpSocketBase::UpdateLastRtt, this));
    NS_ASSERT_MSG(ok, "Could not connect trace source LastRTT");
}

TcpSocketBase::~TcpSocketBase()
{
    NS_LOG_FUNCTION(this);
    m_node = nullptr;
    if (m_endPoint != nullptr)
    {
        NS_ASSERT(m_tcp);
        /*
         * Upon Bind, an Ipv4Endpoint is allocated and set to m_endPoint, and
         * DestroyCallback is set to TcpSocketBase::Destroy. If we called
         * m_tcp->DeAllocate, it will destroy its Ipv4EndpointDemux::DeAllocate,
         * which in turn destroys my m_endPoint, and in turn invokes
         * TcpSocketBase::Destroy to nullify m_node, m_endPoint, and m_tcp.
         */
        NS_ASSERT(m_endPoint != nullptr);
        m_tcp->DeAllocate(m_endPoint);
        NS_ASSERT(m_endPoint == nullptr);
    }
    if (m_endPoint6 != nullptr)
    {
        NS_ASSERT(m_tcp);
        NS_ASSERT(m_endPoint6 != nullptr);
        m_tcp->DeAllocate(m_endPoint6);
        NS_ASSERT(m_endPoint6 == nullptr);
    }
    m_tcp = nullptr;
    CancelAllTimers();
}

/* Associate a node with this TCP socket */
void
TcpSocketBase::SetNode(Ptr<Node> node)
{
    m_node = node;
}

/* Associate the L4 protocol (e.g. mux/demux) with this socket */
void
TcpSocketBase::SetTcp(Ptr<TcpL4Protocol> tcp)
{
    m_tcp = tcp;
}

/* Set an RTT estimator with this socket */
void
TcpSocketBase::SetRtt(Ptr<RttEstimator> rtt)
{
    m_rtt = rtt;
}

/* Inherit from Socket class: Returns error code */
Socket::SocketErrno
TcpSocketBase::GetErrno() const
{
    return m_errno;
}

/* Inherit from Socket class: Returns socket type, NS3_SOCK_STREAM */
Socket::SocketType
TcpSocketBase::GetSocketType() const
{
    return NS3_SOCK_STREAM;
}

/* Inherit from Socket class: Returns associated node */
Ptr<Node>
TcpSocketBase::GetNode() const
{
    return m_node;
}

/* Inherit from Socket class: Bind socket to an end-point in TcpL4Protocol */
int
TcpSocketBase::Bind()
{
    NS_LOG_FUNCTION(this);
    m_endPoint = m_tcp->Allocate();
    if (nullptr == m_endPoint)
    {
        m_errno = ERROR_ADDRNOTAVAIL;
        return -1;
    }

    m_tcp->AddSocket(this);

    return SetupCallback();
}

int
TcpSocketBase::Bind6()
{
    NS_LOG_FUNCTION(this);
    m_endPoint6 = m_tcp->Allocate6();
    if (nullptr == m_endPoint6)
    {
        m_errno = ERROR_ADDRNOTAVAIL;
        return -1;
    }

    m_tcp->AddSocket(this);

    return SetupCallback();
}

/* Inherit from Socket class: Bind socket (with specific address) to an end-point in TcpL4Protocol
 */
int
TcpSocketBase::Bind(const Address& address)
{
    NS_LOG_FUNCTION(this << address);
    if (InetSocketAddress::IsMatchingType(address))
    {
        InetSocketAddress transport = InetSocketAddress::ConvertFrom(address);
        Ipv4Address ipv4 = transport.GetIpv4();
        uint16_t port = transport.GetPort();
        if (ipv4 == Ipv4Address::GetAny() && port == 0)
        {
            m_endPoint = m_tcp->Allocate();
        }
        else if (ipv4 == Ipv4Address::GetAny() && port != 0)
        {
            m_endPoint = m_tcp->Allocate(GetBoundNetDevice(), port);
        }
        else if (ipv4 != Ipv4Address::GetAny() && port == 0)
        {
            m_endPoint = m_tcp->Allocate(ipv4);
        }
        else if (ipv4 != Ipv4Address::GetAny() && port != 0)
        {
            m_endPoint = m_tcp->Allocate(GetBoundNetDevice(), ipv4, port);
        }
        if (nullptr == m_endPoint)
        {
            m_errno = port ? ERROR_ADDRINUSE : ERROR_ADDRNOTAVAIL;
            return -1;
        }
    }
    else if (Inet6SocketAddress::IsMatchingType(address))
    {
        Inet6SocketAddress transport = Inet6SocketAddress::ConvertFrom(address);
        Ipv6Address ipv6 = transport.GetIpv6();
        uint16_t port = transport.GetPort();
        if (ipv6 == Ipv6Address::GetAny() && port == 0)
        {
            m_endPoint6 = m_tcp->Allocate6();
        }
        else if (ipv6 == Ipv6Address::GetAny() && port != 0)
        {
            m_endPoint6 = m_tcp->Allocate6(GetBoundNetDevice(), port);
        }
        else if (ipv6 != Ipv6Address::GetAny() && port == 0)
        {
            m_endPoint6 = m_tcp->Allocate6(ipv6);
        }
        else if (ipv6 != Ipv6Address::GetAny() && port != 0)
        {
            m_endPoint6 = m_tcp->Allocate6(GetBoundNetDevice(), ipv6, port);
        }
        if (nullptr == m_endPoint6)
        {
            m_errno = port ? ERROR_ADDRINUSE : ERROR_ADDRNOTAVAIL;
            return -1;
        }
    }
    else
    {
        m_errno = ERROR_INVAL;
        return -1;
    }

    m_tcp->AddSocket(this);

    NS_LOG_LOGIC("TcpSocketBase " << this << " got an endpoint: " << m_endPoint);

    return SetupCallback();
}

void
TcpSocketBase::SetInitialSSThresh(uint32_t threshold)
{
    NS_ABORT_MSG_UNLESS(
        (m_state == CLOSED) || threshold == m_tcb->m_initialSsThresh,
        "TcpSocketBase::SetSSThresh() cannot change initial ssThresh after connection started.");

    m_tcb->m_initialSsThresh = threshold;
}

uint32_t
TcpSocketBase::GetInitialSSThresh() const
{
    return m_tcb->m_initialSsThresh;
}

void
TcpSocketBase::SetInitialCwnd(uint32_t cwnd)
{
    NS_ABORT_MSG_UNLESS(
        (m_state == CLOSED) || cwnd == m_tcb->m_initialCWnd,
        "TcpSocketBase::SetInitialCwnd() cannot change initial cwnd after connection started.");

    m_tcb->m_initialCWnd = cwnd;
}

uint32_t
TcpSocketBase::GetInitialCwnd() const
{
    return m_tcb->m_initialCWnd;
}

/* Inherit from Socket class: Initiate connection to a remote address:port */
int
TcpSocketBase::Connect(const Address& address)
{
    NS_LOG_FUNCTION(this << address);

    // If haven't do so, Bind() this socket first
    if (InetSocketAddress::IsMatchingType(address))
    {
        // A local OPEN call for an invalid remote IP address must be rejected
        // as an error (RFC 9293, Section 3.9.1.5, MUST-46): the unspecified
        // address, the limited broadcast, a multicast address, and the
        // directed broadcast of a subnet the node is attached to. The
        // directed broadcast of a remote subnet cannot be told apart from a
        // unicast address, as on a real host; a /31 or /32 subnet has none
        // (RFC 3021).
        Ipv4Address peer = InetSocketAddress::ConvertFrom(address).GetIpv4();
        bool invalid = peer.IsAny() || peer.IsBroadcast() || peer.IsMulticast();
        Ptr<Ipv4> ipv4 = m_node->GetObject<Ipv4>();
        for (uint32_t i = 0; ipv4 && !invalid && i < ipv4->GetNInterfaces(); ++i)
        {
            for (uint32_t j = 0; !invalid && j < ipv4->GetNAddresses(i); ++j)
            {
                Ipv4InterfaceAddress ifAddress = ipv4->GetAddress(i, j);
                invalid =
                    ifAddress.GetMask().GetPrefixLength() < 31 && peer == ifAddress.GetBroadcast();
            }
        }
        if (invalid)
        {
            NS_LOG_ERROR("Cannot connect to the invalid remote address " << peer);
            m_errno = ERROR_INVAL;
            return -1;
        }

        if (m_endPoint == nullptr)
        {
            if (Bind() == -1)
            {
                NS_ASSERT(m_endPoint == nullptr);
                return -1; // Bind() failed
            }
            NS_ASSERT(m_endPoint != nullptr);
        }
        InetSocketAddress transport = InetSocketAddress::ConvertFrom(address);
        m_endPoint->SetPeer(transport.GetIpv4(), transport.GetPort());
        m_endPoint6 = nullptr;

        // Get the appropriate local address and port number from the routing protocol and set up
        // endpoint
        if (SetupEndpoint() != 0)
        {
            NS_LOG_ERROR("Route to destination does not exist ?!");
            return -1;
        }
    }
    else if (Inet6SocketAddress::IsMatchingType(address))
    {
        // If we are operating on a v4-mapped address, translate the address to
        // a v4 address and re-call this function
        Inet6SocketAddress transport = Inet6SocketAddress::ConvertFrom(address);
        Ipv6Address v6Addr = transport.GetIpv6();
        if (v6Addr.IsIpv4MappedAddress())
        {
            Ipv4Address v4Addr = v6Addr.GetIpv4MappedAddress();
            return Connect(InetSocketAddress(v4Addr, transport.GetPort()));
        }

        // A local OPEN call for an invalid remote IP address must be rejected
        // as an error (RFC 9293, Section 3.9.1.5, MUST-46)
        if (v6Addr.IsMulticast() || v6Addr.IsAny())
        {
            NS_LOG_ERROR("Cannot connect to the invalid remote address " << v6Addr);
            m_errno = ERROR_INVAL;
            return -1;
        }

        if (m_endPoint6 == nullptr)
        {
            if (Bind6() == -1)
            {
                NS_ASSERT(m_endPoint6 == nullptr);
                return -1; // Bind() failed
            }
            NS_ASSERT(m_endPoint6 != nullptr);
        }
        m_endPoint6->SetPeer(v6Addr, transport.GetPort());
        m_endPoint = nullptr;

        // Get the appropriate local address and port number from the routing protocol and set up
        // endpoint
        if (SetupEndpoint6() != 0)
        {
            NS_LOG_ERROR("Route to destination does not exist ?!");
            return -1;
        }
    }
    else
    {
        m_errno = ERROR_INVAL;
        return -1;
    }

    // Re-initialize parameters in case this socket is being reused after CLOSE
    m_rtt->Reset();
    m_synCount = m_synRetries;
    m_dataRetrCount = m_dataRetries;

    // DoConnect() will do state-checking and send a SYN packet
    return DoConnect();
}

/* Inherit from Socket class: Listen on the endpoint for an incoming connection */
int
TcpSocketBase::Listen()
{
    NS_LOG_FUNCTION(this);

    // Linux quits EINVAL if we're not in CLOSED state, so match what they do
    if (m_state != CLOSED)
    {
        m_errno = ERROR_INVAL;
        return -1;
    }
    // In other cases, set the state to LISTEN and done
    NS_LOG_DEBUG("CLOSED -> LISTEN");
    m_state = LISTEN;
    return 0;
}

/* Inherit from Socket class: Kill this socket and signal the peer (if any) */
int
TcpSocketBase::Close()
{
    NS_LOG_FUNCTION(this);
    if (m_state == CLOSED && m_closeNotified)
    {
        // Already reset or closed, and the application was told
        return 0;
    }
    if (m_linuxRcvWindow)
    {
        // tcp_close() purges the receive queue, such as the skb of a FIN
        for (const auto& skb : m_lxRcvQueue)
        {
            LinuxMemUncharge(skb.truesize);
        }
        m_lxRcvQueue.clear();
    }
    /// @internal
    /// First we check to see if there is any unread rx data.
    /// \bugid{426} claims we should send reset in this case.
    if (m_tcb->m_rxBuffer->Size() != 0)
    {
        NS_LOG_WARN("Socket " << this << " << unread rx data during close.  Sending reset."
                              << "This is probably due to a bad sink application; check its code");
        SendRST();
        return 0;
    }

    if (m_txBuffer->SizeFromSequence(m_tcb->m_nextTxSequence) > 0)
    { // App close with pending data must wait until all data transmitted
        if (!m_closeOnEmpty)
        {
            m_closeOnEmpty = true;
            NS_LOG_INFO("Socket " << this << " deferring close, state " << TcpStateName[m_state]);
        }
        return 0;
    }
    return DoClose();
}

/* Inherit from Socket class: Signal a termination of send */
int
TcpSocketBase::ShutdownSend()
{
    NS_LOG_FUNCTION(this);

    // this prevents data from being added to the buffer
    m_shutdownSend = true;
    m_closeOnEmpty = true;
    // if buffer is already empty, send a fin now
    // otherwise fin will go when buffer empties.
    if (m_txBuffer->Size() == 0)
    {
        if (m_state == ESTABLISHED || m_state == CLOSE_WAIT)
        {
            NS_LOG_INFO("Empty tx buffer, send fin");
            SendEmptyPacket(TcpHeader::FIN);

            if (m_state == ESTABLISHED)
            { // On active close: I am the first one to send FIN
                NS_LOG_DEBUG("ESTABLISHED -> FIN_WAIT_1");
                m_state = FIN_WAIT_1;
            }
            else
            { // On passive close: Peer sent me FIN already
                NS_LOG_DEBUG("CLOSE_WAIT -> LAST_ACK");
                m_state = LAST_ACK;
            }
        }
    }

    return 0;
}

/* Inherit from Socket class: Signal a termination of receive */
int
TcpSocketBase::ShutdownRecv()
{
    NS_LOG_FUNCTION(this);
    m_shutdownRecv = true;
    return 0;
}

/* Inherit from Socket class: Send a packet. Parameter flags is not used.
    Packet has no TCP header. Invoked by upper-layer application */
int
TcpSocketBase::Send(Ptr<Packet> p, uint32_t flags)
{
    NS_LOG_FUNCTION(this << p << flags);
    NS_ABORT_MSG_IF(flags & ~static_cast<uint32_t>(MSG_FLAG_OOB),
                    "only MSG_FLAG_OOB is supported in TcpSocketBase::Send()");
    if (m_state == ESTABLISHED || m_state == SYN_SENT || m_state == CLOSE_WAIT)
    {
        if (m_shutdownSend)
        {
            m_errno = ERROR_SHUTDOWN;
            return -1;
        }

        if ((m_forcedPush || m_tsoDefer) && p->GetSize() > 0)
        {
            if (m_lxNoSpace && !IsLinuxWritable())
            {
                // The application waits for the buffer to be writable
                m_errno = ERROR_MSGSIZE;
                return -1;
            }
            m_lxNoSpace = false;
            SequenceNumber32 start = m_txBuffer->TailSequence();
            uint32_t accepted = QueueLinuxWrite(start, start + SequenceNumber32(p->GetSize()));
            if (accepted == 0)
            {
                m_lxNoSpace = true;
                m_errno = ERROR_MSGSIZE;
                return -1;
            }
            if (accepted < p->GetSize())
            {
                // The application writes the rest once the buffer is writable
                p = p->CreateFragment(0, accepted);
                m_lxNoSpace = true;
            }
        }
        // Store the packet into Tx buffer
        if (!m_txBuffer->Add(p))
        { // TxBuffer overflow, send failed
            m_errno = ERROR_MSGSIZE;
            return -1;
        }

        if (flags & MSG_FLAG_OOB)
        {
            // The urgent pointer points to the sequence number of the octet
            // following the urgent data (RFC 9293, Section 3.8.5, MUST-62).
            // Advancing it segment after segment supports a sequence of urgent
            // data of any length (MUST-31)
            m_sndUrgentPoint = m_txBuffer->TailSequence();
            m_sndUrgentArmed = true;
            NS_LOG_INFO("Urgent data up to " << m_sndUrgentPoint);
        }

        m_rateOps->CalculateAppLimited(m_tcb->m_cWnd,
                                       m_tcb->m_bytesInFlight,
                                       m_tcb->m_segmentSize,
                                       m_txBuffer->TailSequence(),
                                       m_tcb->m_nextTxSequence,
                                       m_txBuffer->GetLost(),
                                       m_txBuffer->GetRetransmitsCount());

        // Submit the data to lower layers
        NS_LOG_LOGIC("txBufSize=" << m_txBuffer->Size() << " state " << TcpStateName[m_state]);
        if ((m_state == ESTABLISHED || m_state == CLOSE_WAIT) && AvailableWindow() > 0)
        { // Try to send the data out: Add a little step to allow the application
            // to fill the buffer
            if (!m_sendPendingDataEvent.IsPending())
            {
                m_sendPendingDataEvent =
                    Simulator::Schedule(TimeStep(1), &TcpSocketBase::SendWrittenData, this);
            }
        }
        return p->GetSize();
    }
    else
    { // Connection not established yet
        m_errno = ERROR_NOTCONN;
        return -1; // Send failure
    }
}

/* Inherit from Socket class: In TcpSocketBase, it is same as Send() call */
int
TcpSocketBase::SendTo(Ptr<Packet> p, uint32_t flags, const Address& /* address */)
{
    return Send(p, flags); // SendTo() and Send() are the same
}

/* Inherit from Socket class: Return data to upper-layer application. Parameter flags
   is not used. Data is returned as a packet of size no larger than maxSize */
Ptr<Packet>
TcpSocketBase::Recv(uint32_t maxSize, uint32_t flags)
{
    NS_LOG_FUNCTION(this);
    NS_ABORT_MSG_IF(flags, "use of flags is not supported in TcpSocketBase::Recv()");
    if (m_tcb->m_rxBuffer->Size() == 0 && m_state == CLOSE_WAIT)
    {
        return Create<Packet>(); // Send EOF on connection close
    }
    uint32_t oldWin = GetReceiveWindow();
    Ptr<Packet> outPacket = m_tcb->m_rxBuffer->Extract(maxSize);
    if (m_rcvUrgentArmed && m_rcvUrgentPoint <= GetRxReadPoint())
    {
        // Disarmed as soon as the urgent data is read: left armed, the urgent
        // point would compare as ahead again once the read point moves 2^31
        // past it
        m_rcvUrgentArmed = false;
    }
    uint32_t newWin = GetReceiveWindow();

    // If the receive window had (nearly) closed, the application read just
    // reopened it. Proactively advertise the larger window with an ACK instead
    // of waiting for the sender's persist (zero-window) timer to probe. This
    // mirrors SetRcvBufSize() and follows the receiver SWS-avoidance rule of
    // @RFC{1122} sec 4.2.3.3: only advertise once the window grows by at least one
    // full-sized segment (or half the buffer), and only from a (near) closed
    // state, to avoid sending a redundant ACK on every read (see @issueid{125}).
    bool windowUpdate = oldWin < m_tcb->m_segmentSize && newWin > oldWin &&
                        newWin >= std::min(m_tcb->m_segmentSize, GetRcvBufSize() / 2);
    if (m_linuxRcvWindow && m_lxEstablished)
    {
        // As tcp_cleanup_rbuf() of Linux: once the application has read,
        // announce a window that has at least doubled, if the one
        // advertised has fallen to half its bound. A read while the segment
        // that brought the data is processed is followed by its ACK anyway
        windowUpdate = false;
        if (!m_lxInReceive && outPacket && outPacket->GetSize() > 0)
        {
            // Sequence numbers wrap: the difference is taken modulo 2^32
            auto rcvWindowNow =
                static_cast<int32_t>(m_lxRcvWup.GetValue() + m_lxRcvWnd -
                                     m_tcb->m_rxBuffer->NextRxSequence().GetValue());
            int64_t windowNow = std::max<int32_t>(rcvWindowNow, 0);
            if (2 * windowNow <= m_lxWindowClamp)
            {
                uint32_t newWindow = LinuxSelectWindow();
                windowUpdate = newWindow > 0 && newWindow >= 2 * windowNow;
            }
        }
    }
    if (m_connected && (m_state == ESTABLISHED || m_state == FIN_WAIT_1 || m_state == FIN_WAIT_2) &&
        windowUpdate)
    {
        if (m_tcb->m_ecnState == TcpSocketState::ECN_CE_RCVD ||
            m_tcb->m_ecnState == TcpSocketState::ECN_SENDING_ECE)
        {
            SendEmptyPacket(TcpHeader::ACK | TcpHeader::ECE);
            m_tcb->m_ecnState = TcpSocketState::ECN_SENDING_ECE;
        }
        else
        {
            SendEmptyPacket(TcpHeader::ACK);
        }
    }
    return outPacket;
}

/* Inherit from Socket class: Recv and return the remote's address */
Ptr<Packet>
TcpSocketBase::RecvFrom(uint32_t maxSize, uint32_t flags, Address& fromAddress)
{
    NS_LOG_FUNCTION(this << maxSize << flags);
    Ptr<Packet> packet = Recv(maxSize, flags);
    // Null packet means no data to read, and an empty packet indicates EOF
    if (packet && packet->GetSize() != 0)
    {
        if (m_endPoint != nullptr)
        {
            fromAddress =
                InetSocketAddress(m_endPoint->GetPeerAddress(), m_endPoint->GetPeerPort());
        }
        else if (m_endPoint6 != nullptr)
        {
            fromAddress =
                Inet6SocketAddress(m_endPoint6->GetPeerAddress(), m_endPoint6->GetPeerPort());
        }
        else
        {
            fromAddress = InetSocketAddress(Ipv4Address::GetZero(), 0);
        }
    }
    return packet;
}

/* Inherit from Socket class: Get the max number of bytes an app can send */
uint32_t
TcpSocketBase::GetTxAvailable() const
{
    NS_LOG_FUNCTION(this);
    if (m_lxNoSpace && !IsLinuxWritable())
    {
        return 0;
    }
    return m_txBuffer->Available();
}

uint64_t
TcpSocketBase::GetLinuxSndBufCost() const
{
    // The data held, and the skbs holding it: those not yet sent and those
    // sent and not fully acknowledged
    auto acked =
        std::upper_bound(m_lxRtxSkbEnds.begin(), m_lxRtxSkbEnds.end(), m_txBuffer->HeadSequence());
    uint64_t skbs = (m_lxRtxSkbEnds.end() - acked) + m_lxSkbs.size();
    return m_txBuffer->Size() + skbs * m_sndSkbOverhead;
}

bool
TcpSocketBase::IsLinuxWritable() const
{
    uint64_t cost = GetLinuxSndBufCost();
    uint64_t sndBuf = m_txBuffer->MaxBufferSize();
    return sndBuf > cost && sndBuf - cost >= cost / 2;
}

/* Inherit from Socket class: Get the max number of bytes an app can read */
uint32_t
TcpSocketBase::GetRxAvailable() const
{
    NS_LOG_FUNCTION(this);
    return m_tcb->m_rxBuffer->Available();
}

/* Inherit from Socket class: Return local address:port */
int
TcpSocketBase::GetSockName(Address& address) const
{
    NS_LOG_FUNCTION(this);
    if (m_endPoint != nullptr)
    {
        address = InetSocketAddress(m_endPoint->GetLocalAddress(), m_endPoint->GetLocalPort());
    }
    else if (m_endPoint6 != nullptr)
    {
        address = Inet6SocketAddress(m_endPoint6->GetLocalAddress(), m_endPoint6->GetLocalPort());
    }
    else
    { // It is possible to call this method on a socket without a name
        // in which case, behavior is unspecified
        // Should this return an InetSocketAddress or an Inet6SocketAddress?
        address = InetSocketAddress(Ipv4Address::GetZero(), 0);
    }
    return 0;
}

int
TcpSocketBase::GetPeerName(Address& address) const
{
    NS_LOG_FUNCTION(this << address);

    if (!m_endPoint && !m_endPoint6)
    {
        m_errno = ERROR_NOTCONN;
        return -1;
    }

    if (m_endPoint)
    {
        address = InetSocketAddress(m_endPoint->GetPeerAddress(), m_endPoint->GetPeerPort());
    }
    else if (m_endPoint6)
    {
        address = Inet6SocketAddress(m_endPoint6->GetPeerAddress(), m_endPoint6->GetPeerPort());
    }
    else
    {
        NS_ASSERT(false);
    }

    return 0;
}

/* Inherit from Socket class: Bind this socket to the specified NetDevice */
void
TcpSocketBase::BindToNetDevice(Ptr<NetDevice> netdevice)
{
    NS_LOG_FUNCTION(netdevice);
    Socket::BindToNetDevice(netdevice); // Includes sanity check
    if (m_endPoint != nullptr)
    {
        m_endPoint->BindToNetDevice(netdevice);
    }

    if (m_endPoint6 != nullptr)
    {
        m_endPoint6->BindToNetDevice(netdevice);
    }
}

/* Clean up after Bind. Set up callback functions in the end-point. */
int
TcpSocketBase::SetupCallback()
{
    NS_LOG_FUNCTION(this);

    if (m_endPoint == nullptr && m_endPoint6 == nullptr)
    {
        return -1;
    }
    if (m_endPoint != nullptr)
    {
        m_endPoint->SetRxCallback(
            MakeCallback(&TcpSocketBase::ForwardUp, Ptr<TcpSocketBase>(this)));
        m_endPoint->SetIcmpCallback(
            MakeCallback(&TcpSocketBase::ForwardIcmp, Ptr<TcpSocketBase>(this)));
        m_endPoint->SetDestroyCallback(
            MakeCallback(&TcpSocketBase::Destroy, Ptr<TcpSocketBase>(this)));
    }
    if (m_endPoint6 != nullptr)
    {
        m_endPoint6->SetRxCallback(
            MakeCallback(&TcpSocketBase::ForwardUp6, Ptr<TcpSocketBase>(this)));
        m_endPoint6->SetIcmpCallback(
            MakeCallback(&TcpSocketBase::ForwardIcmp6, Ptr<TcpSocketBase>(this)));
        m_endPoint6->SetDestroyCallback(
            MakeCallback(&TcpSocketBase::Destroy6, Ptr<TcpSocketBase>(this)));
    }

    return 0;
}

/* Perform the real connection tasks: Send SYN if allowed, RST if invalid */
int
TcpSocketBase::DoConnect()
{
    NS_LOG_FUNCTION(this);

    // A new connection is allowed only if this socket does not have a connection
    if (m_state == CLOSED || m_state == LISTEN || m_state == SYN_SENT || m_state == LAST_ACK ||
        m_state == CLOSE_WAIT)
    { // send a SYN packet and change state into SYN_SENT
        // The connection is being opened actively, which has to be told apart
        // from a passive open (RFC 9293, Section 3.5, MUST-11)
        m_activeOpen = true;

        SelectIsn();

        // send a SYN packet with ECE and CWR flags set if sender is ECN capable
        if (m_tcb->m_useEcn == TcpSocketState::On)
        {
            SendEmptyPacket(TcpHeader::SYN | TcpHeader::ECE | TcpHeader::CWR);
        }
        else
        {
            SendEmptyPacket(TcpHeader::SYN);
        }
        NS_LOG_DEBUG(TcpStateName[m_state] << " -> SYN_SENT");
        m_state = SYN_SENT;
        m_tcb->m_ecnState = TcpSocketState::ECN_DISABLED; // because sender is not yet aware about
                                                          // receiver's ECN capability
    }
    else if (m_state != TIME_WAIT)
    { // In states SYN_RCVD, ESTABLISHED, FIN_WAIT_1, FIN_WAIT_2, and CLOSING, an connection
        // exists. We send RST, tear down everything, and close this socket.
        SendRST();
        CloseAndNotify();
    }
    return 0;
}

/* Do the action to close the socket. Usually send a packet with appropriate
    flags depended on the current m_state. */
int
TcpSocketBase::DoClose()
{
    NS_LOG_FUNCTION(this);
    switch (m_state)
    {
    case SYN_RCVD:
    case ESTABLISHED:
        // send FIN to close the peer
        SendEmptyPacket(TcpHeader::FIN);
        NS_LOG_DEBUG("ESTABLISHED -> FIN_WAIT_1");
        m_state = FIN_WAIT_1;
        break;
    case CLOSE_WAIT:
        // send FIN+ACK to close the peer
        SendEmptyPacket(TcpHeader::FIN | TcpHeader::ACK);
        NS_LOG_DEBUG("CLOSE_WAIT -> LAST_ACK");
        m_state = LAST_ACK;
        break;
    case SYN_SENT:
    case CLOSING:
        // Send RST if application closes in SYN_SENT and CLOSING
        SendRST();
        CloseAndNotify();
        break;
    case LISTEN:
        // In this state, move to CLOSED and tear down the end point
        CloseAndNotify();
        break;
    case LAST_ACK:
    case CLOSED:
    case FIN_WAIT_1:
    case FIN_WAIT_2:
    case TIME_WAIT:
    default: /* mute compiler */
        // Do nothing in these five states
        break;
    }
    return 0;
}

/* Peacefully close the socket by notifying the upper layer and deallocate end point */
void
TcpSocketBase::CloseAndNotify()
{
    NS_LOG_FUNCTION(this);

    if (!m_closeNotified)
    {
        NotifyNormalClose();
        m_closeNotified = true;
    }
    if (m_lastAckEvent.IsPending())
    {
        m_lastAckEvent.Cancel();
    }
    NS_LOG_DEBUG(TcpStateName[m_state] << " -> CLOSED");
    m_state = CLOSED;
    DeallocateEndPoint();
}

/* Tell if a sequence number range is out side the range that my rx buffer can
    accept */
bool
TcpSocketBase::OutOfRange(SequenceNumber32 head, SequenceNumber32 tail) const
{
    if (m_state == LISTEN || m_state == SYN_SENT || m_state == SYN_RCVD)
    { // Rx buffer in these states are not initialized.
        return false;
    }
    if (m_state == LAST_ACK || m_state == CLOSING || m_state == CLOSE_WAIT)
    { // In LAST_ACK and CLOSING states, it only wait for an ACK and the
        // sequence number must equals to m_rxBuffer->NextRxSequence ()
        return (m_tcb->m_rxBuffer->NextRxSequence() != head);
    }

    // In all other cases, check if the sequence number is in range
    return (tail < m_tcb->m_rxBuffer->NextRxSequence() ||
            m_tcb->m_rxBuffer->MaxRxSequence() <= head);
}

/* Function called by the L3 protocol when it received a packet to pass on to
    the TCP. This function is registered as the "RxCallback" function in
    SetupCallback(), which invoked by Bind(), and CompleteFork() */
void
TcpSocketBase::ForwardUp(Ptr<Packet> packet,
                         Ipv4Header header,
                         uint16_t port,
                         Ptr<Ipv4Interface> incomingInterface)
{
    NS_LOG_LOGIC("Socket " << this << " forward up " << m_endPoint->GetPeerAddress() << ":"
                           << m_endPoint->GetPeerPort() << " to " << m_endPoint->GetLocalAddress()
                           << ":" << m_endPoint->GetLocalPort());

    Address fromAddress = InetSocketAddress(header.GetSource(), port);
    Address toAddress = InetSocketAddress(header.GetDestination(), m_endPoint->GetLocalPort());

    if (m_linuxRcvWindow)
    {
        // Each segment received, even dropped as unacceptable, takes a
        // receive buffer of the Linux driver
        ++m_lxRxPackets;
    }

    TcpHeader tcpHeader;
    uint32_t bytesRemoved = packet->PeekHeader(tcpHeader);

    if (!IsValidTcpSegment(tcpHeader, bytesRemoved, packet->GetSize() - bytesRemoved))
    {
        ProcessUrgentOfDroppedSegment(tcpHeader, packet->GetSize() - bytesRemoved);
        return;
    }

    if (tcpHeader.IsMalformed() && !(tcpHeader.GetFlags() & TcpHeader::RST))
    {
        // Only an acceptable segment resets the connection; a reset needs no
        // option to be processed
        AbortOnMalformedSegment(tcpHeader,
                                packet->GetSize() - bytesRemoved,
                                fromAddress,
                                toAddress);
        return;
    }

    if (header.GetEcn() == Ipv4Header::ECN_CE && m_ecnCESeq < tcpHeader.GetSequenceNumber())
    {
        NS_LOG_INFO("Received CE flag is valid");
        if (m_tcb->m_ecnState != TcpSocketState::ECN_DISABLED &&
            m_tcb->m_ecnState != TcpSocketState::ECN_CE_RCVD &&
            m_tcb->m_ecnState != TcpSocketState::ECN_SENDING_ECE)
        {
            // The sender may have a very small window: as Linux, do not delay
            // the ACKs until it reduces it
            EnterQuickAckMode(CE_QUICK_ACKS);
        }
        NS_LOG_DEBUG(TcpSocketState::EcnStateName[m_tcb->m_ecnState] << " -> ECN_CE_RCVD");
        m_ecnCESeq = tcpHeader.GetSequenceNumber();
        m_tcb->m_ecnState = TcpSocketState::ECN_CE_RCVD;
        m_congestionControl->CwndEvent(m_tcb, TcpSocketState::CA_EVENT_ECN_IS_CE);
    }
    else if (header.GetEcn() != Ipv4Header::ECN_NotECT &&
             m_tcb->m_ecnState != TcpSocketState::ECN_DISABLED)
    {
        m_congestionControl->CwndEvent(m_tcb, TcpSocketState::CA_EVENT_ECN_NO_CE);
    }

    DoForwardUp(packet, fromAddress, toAddress);
}

void
TcpSocketBase::ForwardUp6(Ptr<Packet> packet,
                          Ipv6Header header,
                          uint16_t port,
                          Ptr<Ipv6Interface> incomingInterface)
{
    NS_LOG_LOGIC("Socket " << this << " forward up " << m_endPoint6->GetPeerAddress() << ":"
                           << m_endPoint6->GetPeerPort() << " to " << m_endPoint6->GetLocalAddress()
                           << ":" << m_endPoint6->GetLocalPort());

    Address fromAddress = Inet6SocketAddress(header.GetSource(), port);
    Address toAddress = Inet6SocketAddress(header.GetDestination(), m_endPoint6->GetLocalPort());

    if (m_linuxRcvWindow)
    {
        // Each segment received, even dropped as unacceptable, takes a
        // receive buffer of the Linux driver
        ++m_lxRxPackets;
    }

    TcpHeader tcpHeader;
    uint32_t bytesRemoved = packet->PeekHeader(tcpHeader);

    if (!IsValidTcpSegment(tcpHeader, bytesRemoved, packet->GetSize() - bytesRemoved))
    {
        ProcessUrgentOfDroppedSegment(tcpHeader, packet->GetSize() - bytesRemoved);
        return;
    }

    if (tcpHeader.IsMalformed() && !(tcpHeader.GetFlags() & TcpHeader::RST))
    {
        // Only an acceptable segment resets the connection; a reset needs no
        // option to be processed
        AbortOnMalformedSegment(tcpHeader,
                                packet->GetSize() - bytesRemoved,
                                fromAddress,
                                toAddress);
        return;
    }

    if (header.GetEcn() == Ipv6Header::ECN_CE && m_ecnCESeq < tcpHeader.GetSequenceNumber())
    {
        NS_LOG_INFO("Received CE flag is valid");
        if (m_tcb->m_ecnState != TcpSocketState::ECN_DISABLED &&
            m_tcb->m_ecnState != TcpSocketState::ECN_CE_RCVD &&
            m_tcb->m_ecnState != TcpSocketState::ECN_SENDING_ECE)
        {
            // The sender may have a very small window: as Linux, do not delay
            // the ACKs until it reduces it
            EnterQuickAckMode(CE_QUICK_ACKS);
        }
        NS_LOG_DEBUG(TcpSocketState::EcnStateName[m_tcb->m_ecnState] << " -> ECN_CE_RCVD");
        m_ecnCESeq = tcpHeader.GetSequenceNumber();
        m_tcb->m_ecnState = TcpSocketState::ECN_CE_RCVD;
        m_congestionControl->CwndEvent(m_tcb, TcpSocketState::CA_EVENT_ECN_IS_CE);
    }
    else if (header.GetEcn() != Ipv6Header::ECN_NotECT &&
             m_tcb->m_ecnState != TcpSocketState::ECN_DISABLED)
    {
        m_congestionControl->CwndEvent(m_tcb, TcpSocketState::CA_EVENT_ECN_NO_CE);
    }

    DoForwardUp(packet, fromAddress, toAddress);
}

void
TcpSocketBase::ForwardIcmp(Ipv4Address icmpSource,
                           uint8_t icmpTtl,
                           uint8_t icmpType,
                           uint8_t icmpCode,
                           uint32_t icmpInfo)
{
    NS_LOG_FUNCTION(this << icmpSource << static_cast<uint32_t>(icmpTtl)
                         << static_cast<uint32_t>(icmpType) << static_cast<uint32_t>(icmpCode)
                         << icmpInfo);
    if (!m_icmpCallback.IsNull())
    {
        m_icmpCallback(icmpSource, icmpTtl, icmpType, icmpCode, icmpInfo);
    }
}

void
TcpSocketBase::ForwardIcmp6(Ipv6Address icmpSource,
                            uint8_t icmpTtl,
                            uint8_t icmpType,
                            uint8_t icmpCode,
                            uint32_t icmpInfo)
{
    NS_LOG_FUNCTION(this << icmpSource << static_cast<uint32_t>(icmpTtl)
                         << static_cast<uint32_t>(icmpType) << static_cast<uint32_t>(icmpCode)
                         << icmpInfo);
    if (!m_icmpCallback6.IsNull())
    {
        m_icmpCallback6(icmpSource, icmpTtl, icmpType, icmpCode, icmpInfo);
    }
}

void
TcpSocketBase::NotifyReset()
{
    NS_LOG_FUNCTION(this);
    if (m_closeNotified)
    {
        return;
    }
    // A connection closed by a RST was aborted, and the application is told
    // apart from a normal close (RFC 9293, Section 3.6, MUST-12). The
    // connection is CLOSED before the application hears of it, so that a
    // Close() from the callback neither resets nor notifies again.
    m_errno = ERROR_NOTCONN;
    m_closeNotified = true;
    NS_LOG_DEBUG(TcpStateName[m_state] << " -> CLOSED");
    m_state = CLOSED;
    NotifyErrorClose();
}

void
TcpSocketBase::AbortOnMalformedSegment(const TcpHeader& tcpHeader,
                                       uint32_t payloadSize,
                                       const Address& fromAddress,
                                       const Address& toAddress)
{
    NS_LOG_FUNCTION(this << tcpHeader << payloadSize << fromAddress << toAddress);
    NS_ASSERT(!(tcpHeader.GetFlags() & TcpHeader::RST));

    if (m_state == LISTEN || m_state == CLOSED || m_state == SYN_SENT)
    {
        bool acksSyn = (tcpHeader.GetFlags() & TcpHeader::ACK) &&
                       tcpHeader.GetAckNumber() == m_tcb->m_nextTxSequence + SequenceNumber32(1);
        if (m_state == SYN_SENT && (tcpHeader.GetFlags() & TcpHeader::ACK) && !acksSyn)
        {
            // Not acceptable in SYN-SENT (RFC 9293, Section 3.10.7.3)
            NS_LOG_LOGIC("Dropping a malformed segment which does not acknowledge the SYN");
            return;
        }

        // There is no synchronized connection to take the sequence numbers
        // of the reset from: they are built from the segment, as for a
        // closed port (RFC 9293, Section 3.10.7.1)
        NS_LOG_WARN("Resetting the connection upon a segment with malformed options");
        TcpHeader rst;
        if (tcpHeader.GetFlags() & TcpHeader::ACK)
        {
            rst.SetFlags(TcpHeader::RST);
            rst.SetSequenceNumber(tcpHeader.GetAckNumber());
        }
        else
        {
            uint32_t length = payloadSize;
            length += (tcpHeader.GetFlags() & TcpHeader::SYN) ? 1 : 0;
            length += (tcpHeader.GetFlags() & TcpHeader::FIN) ? 1 : 0;
            rst.SetFlags(TcpHeader::RST | TcpHeader::ACK);
            rst.SetSequenceNumber(SequenceNumber32(0));
            rst.SetAckNumber(tcpHeader.GetSequenceNumber() + SequenceNumber32(length));
        }
        rst.SetSourcePort(tcpHeader.GetDestinationPort());
        rst.SetDestinationPort(tcpHeader.GetSourcePort());
        Ptr<Packet> p = Create<Packet>();
        m_txTrace(p, rst, this);
        m_tcp->SendPacket(p, rst, toAddress, fromAddress, m_boundnetdevice);

        if (m_state == SYN_SENT)
        {
            // The application which called Connect() is told through the
            // connection failure callback
            m_errno = ERROR_NOTCONN;
            NotifyConnectionFailed();
            m_closeNotified = true;
            CloseAndNotify();
        }
        return;
    }

    // The segment passed the acceptability test, so that the reset, which
    // starts at SND.NXT, lies in the window of the peer
    NS_LOG_WARN("Aborting the connection upon a segment with malformed options");
    SendEmptyPacket(TcpHeader::RST);
    m_errno = ERROR_NOTCONN;
    if (m_state == SYN_RCVD)
    {
        if (m_activeOpen)
        {
            NotifyConnectionFailed();
        }
        // A connection opened passively was not handed to the application
        // yet: there is nobody to notify
    }
    else if (!m_closeNotified)
    {
        NotifyErrorClose();
    }
    m_closeNotified = true;
    CloseAndNotify();
}

void
TcpSocketBase::SetTxHeadSequence(SequenceNumber32 seq)
{
    NS_LOG_FUNCTION(this << seq);
    if (m_sndUrgentArmed)
    {
        // The urgent data queued before the handshake moves with the data
        m_sndUrgentPoint = seq + (m_sndUrgentPoint - m_txBuffer->HeadSequence());
    }
    m_txBuffer->SetHeadSequence(seq);
}

uint32_t
TcpSocketBase::GetSegmentOptionsSize(bool timestamp)
{
    if (!timestamp)
    {
        return 0;
    }
    // 10 bytes of option, padded to a 4 byte boundary
    return ((CreateObject<TcpOptionTS>()->GetSerializedSize() + 3) / 4) * 4;
}

bool
TcpSocketBase::IsValidTcpSegment(const TcpHeader& tcpHeader,
                                 const uint32_t tcpHeaderSize,
                                 const uint32_t tcpPayloadSize)
{
    SequenceNumber32 seq = tcpHeader.GetSequenceNumber();
    uint8_t flags = tcpHeader.GetFlags();
    if (tcpHeaderSize == 0 || tcpHeaderSize > 60)
    {
        NS_LOG_ERROR("Bytes removed: " << tcpHeaderSize << " invalid");
        return false; // Discard invalid packet
    }

    if ((flags & TcpHeader::RST) && m_state != CLOSED && m_state != LISTEN && m_state != SYN_SENT)
    {
        // A reset is accepted only at RCV.NXT: anywhere else in the window,
        // it is answered with a challenge ACK, and outside of it, dropped
        // silently (RFC 9293, Section 3.10.7.4, and RFC 5961, Section 3.2),
        // so that a blind attacker has to guess the exact sequence number
        if (seq == m_tcb->m_rxBuffer->NextRxSequence())
        {
            return true;
        }
        if (!ZeroLengthOutOfRange(seq))
        {
            NS_LOG_WARN("At state " << TcpStateName[m_state] << " received a RST of seq " << seq
                                    << " instead of " << m_tcb->m_rxBuffer->NextRxSequence()
                                    << "; sending a challenge ACK");
            SendEmptyPacket(TcpHeader::ACK);
        }
        else
        {
            NS_LOG_WARN("At state " << TcpStateName[m_state] << " received a RST of seq " << seq
                                    << " out of the window; dropping it");
        }
        return false;
    }

    if ((flags & TcpHeader::SYN) && m_state != CLOSED && m_state != LISTEN && m_state != SYN_SENT &&
        m_state != SYN_RCVD)
    {
        // A SYN reaching a synchronized connection, whatever its sequence
        // number, is answered with a challenge ACK and dropped without
        // touching the state of the connection (RFC 9293, Section 3.10.7.4,
        // and RFC 5961, Section 4.2): a retransmitted SYN+ACK must neither
        // renegotiate the options nor reset the congestion window
        NS_LOG_WARN("At state " << TcpStateName[m_state]
                                << " received a SYN; sending a challenge ACK");
        SendEmptyPacket(TcpHeader::ACK);
        return false;
    }

    if (tcpPayloadSize > 0 && OutOfRange(seq, seq + tcpPayloadSize))
    {
        // Discard fully out of range data packets
        NS_LOG_WARN("At state " << TcpStateName[m_state] << " received packet of seq [" << seq
                                << ":" << seq + tcpPayloadSize << ") out of range ["
                                << m_tcb->m_rxBuffer->NextRxSequence() << ":"
                                << m_tcb->m_rxBuffer->MaxRxSequence() << ")");
        if (seq + tcpPayloadSize <= m_tcb->m_rxBuffer->NextRxSequence())
        {
            // The ACK of the data was lost, or the data was retransmitted
            // spuriously: as Linux, stop delaying the ACKs
            EnterQuickAckMode(MAX_QUICK_ACKS);
        }
        // Acknowledgement should be sent for all unacceptable packets (RFC793, p.69)
        SendEmptyPacket(TcpHeader::ACK);
        return false;
    }

    if (tcpPayloadSize == 0 && !(flags & TcpHeader::SYN) && ZeroLengthOutOfRange(seq))
    {
        if ((flags & TcpHeader::ACK) && !(flags & TcpHeader::FIN) &&
            seq < m_tcb->m_rxBuffer->NextRxSequence() && AcksNewData(tcpHeader.GetAckNumber()))
        {
            // An old sequence number does not invalidate the acknowledgment
            // of new data: dropping it would stall a peer whose pure ACKs
            // still carry the sequence number its retransmissions rewound
            // to. Linux accepts these from its own window edge, RCV.WUP.
            NS_LOG_LOGIC("Processing the ACK of an empty segment of old seq " << seq);
            return true;
        }
        // A segment without data lying outside the window, such as a
        // keep-alive probe repeating an acknowledged sequence number, or a
        // retransmitted FIN, is answered with an acknowledgment and dropped
        // (RFC 9293, Section 3.10.7.4)
        NS_LOG_WARN("At state " << TcpStateName[m_state] << " received an empty segment of seq "
                                << seq << " out of range [" << m_tcb->m_rxBuffer->NextRxSequence()
                                << ":" << m_tcb->m_rxBuffer->MaxRxSequence() << ")");
        SendEmptyPacket(TcpHeader::ACK);
        if (m_state == TIME_WAIT && (flags & TcpHeader::FIN))
        {
            // The FIN of the peer is retransmitted, so the ACK of it was
            // lost: the 2 MSL timeout restarts (RFC 9293, Section 3.10.7.4)
            NS_LOG_LOGIC("Restarting the TIME-WAIT timeout upon a retransmitted FIN");
            m_timewaitEvent.Cancel();
            m_timewaitEvent =
                Simulator::Schedule(Seconds(2 * m_msl), &TcpSocketBase::CloseAndNotify, this);
        }
        return false;
    }
    return true;
}

bool
TcpSocketBase::AcksNewData(SequenceNumber32 ack) const
{
    switch (m_state)
    {
    case SYN_RCVD:
    case ESTABLISHED:
    case CLOSE_WAIT:
        return m_txBuffer->HeadSequence() < ack && ack <= m_tcb->m_highTxMark;
    case FIN_WAIT_1:
    case CLOSING:
    case LAST_ACK:
        // The FIN takes the sequence number following the data
        return m_txBuffer->HeadSequence() < ack && ack <= m_tcb->m_highTxMark + SequenceNumber32(1);
    default:
        return false;
    }
}

bool
TcpSocketBase::ZeroLengthOutOfRange(SequenceNumber32 seq) const
{
    SequenceNumber32 next = m_tcb->m_rxBuffer->NextRxSequence();
    switch (m_state)
    {
    case SYN_RCVD:
    case ESTABLISHED:
    case FIN_WAIT_1:
    case FIN_WAIT_2:
        break;
    case CLOSE_WAIT:
    case CLOSING:
    case LAST_ACK:
    case TIME_WAIT:
        // The FIN of the peer took the last sequence number it may use, so
        // that RCV.NXT is the only acceptable one; the receive buffer no
        // longer accounts for a window beyond it
        return seq != next;
    default:
        // Before the connection is synchronized, the receive buffer is not
        // initialized
        return false;
    }

    SequenceNumber32 max = m_tcb->m_rxBuffer->MaxRxSequence();
    if (m_tcb->m_rxBuffer->GotFin())
    {
        // The FIN of the peer arrived out of order, and MaxRxSequence() is
        // its sequence number: the segments the peer sends after its FIN,
        // such as its ACKs, carry the next one
        return seq < next || max + SequenceNumber32(1) < seq;
    }
    if (next == max)
    {
        // The window is closed: only a segment at RCV.NXT is acceptable
        return seq != next;
    }
    return seq < next || max <= seq;
}

void
TcpSocketBase::ProcessSynOptions(const TcpHeader& tcpHeader)
{
    NS_LOG_FUNCTION(this << tcpHeader);

    /* The window field in a segment where the SYN bit is set (i.e., a <SYN>
     * or <SYN,ACK>) MUST NOT be scaled (from RFC 7323 page 9). But should be
     * saved anyway..
     */
    m_rWnd = tcpHeader.GetWindowSize();

    if (tcpHeader.HasOption(TcpOption::WINSCALE) && m_winScalingEnabled)
    {
        ProcessOptionWScale(tcpHeader.GetOption(TcpOption::WINSCALE));
    }
    else
    {
        m_winScalingEnabled = false;
    }

    if (tcpHeader.HasOption(TcpOption::SACKPERMITTED) && m_sackEnabled)
    {
        ProcessOptionSackPermitted(tcpHeader.GetOption(TcpOption::SACKPERMITTED));
    }
    else
    {
        m_sackEnabled = false;
        m_txBuffer->SetSackEnabled(false);
    }
    // Mirror the negotiated SACK state into the shared TcpSocketState so that
    // recovery algorithms (e.g. PRR) can distinguish SACK from non-SACK
    // DeliveredData accounting.
    m_tcb->m_sackEnabled = m_sackEnabled;

    if (m_advertisedMss == 0)
    {
        // Save the configured segment size before it is reduced below: the
        // MSS option falls back to it when the MTU of the interface is
        // unknown
        m_advertisedMss = m_tcb->m_segmentSize;
    }

    if (tcpHeader.HasOption(TcpOption::MSS))
    {
        ProcessOptionMss(tcpHeader.GetOption(TcpOption::MSS));
    }
    else
    {
        // No MSS option received: assume the default maximum segment size
        // of 536 bytes for IPv4 and 1220 bytes for IPv6 (RFC 9293,
        // Section 3.7.1, MUST-15). An IPv6 socket talking to a v4-mapped
        // address runs over IPv4.
        bool overIpv4 =
            m_endPoint != nullptr ||
            (m_endPoint6 != nullptr && m_endPoint6->GetPeerAddress().IsIpv4MappedAddress());
        uint32_t defaultMss = overIpv4 ? 536 : 1220;
        m_tcb->m_segmentSize = std::min(m_tcb->m_segmentSize, defaultMss);
    }
    // The effective send MSS is bounded by what the local interface carries
    // whole as well as by what the peer receives (RFC 9293, Section 3.7.1,
    // MUST-16)
    m_tcb->m_segmentSize = std::min(m_tcb->m_segmentSize, GetInterfaceMss());
    if (!m_ipOptionsAccounted)
    {
        m_ipOptionsAccounted = true;
        UpdateSegmentSizeForIpOptions();
    }

    // When receiving a <SYN> or <SYN-ACK> we should adapt TS to the other end
    if (tcpHeader.HasOption(TcpOption::TS) && m_timestampEnabled)
    {
        Ptr<const TcpOption> ts = tcpHeader.GetOption(TcpOption::TS);
        ProcessOptionTimestamp(ts, tcpHeader.GetSequenceNumber());
        // The timestamp of the opening segment is always the one to echo,
        // whatever its sequence number (RFC 7323, Section 3.2)
        m_timestampToEcho = DynamicCast<const TcpOptionTS>(ts)->GetTimestamp();
    }
    else
    {
        m_timestampEnabled = false;
    }

    if (m_timestampEnabled && !m_segmentSizeAdjusted)
    {
        // The MSS counts only data octets, it does not count the TCP
        // header or the TCP options, so the sender must reduce the TCP data
        // length to account for the options it includes (RFC 6691, Section
        // 2): decrease the segment size by the size of the timestamp
        // option (and its padding to a word), which is carried by every
        // segment. The segment size may have been clamped by the MSS the
        // peer advertised, so a tiny value cannot bring it to zero.
        const uint32_t tsOptionSize = GetSegmentOptionsSize(true);
        if (m_tcb->m_segmentSize > tsOptionSize)
        {
            m_tcb->m_segmentSize -= tsOptionSize;
        }
        else
        {
            m_tcb->m_segmentSize = 1;
        }
        m_segmentSizeAdjusted = true;
        NS_LOG_INFO("Decreased the segment size to " << m_tcb->m_segmentSize
                                                     << " to accommodate the TCP options");
    }

    // Initialize cWnd and ssThresh
    m_tcb->m_cWnd = GetInitialCwnd() * GetSegSize();
    m_tcb->m_cWndInfl = m_tcb->m_cWnd;
    m_tcb->m_ssThresh = GetInitialSSThresh();

    // The transmission buffer sizes the segments it hands out, and tells a
    // loss apart from reordering, in units of the negotiated segment size
    m_txBuffer->SetSegmentSize(m_tcb->m_segmentSize);
}

void
TcpSocketBase::DoForwardUp(Ptr<Packet> packet, const Address& fromAddress, const Address& toAddress)
{
    // in case the packet still has a priority tag attached, remove it
    SocketPriorityTag priorityTag;
    packet->RemovePacketTag(priorityTag);

    // Peel off TCP header
    TcpHeader tcpHeader;
    packet->RemoveHeader(tcpHeader);
    SequenceNumber32 seq = tcpHeader.GetSequenceNumber();

    if (m_state == ESTABLISHED && !(tcpHeader.GetFlags() & TcpHeader::RST))
    {
        // Check if the sender has responded to ECN echo by reducing the Congestion Window
        if (tcpHeader.GetFlags() & TcpHeader::CWR)
        {
            // Check if a packet with CE bit set is received. If there is no CE bit set, then change
            // the state to ECN_IDLE to stop sending ECN Echo messages. If there is CE bit set, the
            // packet should continue sending ECN Echo messages
            //
            if (m_tcb->m_ecnState != TcpSocketState::ECN_CE_RCVD)
            {
                NS_LOG_DEBUG(TcpSocketState::EcnStateName[m_tcb->m_ecnState] << " -> ECN_IDLE");
                m_tcb->m_ecnState = TcpSocketState::ECN_IDLE;
            }
            // As tcp_ecn_accept_cwr() of Linux, data with CWR is acknowledged
            // at once (ICSK_ACK_NOW), since the window of the sender may be
            // very small
            if (m_quickAck && packet->GetSize() > 0)
            {
                m_ackNow = true;
            }
        }
    }

    m_rxTrace(packet, tcpHeader, this);

    if ((tcpHeader.GetFlags() & TcpHeader::RST) && m_state != SYN_SENT)
    {
        // A reset which got through the acceptability test aborts the
        // connection whatever the other bits it carries, which the state
        // functions would otherwise take for an illegal combination to
        // answer with a reset of their own. In SYN-SENT, its ACK bit tells
        // whether it is acceptable (RFC 9293, Section 3.10.7.3).
        tcpHeader.SetFlags(TcpHeader::RST);
    }

    Ipv4ReturnRouteTag returnRoute;
    bool hasReturnRoute = packet->RemovePacketTag(returnRoute);

    KeepAliveSegmentReceived();
    TcpStates_t stateBefore = m_state;

    if ((tcpHeader.GetFlags() & TcpHeader::URG) && !(tcpHeader.GetFlags() & TcpHeader::RST) &&
        (m_state == ESTABLISHED || m_state == FIN_WAIT_1 || m_state == FIN_WAIT_2))
    {
        // The urgent field is processed for every incoming segment of a
        // synchronized connection which lies in the window (RFC 9293,
        // Section 3.10.7.4), which ForwardUp checked already; with a zero
        // receive window, ProcessUrgentOfDroppedSegment takes over (Section
        // 3.4, MUST-66)
        ProcessUrgentPointer(tcpHeader);
    }

    if (tcpHeader.GetFlags() & TcpHeader::SYN)
    {
        /* The window field in a segment where the SYN bit is set (i.e., a <SYN>
         * or <SYN,ACK>) MUST NOT be scaled (from RFC 7323 page 9). But should be
         * saved anyway..
         */
        m_rWnd = tcpHeader.GetWindowSize();

        if (m_state == SYN_SENT)
        {
            // The options are negotiated by the first SYN received. A
            // listening socket is left untouched by the SYNs which reach it:
            // the options of each one are negotiated by the socket forked for
            // its connection (see CompleteFork), so that neither the listener
            // nor the connections accepted later inherit them. In SYN-RECEIVED,
            // a retransmitted SYN, or the SYN+ACK completing a simultaneous
            // open, repeats a SYN whose options were processed already.
            ProcessSynOptions(tcpHeader);
        }

        if (tcpHeader.GetFlags() & TcpHeader::ACK)
        {
            EstimateRtt(tcpHeader);
            m_highRxAckMark = tcpHeader.GetAckNumber();
        }
    }
    else if ((tcpHeader.GetFlags() & TcpHeader::ACK) && !(tcpHeader.GetFlags() & TcpHeader::RST))
    {
        NS_ASSERT(!(tcpHeader.GetFlags() & TcpHeader::SYN));
        if (m_timestampEnabled)
        {
            if (!tcpHeader.HasOption(TcpOption::TS) && !(tcpHeader.GetFlags() & TcpHeader::RST))
            {
                // Ignoring segment without TS, RFC 7323. A reset is processed
                // without it (RFC 7323, Section 3.2): the peer which lost the
                // connection state sends it with no option at all
                NS_LOG_LOGIC("At state " << TcpStateName[m_state] << " received packet of seq ["
                                         << seq << ":" << seq + packet->GetSize()
                                         << ") without TS option. Silently discard it");
                return;
            }
            else if (tcpHeader.HasOption(TcpOption::TS))
            {
                ProcessOptionTimestamp(tcpHeader.GetOption(TcpOption::TS),
                                       tcpHeader.GetSequenceNumber());
            }
        }

        EstimateRtt(tcpHeader);
        UpdateWindowSize(tcpHeader);
    }

    // A socket in SYN-SENT learns the window of the peer from its SYN only,
    // and a reset carries no window to act upon
    bool windowKnown = !(tcpHeader.GetFlags() & TcpHeader::RST) &&
                       (m_state != SYN_SENT || (tcpHeader.GetFlags() & TcpHeader::SYN));
    if (m_rWnd.Get() == 0 && m_persistEvent.IsExpired() && m_state != LISTEN && windowKnown)
    { // Zero window: Enter persist state to send 1 byte to probe. A listening
      // socket has no connection to probe: the SYNs it receives are answered
      // by the sockets forked for them
        NS_LOG_LOGIC(this << " Enter zerowindow persist state");
        NS_LOG_LOGIC(
            this << " Cancelled ReTxTimeout event which was set to expire at "
                 << (Simulator::Now() + Simulator::GetDelayLeft(m_retxEvent)).GetSeconds());
        m_retxEvent.Cancel();
        NS_LOG_LOGIC("Schedule persist timeout at time "
                     << Simulator::Now().GetSeconds() << " to expire at time "
                     << (Simulator::Now() + m_persistTimeout).GetSeconds());
        m_persistEvent =
            Simulator::Schedule(m_persistTimeout, &TcpSocketBase::PersistTimeout, this);
        NS_ASSERT(m_persistTimeout == Simulator::GetDelayLeft(m_persistEvent));
    }

    if (hasReturnRoute && !m_appSourceRoute && m_endPoint != nullptr &&
        IsAcceptableForReturnRoute(tcpHeader, packet->GetSize()))
    {
        // The way back recorded by the datagram which arrived, which the
        // segments of this connection follow unless the application asked for
        // a route of its own (RFC 9293, Section 3.9.2.1, MUST-52 and MUST-53).
        // A listener keeps it only for the connection it forks for the SYN
        // (see ProcessListen).
        m_sourceRoute = returnRoute.GetRoute();
        NS_LOG_LOGIC("Saved a return route of " << m_sourceRoute.size() << " hops");
        UpdateSegmentSizeForIpOptions();
    }

    // TCP state machine code in different process functions
    // C.f.: tcp_rcv_state_process() in tcp_input.c in Linux kernel
    switch (m_state)
    {
    case ESTABLISHED:
        ProcessEstablished(packet, tcpHeader);
        break;
    case LISTEN:
        ProcessListen(packet, tcpHeader, fromAddress, toAddress);
        break;
    case TIME_WAIT:
        // Do nothing
        break;
    case CLOSED:
        // Send RST if the incoming packet is not a RST
        if ((tcpHeader.GetFlags() & ~(TcpHeader::PSH | TcpHeader::URG)) != TcpHeader::RST)
        { // Since m_endPoint is not configured yet, we cannot use SendRST here
            TcpHeader h;
            Ptr<Packet> p = Create<Packet>();
            h.SetFlags(TcpHeader::RST);
            h.SetSequenceNumber(m_tcb->m_nextTxSequence);
            h.SetAckNumber(m_tcb->m_rxBuffer->NextRxSequence());
            h.SetSourcePort(tcpHeader.GetDestinationPort());
            h.SetDestinationPort(tcpHeader.GetSourcePort());
            h.SetWindowSize(AdvertisedWindowSize());
            AddOptions(h);
            // The route of the application, else the way back of the segment
            Ipv4SourceRouteTag routeTag;
            routeTag.SetRoute(m_appSourceRoute && m_endPoint != nullptr ? m_sourceRoute
                                                                        : returnRoute.GetRoute());
            if (!routeTag.GetRoute().empty())
            {
                p->AddPacketTag(routeTag);
            }
            m_txTrace(p, h, this);
            m_tcp->SendPacket(p, h, toAddress, fromAddress, m_boundnetdevice);
        }
        break;
    case SYN_SENT:
        ProcessSynSent(packet, tcpHeader);
        break;
    case SYN_RCVD:
        ProcessSynRcvd(packet, tcpHeader, fromAddress, toAddress);
        break;
    case FIN_WAIT_1:
    case FIN_WAIT_2:
    case CLOSE_WAIT:
        ProcessWait(packet, tcpHeader);
        break;
    case CLOSING:
        ProcessClosing(packet, tcpHeader);
        break;
    case LAST_ACK:
        ProcessLastAck(packet, tcpHeader);
        break;
    default: // mute compiler
        break;
    }

    if (m_state != stateBefore)
    {
        // The keep-alives start, or stop, with the state the segment led to
        RearmKeepAlive();
    }

    if (m_rWnd.Get() != 0 && m_persistEvent.IsPending())
    { // persist probes end, the other end has increased the window
        NS_ASSERT(m_connected);
        NS_LOG_LOGIC(this << " Leaving zerowindow persist state");
        m_persistEvent.Cancel();

        SendPendingData(m_connected);
    }
}

/* Received a packet upon ESTABLISHED state. This function is mimicking the
    role of tcp_rcv_established() in tcp_input.c in Linux kernel. */
void
TcpSocketBase::ProcessEstablished(Ptr<Packet> packet, const TcpHeader& tcpHeader)
{
    NS_LOG_FUNCTION(this << tcpHeader);

    // Extract the flags. PSH, URG, CWR and ECE are disregarded.
    uint8_t tcpflags =
        tcpHeader.GetFlags() & ~(TcpHeader::PSH | TcpHeader::URG | TcpHeader::CWR | TcpHeader::ECE);

    // Different flags are different events
    if (tcpflags == TcpHeader::ACK)
    {
        if (tcpHeader.GetAckNumber() < m_txBuffer->HeadSequence())
        {
            // Case 1:  If the ACK is a duplicate (SEG.ACK < SND.UNA), it can be ignored.
            // Pag. 72 RFC 793
            NS_LOG_WARN("Ignored ack of " << tcpHeader.GetAckNumber()
                                          << " SND.UNA = " << m_txBuffer->HeadSequence());

            // TODO: RFC 5961 5.2 [Blind Data Injection Attack].[Mitigation]
        }
        else if (tcpHeader.GetAckNumber() > m_tcb->m_highTxMark)
        {
            // If the ACK acks something not yet sent (SEG.ACK > HighTxMark) then
            // send an ACK, drop the segment, and return.
            // Pag. 72 RFC 793
            NS_LOG_WARN("Ignored ack of " << tcpHeader.GetAckNumber()
                                          << " HighTxMark = " << m_tcb->m_highTxMark);

            // Receiver sets ECE flags when it receives a packet with CE bit on or sender hasn't
            // responded to ECN echo sent by receiver
            if (m_tcb->m_ecnState == TcpSocketState::ECN_CE_RCVD ||
                m_tcb->m_ecnState == TcpSocketState::ECN_SENDING_ECE)
            {
                SendEmptyPacket(TcpHeader::ACK | TcpHeader::ECE);
                NS_LOG_DEBUG(TcpSocketState::EcnStateName[m_tcb->m_ecnState]
                             << " -> ECN_SENDING_ECE");
                m_tcb->m_ecnState = TcpSocketState::ECN_SENDING_ECE;
            }
            else
            {
                SendEmptyPacket(TcpHeader::ACK);
            }
        }
        else
        {
            // SND.UNA < SEG.ACK =< HighTxMark
            // Pag. 72 RFC 793
            ReceivedAck(packet, tcpHeader);
        }
    }
    else if (tcpflags == TcpHeader::SYN || tcpflags == (TcpHeader::SYN | TcpHeader::ACK))
    {
        // (a) Received SYN, old NS-3 behaviour is to set state to SYN_RCVD and
        // respond with a SYN+ACK. But it is not a legal state transition as of
        // RFC793. Thus this is ignored.

        // (b) No action for received SYN+ACK, it is probably a duplicated packet
    }
    else if (tcpflags == TcpHeader::FIN || tcpflags == (TcpHeader::FIN | TcpHeader::ACK))
    { // Received FIN or FIN+ACK, bring down this socket nicely
        PeerClose(packet, tcpHeader);
    }
    else if (tcpflags == 0)
    { // No flags means there is only data
        ReceivedData(packet, tcpHeader);
        if (m_tcb->m_rxBuffer->Finished())
        {
            PeerClose(packet, tcpHeader);
        }
    }
    else
    { // Received RST or the TCP flags is invalid, in either case, terminate this socket
        if (tcpflags != TcpHeader::RST)
        { // this must be an invalid flag, send reset
            NS_LOG_LOGIC("Illegal flag " << TcpHeader::FlagsToString(tcpflags)
                                         << " received. Reset packet is sent.");
            SendRST();
        }
        else
        {
            NotifyReset();
        }
        CloseAndNotify();
    }
}

bool
TcpSocketBase::IsTcpOptionEnabled(uint8_t kind) const
{
    NS_LOG_FUNCTION(this << static_cast<uint32_t>(kind));

    switch (kind)
    {
    case TcpOption::TS:
        return m_timestampEnabled;
    case TcpOption::WINSCALE:
        return m_winScalingEnabled;
    case TcpOption::SACKPERMITTED:
    case TcpOption::SACK:
        return m_sackEnabled;
    default:
        break;
    }
    return false;
}

void
TcpSocketBase::ReadOptions(const TcpHeader& tcpHeader, uint32_t* bytesSacked)
{
    NS_LOG_FUNCTION(this << tcpHeader);

    for (const auto& option : tcpHeader.GetOptionList())
    {
        // Check only for ACK options here
        switch (option->GetKind())
        {
        case TcpOption::SACK:
            *bytesSacked = ProcessOptionSack(option);
            break;
        default:
            continue;
        }
    }
}

// Sender should reduce the Congestion Window as a response to receiver's
// ECN Echo notification only once per window
void
TcpSocketBase::EnterCwr(uint32_t currentDelivered)
{
    NS_LOG_FUNCTION(this << currentDelivered);
    // RFC 3168, Section 6.1.2, with the FlightSize of RFC 5681 (equation 4)
    m_tcb->m_ssThresh = m_congestionControl->GetSsThresh(m_tcb, CongestionFlightSize());
    NS_LOG_DEBUG("Reduce ssThresh to " << m_tcb->m_ssThresh);
    QueueCwr();
    // Do not update m_cWnd, under assumption that recovery process will
    // gradually bring it down to m_ssThresh.  Update the 'inflated' value of
    // cWnd used for tracing, however.
    m_tcb->m_cWndInfl = m_tcb->m_ssThresh;
    NS_ASSERT(m_tcb->m_congState != TcpSocketState::CA_CWR);
    NS_LOG_DEBUG(TcpSocketState::TcpCongStateName[m_tcb->m_congState] << " -> CA_CWR");
    m_congestionControl->CongestionStateSet(m_tcb, TcpSocketState::CA_CWR);
    m_tcb->m_congState = TcpSocketState::CA_CWR;
    // CWR state will be exited when the ack exceeds the m_recover variable.
    // Do not set m_recoverActive (which applies to a loss-based recovery)
    // m_recover corresponds to Linux tp->high_seq
    m_recover = m_tcb->m_highTxMark;
    if (!m_congestionControl->HasCongControl())
    {
        // If there is a recovery algorithm, invoke it.
        m_recoveryOps->EnterRecovery(m_tcb,
                                     m_dupAckCount,
                                     UnAckDataCount(),
                                     currentDelivered,
                                     m_txBuffer->GetSacked());
        NS_LOG_INFO("Enter CWR recovery mode; set cwnd to " << m_tcb->m_cWnd << ", ssthresh to "
                                                            << m_tcb->m_ssThresh << ", recover to "
                                                            << m_recover);
    }
}

void
TcpSocketBase::EnterRecovery(uint32_t currentDelivered)
{
    NS_LOG_FUNCTION(this);
    NS_ASSERT(m_tcb->m_congState != TcpSocketState::CA_RECOVERY);

    NS_LOG_DEBUG(TcpSocketState::TcpCongStateName[m_tcb->m_congState] << " -> CA_RECOVERY");

    NS_LOG_INFO("Enter CA_RECOVERY from "
                << TcpSocketState::TcpCongStateName[m_tcb->m_congState] << ": recovering highRxAck "
                << m_highRxAckMark << "; prior recover " << m_recover << " (active "
                << m_recoverActive << "), new recover " << m_tcb->m_highTxMark);

    if (!m_sackEnabled)
    {
        // One segment has left the network, PLUS the head is lost
        m_txBuffer->AddRenoSack();
        m_txBuffer->MarkHeadAsLost();
    }
    else
    {
        if (!m_txBuffer->IsLost(m_txBuffer->HeadSequence()))
        {
            // We received 3 dupacks, but the head is not marked as lost
            // (received less than 3 SACK block ahead).
            // Manually set it as lost.
            m_txBuffer->MarkHeadAsLost();
        }
    }

    InitUndo();

    // A window reduction already in progress (CA_CWR) was signaled
    if (m_tcb->m_congState != TcpSocketState::CA_CWR)
    {
        QueueCwr();
    }

    // RFC 6675, point (4):
    // (4) Invoke fast retransmit and enter loss recovery as follows:
    // (4.1) RecoveryPoint = HighData
    m_recover = m_tcb->m_highTxMark;
    m_recoverActive = true;

    m_congestionControl->CongestionStateSet(m_tcb, TcpSocketState::CA_RECOVERY);
    m_tcb->m_congState = TcpSocketState::CA_RECOVERY;

    // (4.2) ssthresh = cwnd = (FlightSize / 2), FlightSize being the data
    // sent and not cumulatively acknowledged (RFC 5681, Section 2), which
    // includes the segments SACKed or deemed lost, unlike the pipe; it is
    // bounded by cwnd (see CongestionFlightSize())
    uint32_t bytesInFlight = CongestionFlightSize();
    m_tcb->m_ssThresh = m_congestionControl->GetSsThresh(m_tcb, bytesInFlight);

    if (!m_congestionControl->HasCongControl())
    {
        m_recoveryOps->EnterRecovery(m_tcb,
                                     m_dupAckCount,
                                     UnAckDataCount(),
                                     currentDelivered,
                                     m_txBuffer->GetSacked());
        NS_LOG_INFO(m_dupAckCount << " dupack. Enter fast recovery mode."
                                  << "Reset cwnd to " << m_tcb->m_cWnd << ", ssthresh to "
                                  << m_tcb->m_ssThresh << " at fast recovery seqnum " << m_recover
                                  << " calculated in flight: " << bytesInFlight);
    }

    // (4.3) Retransmit the first data segment presumed dropped. A congestion
    // control which sets the window itself on each ACK (e.g., TcpBbr) does so
    // first, on the data in flight before the retransmission, as Linux
    // (tcp_cong_control() before tcp_xmit_recovery()): the retransmission of
    // the segment marked lost is then left to SendPendingData()
    if (!m_congestionControl->HasCongControl() && !m_txBuffer->IsHeadRetransmitted())
    {
        uint32_t sz = SendDataPacket(m_highRxAckMark, m_tcb->m_segmentSize, true);
        NS_ASSERT_MSG(sz > 0, "SendDataPacket returned zero, indicating zero bytes were sent");
    }
    // (4.4) Run SetPipe ()
    // (4.5) Proceed to step (C)
    // these steps are done after the ProcessAck function (SendPendingData)
}

void
TcpSocketBase::DupAck(uint32_t currentDelivered)
{
    NS_LOG_FUNCTION(this);
    // NOTE: We do not count the DupAcks received in CA_LOSS, because we
    // don't know if they are generated by a spurious retransmission or because
    // of a real packet loss. With SACK, it is easy to know, but we do not consider
    // dupacks. Without SACK, there are some heuristics in the RFC 6582, but
    // for now, we do not implement it, leading to ignoring the dupacks.
    if (m_tcb->m_congState == TcpSocketState::CA_LOSS)
    {
        return;
    }

    // RFC 6675, Section 5, 3rd paragraph:
    // If the incoming ACK is a duplicate acknowledgment per the definition
    // in Section 2 (regardless of its status as a cumulative
    // acknowledgment), and the TCP is not currently in loss recovery
    // the TCP MUST increase DupAcks by one ...
    if (m_tcb->m_congState != TcpSocketState::CA_RECOVERY)
    {
        ++m_dupAckCount;
    }

    if (m_tcb->m_congState == TcpSocketState::CA_OPEN)
    {
        // From Open we go Disorder
        NS_ASSERT_MSG(m_dupAckCount == 1,
                      "From OPEN->DISORDER but with " << m_dupAckCount << " dup ACKs");

        m_congestionControl->CongestionStateSet(m_tcb, TcpSocketState::CA_DISORDER);
        m_tcb->m_congState = TcpSocketState::CA_DISORDER;

        NS_LOG_DEBUG("CA_OPEN -> CA_DISORDER");
    }

    if (m_tcb->m_congState == TcpSocketState::CA_RECOVERY)
    {
        // Without SACK, a duplicate ACK that no segment is left to be taken
        // as the SACK of (all the segments sent but the first are) delivers
        // nothing
        bool delivered = true;
        if (!m_sackEnabled)
        {
            // If we are in recovery and we receive a dupack, one segment
            // has left the network. This is equivalent to a SACK of one block.
            uint32_t sacked = m_txBuffer->GetSacked();
            m_txBuffer->AddRenoSack();
            delivered = m_txBuffer->GetSacked() > sacked;
            // The recovery algorithm sizes cWnd from the bytes in flight,
            // which must account for the segment that has just left
            BytesInFlight();
        }
        if (!m_congestionControl->HasCongControl())
        {
            m_recoveryOps->DoRecovery(m_tcb, currentDelivered, delivered);
            NS_LOG_INFO(m_dupAckCount << " Dupack received in fast recovery mode."
                                         "Increase cwnd to "
                                      << m_tcb->m_cWnd);
        }
    }
    else if (m_tcb->m_congState == TcpSocketState::CA_DISORDER)
    {
        // m_dupackCount should not exceed its threshold in CA_DISORDER state
        // when m_recoverActive has not been set. When recovery point
        // have been set after timeout, the sender could enter into CA_DISORDER
        // after receiving new ACK smaller than m_recover. After that, m_dupackCount
        // can be equal and larger than m_retxThresh and we should avoid entering
        // CA_RECOVERY and reducing sending rate again.
        NS_ASSERT((m_dupAckCount <= m_retxThresh) || m_recoverActive);

        uint32_t fackDiff = 0;
        if (m_fackEnabled)
        {
            uint32_t headSeq = m_txBuffer->HeadSequence().GetValue();
            if (SequenceNumber32(m_sndFack) > SequenceNumber32(headSeq))
            {
                fackDiff = m_sndFack - headSeq;
            }
        }

        // RFC 6675, Section 5, continuing:
        // ... and take the following steps:
        // (1) If DupAcks >= DupThresh, go to step (4).
        //     Sequence number comparison (m_highRxAckMark >= m_recover) will take
        //     effect only when m_recover has been set. Hence, we can avoid to use
        //     m_recover in the last congestion event and fail to enter
        //     CA_RECOVERY when sequence number is advanced significantly since
        //     the last congestion event, which could be common for
        //     bandwidth-greedy application in high speed and reliable network
        //     (such as datacenter network) whose sending rate is constrained by
        //     TCP socket buffer size at receiver side.

        // Check FACK recovery condition

        if ((m_fackEnabled && fackDiff > m_tcb->m_segmentSize * 3) ||
            ((m_dupAckCount == m_retxThresh) &&
             (m_highRxAckMark >= m_recover || (!m_recoverActive))))
        {
            EnterRecovery(currentDelivered);
            NS_ASSERT(m_tcb->m_congState == TcpSocketState::CA_RECOVERY);
        }
        // (2) If DupAcks < DupThresh but IsLost (HighACK + 1) returns true
        // (indicating at least three segments have arrived above the current
        // cumulative acknowledgment point, which is taken to indicate loss)
        // go to step (4).  Note that m_highRxAckMark is (HighACK + 1)
        else if (m_txBuffer->IsLost(m_highRxAckMark))
        {
            EnterRecovery(currentDelivered);
            NS_ASSERT(m_tcb->m_congState == TcpSocketState::CA_RECOVERY);
        }
        else
        {
            // (3) The TCP MAY transmit previously unsent data segments as per
            // Limited Transmit [RFC5681] ...except that the number of octets
            // which may be sent is governed by pipe and cwnd as follows:
            //
            // (3.1) Set HighRxt to HighACK.
            // Not clear in RFC. We don't do this here, since we still have
            // to retransmit the segment.

            if (!m_sackEnabled && m_limitedTx)
            {
                m_txBuffer->AddRenoSack();

                // In limited transmit, cwnd Infl is not updated.
            }
        }
    }
}

/* Process the newly received ACK */
void
TcpSocketBase::ReceivedAck(Ptr<Packet> packet, const TcpHeader& tcpHeader)
{
    NS_LOG_FUNCTION(this << tcpHeader);

    NS_ASSERT(0 != (tcpHeader.GetFlags() & TcpHeader::ACK));
    NS_ASSERT(m_tcb->m_segmentSize > 0);

    uint32_t previousLost = m_txBuffer->GetLost();
    uint64_t previousLostDelivered = m_txBuffer->GetLostDelivered();
    uint32_t priorInFlight = m_tcb->m_bytesInFlight.Get();

    // RFC 6675, Section 5, 1st paragraph:
    // Upon the receipt of any ACK containing SACK information, the
    // scoreboard MUST be updated via the Update () routine (done in ReadOptions)
    uint32_t bytesSacked = 0;
    uint64_t previousDelivered = m_rateOps->GetConnectionRate().m_delivered;
    ReadOptions(tcpHeader, &bytesSacked);

    SequenceNumber32 ackNumber = tcpHeader.GetAckNumber();
    SequenceNumber32 oldHeadSequence = m_txBuffer->HeadSequence();

    if (ackNumber < oldHeadSequence)
    {
        NS_LOG_DEBUG("Possibly received a stale ACK (ack number < head sequence)");
        // If there is any data piggybacked, store it into m_rxBuffer
        if (packet->GetSize() > 0)
        {
            ReceivedData(packet, tcpHeader);
        }
        return;
    }

    // RFC 5681, Section 2, condition (e): an ACK which changes the advertised
    // window is a window update, and it is not counted as a duplicate ACK
    uint32_t ackWindow = static_cast<uint32_t>(tcpHeader.GetWindowSize()) << m_sndWindShift;
    m_ackWindowUpdate = m_lastAckWindow.has_value() && *m_lastAckWindow != ackWindow;
    m_lastAckWindow = ackWindow;

    if ((ackNumber > oldHeadSequence) && (ackNumber < m_recover) &&
        (m_tcb->m_congState == TcpSocketState::CA_RECOVERY))
    {
        uint32_t segAcked = (ackNumber - oldHeadSequence) / m_tcb->m_segmentSize;
        for (uint32_t i = 0; i < segAcked; i++)
        {
            if (m_txBuffer->IsRetransmittedDataAcked(ackNumber - (i * m_tcb->m_segmentSize)))
            {
                m_tcb->m_isRetransDataAcked = true;
                NS_LOG_DEBUG("Ack Number " << ackNumber << "is ACK of retransmitted packet.");
            }
        }
    }

    // Without SACK, the segments guessed as SACKed by the duplicate ACKs of a
    // fast recovery were already counted as delivered by the recovery
    uint32_t renoSackedBefore =
        (!m_sackEnabled && m_tcb->m_congState == TcpSocketState::CA_RECOVERY)
            ? m_txBuffer->GetSacked()
            : 0;

    m_txBuffer->DiscardUpTo(ackNumber, MakeCallback(&TcpRateOps::SkbDelivered, m_rateOps));

    if (m_linuxCaRtt)
    {
        UpdateLinuxCaRtt(tcpHeader, ackNumber > oldHeadSequence, bytesSacked > 0);
    }

    if (m_sackRttSamples && ackNumber == oldHeadSequence && bytesSacked > 0 &&
        m_txBuffer->GetOldestNewlySackedSentTime() != Time::Max())
    {
        Time rtt = Simulator::Now() - m_txBuffer->GetOldestNewlySackedSentTime();
        NS_LOG_DEBUG("RTT sample from the SACKed data: " << rtt);
        UpdateRttEstimate(rtt);
    }

    // As Linux, which re-arms the timer only when the SACKed data was sent
    // before the head was last (re)transmitted, so that the timeout does
    // not keep moving away during the recovery
    if (m_sackRestartsRto && ackNumber == oldHeadSequence && bytesSacked > 0 &&
        m_retxEvent.IsPending() &&
        m_txBuffer->GetOldestNewlySackedSentTime() < m_txBuffer->GetHeadLastSent())
    {
        NS_LOG_LOGIC(this << " SACK restarts the RTO, to expire at "
                          << (Simulator::Now() + GetRtoTimerDelay()).GetSeconds());
        m_retxEvent.Cancel();
        m_retxEvent = Simulator::Schedule(GetRtoTimerDelay(), &TcpSocketBase::ReTxTimeout, this);
    }

    // Without SACK, the guessed SACKs only have meaning during fast recovery
    if (!m_sackEnabled && ackNumber > oldHeadSequence)
    {
        if (m_tcb->m_congState == TcpSocketState::CA_RECOVERY && ackNumber < m_recover)
        {
            // Partial ACK: the next segment is presumed lost (RFC 6582)
            // and will be retransmitted
            NS_LOG_INFO("Partial ACK. Manually setting head as lost");
            m_txBuffer->MarkHeadAsLost();
        }
        else
        {
            // A cumulative ACK that advances outside of fast recovery,
            // or that ends it, clears the guesses
            m_txBuffer->ResetRenoSack();
        }
    }

    auto currentDelivered =
        static_cast<uint32_t>(m_rateOps->GetConnectionRate().m_delivered - previousDelivered);
    if (renoSackedBefore > 0 && currentDelivered > 0)
    {
        // As Linux tcp_remove_reno_sacks(): the cumulative ACK delivers the
        // hole, and the rest of the acknowledged data was delivered as
        // duplicate ACKs (RFC 9937, Section 6.2, DeliveredData without SACK)
        currentDelivered = currentDelivered > renoSackedBefore + m_tcb->m_segmentSize
                               ? currentDelivered - renoSackedBefore
                               : std::min(currentDelivered, m_tcb->m_segmentSize);
    }
    m_tcb->m_lastAckedSackedBytes = currentDelivered;

    // As Linux tcp_clean_rtx_queue(), before tcp_fastretrans_alert(), with
    // SND.UNA already updated
    if (m_linuxPktsAcked && oldHeadSequence < m_tcb->m_highTxMark)
    {
        const uint32_t mss = m_tcb->m_segmentSize;
        m_tcb->m_lastAckedSeq = ackNumber;
        m_congestionControl->PktsAcked(m_tcb,
                                       (ackNumber - oldHeadSequence + mss - 1) / mss,
                                       m_ackRttSample);
    }

    if (m_tcb->m_congState == TcpSocketState::CA_CWR && (ackNumber > m_recover))
    {
        // Recovery is over after the window exceeds m_recover
        // (although it may be re-entered below if ECE is still set)
        NS_LOG_DEBUG(TcpSocketState::TcpCongStateName[m_tcb->m_congState] << " -> CA_OPEN");
        m_congestionControl->CongestionStateSet(m_tcb, TcpSocketState::CA_OPEN);
        m_tcb->m_congState = TcpSocketState::CA_OPEN;
        if (!m_congestionControl->HasCongControl())
        {
            m_tcb->m_cWnd = m_tcb->m_ssThresh.Get();
            m_recoveryOps->ExitRecovery(m_tcb);
            m_congestionControl->CwndEvent(m_tcb, TcpSocketState::CA_EVENT_COMPLETE_CWR);
        }
    }

    if (ackNumber > oldHeadSequence && (m_tcb->m_ecnState != TcpSocketState::ECN_DISABLED) &&
        (tcpHeader.GetFlags() & TcpHeader::ECE))
    {
        if (m_ecnEchoSeq < ackNumber)
        {
            NS_LOG_INFO("Received ECN Echo is valid");
            m_ecnEchoSeq = ackNumber;
            NS_LOG_DEBUG(TcpSocketState::EcnStateName[m_tcb->m_ecnState] << " -> ECN_ECE_RCVD");
            m_tcb->m_ecnState = TcpSocketState::ECN_ECE_RCVD;
            if (m_tcb->m_congState != TcpSocketState::CA_CWR)
            {
                EnterCwr(currentDelivered);
            }
        }
    }
    else if (m_tcb->m_ecnState == TcpSocketState::ECN_ECE_RCVD &&
             !(tcpHeader.GetFlags() & TcpHeader::ECE))
    {
        m_tcb->m_ecnState = TcpSocketState::ECN_IDLE;
    }

    // Update bytes in flight before processing the ACK for proper calculation of congestion window
    NS_LOG_INFO("Update bytes in flight before processing the ACK.");
    BytesInFlight();

    bool receivedData = packet->GetSize() > 0;

    // RFC 6675 Section 5: 2nd, 3rd paragraph and point (A), (B) implementation
    // are inside the function ProcessAck
    ProcessAck(ackNumber, (bytesSacked > 0), currentDelivered, oldHeadSequence, receivedData);
    m_tcb->m_isRetransDataAcked = false;

    if (m_congestionControl->HasCongControl())
    {
        // The bytes newly deemed lost by this ACK (rs->losses of Linux),
        // whether or not it also acknowledges bytes deemed lost before
        uint64_t lostBefore = previousLost + m_txBuffer->GetLostDelivered() - previousLostDelivered;
        uint32_t currentLost = m_txBuffer->GetLost();
        auto lost = static_cast<uint32_t>(currentLost > lostBefore ? currentLost - lostBefore : 0);
        auto rateSample = m_rateOps->GenerateSample(currentDelivered,
                                                    lost,
                                                    false,
                                                    priorInFlight,
                                                    m_tcb->m_minRtt);
        auto rateConn = m_rateOps->GetConnectionRate();
        m_congestionControl->CongControl(m_tcb, rateConn, rateSample);
    }

    // If there is any data piggybacked, store it into m_rxBuffer
    if (receivedData)
    {
        ReceivedData(packet, tcpHeader);
    }

    // RFC 6675, Section 5, point (C), try to send more data. NB: (C) is implemented
    // inside SendPendingData
    SendPendingData(m_connected);

    if (m_sndSkbOverhead > 0 && m_lxNoSpace && GetTxAvailable() > 0)
    {
        NotifySend(GetTxAvailable());
    }
}

void
TcpSocketBase::ProcessAck(const SequenceNumber32& ackNumber,
                          bool scoreboardUpdated,
                          uint32_t currentDelivered,
                          const SequenceNumber32& oldHeadSequence,
                          bool receivedData)
{
    NS_LOG_FUNCTION(this << ackNumber << scoreboardUpdated << currentDelivered << oldHeadSequence);
    // RFC 6675, Section 5, 2nd paragraph:
    // If the incoming ACK is a cumulative acknowledgment, the TCP MUST
    // reset DupAcks to zero.
    bool exitedFastRecovery = false;
    uint32_t oldDupAckCount = m_dupAckCount; // remember the old value
    m_tcb->m_lastAckedSeq = ackNumber;       // Update lastAckedSeq
    uint32_t bytesAcked = 0;

    /* In RFC 5681 the definition of duplicate acknowledgment was strict:
     *
     * (a) the receiver of the ACK has outstanding data,
     * (b) the incoming acknowledgment carries no data,
     * (c) the SYN and FIN bits are both off,
     * (d) the acknowledgment number is equal to the greatest acknowledgment
     *     received on the given connection (TCP.UNA from [RFC793]),
     * (e) the advertised window in the incoming acknowledgment equals the
     *     advertised window in the last incoming acknowledgment.
     *
     * With RFC 6675, this definition has been reduced:
     *
     * (a) the ACK is carrying a SACK block that identifies previously
     *     unacknowledged and un-SACKed octets between HighACK (TCP.UNA) and
     *     HighData (m_highTxMark)
     *
     * Without SACK, the check below implements conditions a), b), d) and e), and c) is
     * prevented by virtue of not reaching this code if SYN or FIN is set. With SACK, an ACK
     * carrying new SACK information is a duplicate ACK even if it updates the window.
     */

    if (m_fackEnabled && ackNumber == m_txBuffer->HeadSequence() &&
        m_tcb->m_congState == TcpSocketState::CA_RECOVERY)
    {
        if (m_outstandingRetransBytes > m_tcb->m_segmentSize)
        {
            m_outstandingRetransBytes -= m_tcb->m_segmentSize;
        }
        else
        {
            m_outstandingRetransBytes = 0;
        }
    }

    bool isDupack = m_sackEnabled
                        ? scoreboardUpdated
                        : (ackNumber == oldHeadSequence && ackNumber < m_tcb->m_highTxMark &&
                           !receivedData && !m_ackWindowUpdate);

    NS_LOG_DEBUG("ACK of " << ackNumber << " SND.UNA=" << oldHeadSequence
                           << " SND.NXT=" << m_tcb->m_nextTxSequence
                           << " in state: " << TcpSocketState::TcpCongStateName[m_tcb->m_congState]
                           << " with m_recover: " << m_recover);

    // The congestion control is told of the ACK before the window is set by
    // the recovery (as Linux calls pkts_acked() before tcp_fastretrans_alert()
    // and tcp_cwnd_reduction()): a window that the congestion control changes
    // on an ACK must not override the one that PRR sets for it (RFC 9937)
    if (ackNumber == oldHeadSequence && ackNumber < m_tcb->m_highTxMark)
    {
        // DupAck. Artificially call PktsAcked: after all, one segment has been ACKed.
        StatePktsAcked(1);
    }

    // RFC 6675, Section 5, 3rd paragraph:
    // If the incoming ACK is a duplicate acknowledgment per the definition
    // in Section 2 (regardless of its status as a cumulative
    // acknowledgment), and the TCP is not currently in loss recovery
    if (isDupack)
    {
        // loss recovery check is done inside this function thanks to
        // the congestion state machine
        DupAck(currentDelivered);
    }

    if (ackNumber == oldHeadSequence && ackNumber == m_tcb->m_highTxMark)
    {
        // Dupack, but the ACK is precisely equal to the nextTxSequence
        return;
    }
    else if (ackNumber == oldHeadSequence && ackNumber > m_tcb->m_highTxMark)
    {
        // ACK of the FIN bit ... nextTxSequence is not updated since we
        // don't have anything to transmit
        NS_LOG_DEBUG("Update nextTxSequence manually to " << ackNumber);
        m_tcb->m_nextTxSequence = ackNumber;
    }
    else if (ackNumber == oldHeadSequence)
    {
        // DupAck, the congestion control was told of it above
    }
    else if (ackNumber > oldHeadSequence)
    {
        // Please remember that, with SACK, we can enter here even if we
        // received a dupack.
        bytesAcked = currentDelivered;
        uint32_t segsAcked = bytesAcked / m_tcb->m_segmentSize;
        m_bytesAckedNotProcessed += bytesAcked % m_tcb->m_segmentSize;
        bytesAcked -= bytesAcked % m_tcb->m_segmentSize;

        if (m_bytesAckedNotProcessed >= m_tcb->m_segmentSize)
        {
            segsAcked += 1;
            bytesAcked += m_tcb->m_segmentSize;
            m_bytesAckedNotProcessed -= m_tcb->m_segmentSize;
        }
        NS_LOG_DEBUG("Set segsAcked: " << segsAcked
                                       << " based on currentDelivered: " << currentDelivered);

        // Dupack count is reset to eventually fast-retransmit after 3 dupacks.
        // Any SACK-ed segment will be cleaned up by DiscardUpTo.
        // In the case that we advanced SND.UNA, but the ack contains SACK blocks,
        // we do not reset. At the third one we will retransmit.
        // If we are already in recovery, this check is useless since dupAcks
        // are not considered in this phase. When from Recovery we go back
        // to open, then dupAckCount is reset anyway.
        if (!isDupack)
        {
            m_dupAckCount = 0;
        }

        // RFC 6675, Section 5, part (B)
        // (B) Upon receipt of an ACK that does not cover RecoveryPoint, the
        // following actions MUST be taken:
        //
        // (B.1) Use Update () to record the new SACK information conveyed
        //       by the incoming ACK.
        // (B.2) Use SetPipe () to re-calculate the number of octets still
        //       in the network.
        //
        // (B.1) is done at the beginning, while (B.2) is delayed to part (C) while
        // trying to transmit with SendPendingData. We are not allowed to exit
        // the CA_RECOVERY phase. Just process this partial ack (RFC 5681)
        if (ackNumber < m_recover && m_tcb->m_congState == TcpSocketState::CA_RECOVERY)
        {
            // This partial ACK acknowledge the fact that one segment has been
            // previously lost and now successfully received. All others have
            // been processed when they come under the form of dupACKs
            StatePktsAcked(1);

            // Before retransmitting the packet perform DoRecovery and check if
            // there is available window
            if (!m_congestionControl->HasCongControl() && currentDelivered > 0)
            {
                m_recoveryOps->DoRecovery(m_tcb, currentDelivered, false);
            }

            // In NewReno-style recovery (RFC 6582, Section 3.2), a partial ACK
            // implies that the first unacknowledged segment was also lost:
            // retransmit it (ReceivedAck() has marked it as lost). With
            // SACK, retransmissions are governed by the scoreboard and the
            // pipe (RFC 6675, Section 5), and are performed by
            // SendPendingData(), so no forced retransmission must occur here
            if (!m_sackEnabled && !m_partialAckInWindow)
            {
                DoRetransmit(); // Assume the next seq is lost. Retransmit lost packet
            }
            // The inflated window (traced only) is deflated by the newly
            // acknowledged data, with or without SACK
            m_tcb->m_cWndInfl = SafeSubtraction(m_tcb->m_cWndInfl, bytesAcked);
            // RFC 6298, Section 5.3: restart the retransmission timer on every
            // ACK acknowledging new data. This is the Slow-but-Steady variant
            // of NewReno (RFC 3782, Section 4, dropped by RFC 6582, see its
            // Appendix B), which departs from RFC 6582, Section 3.2, step 3,
            // where only the first partial ACK of a fast recovery resets the
            // timer: a recovery spanning several partial ACKs would otherwise
            // be cut short by a spurious timeout
            NewAck(ackNumber, true);
            NS_LOG_DEBUG("Partial ACK of " << ackNumber
                                           << " in fast recovery (RTO reset);"
                                              " cwnd set to "
                                           << m_tcb->m_cWnd << " recover seq: " << m_recover
                                           << " dupAck count: " << m_dupAckCount);
        }
        // From RFC 6675 section 5.1
        // In addition, a new recovery phase (as described in Section 5) MUST NOT
        // be initiated until HighACK is greater than or equal to the new value
        // of RecoveryPoint.
        else if (ackNumber < m_recover && m_tcb->m_congState == TcpSocketState::CA_LOSS)
        {
            StatePktsAcked(segsAcked);
            IncreaseWindow(segsAcked);

            NS_LOG_DEBUG(" Cong Control Called, cWnd=" << m_tcb->m_cWnd
                                                       << " ssTh=" << m_tcb->m_ssThresh);
            if (!m_sackEnabled)
            {
                NS_ASSERT_MSG(
                    m_txBuffer->GetSacked() == 0,
                    "Some segment got dup-acked in CA_LOSS state: " << m_txBuffer->GetSacked());
            }
            // RFC 6298, Section 5.3: restart the retransmission timer on every
            // ACK acknowledging new data, after a timeout as during a fast
            // recovery
            NewAck(ackNumber, true);
            NS_LOG_DEBUG("Partial ACK of " << ackNumber << " in CA_LOSS (RTO reset)");
        }
        else if (m_nonSackUndo && !m_sackEnabled && ackNumber == m_recover &&
                 (m_tcb->m_congState == TcpSocketState::CA_RECOVERY ||
                  m_tcb->m_congState == TcpSocketState::CA_LOSS))
        {
            // As Linux tcp_is_non_sack_preventing_reopen(): the recovery is
            // held until data above the recovery point is acknowledged. With
            // no retransmission left in flight, Linux clears retrans_stamp
            // here, and the next ACK finds the recovery undoable
            // (tcp_packet_delayed())
            m_undoPending = m_undoSsThresh > 0;
            if (m_tcb->m_congState == TcpSocketState::CA_RECOVERY)
            {
                StatePktsAcked(segsAcked);
                if (!m_congestionControl->HasCongControl() && currentDelivered > 0)
                {
                    m_recoveryOps->DoRecovery(m_tcb, currentDelivered, false);
                }
                m_tcb->m_cWndInfl = SafeSubtraction(m_tcb->m_cWndInfl, bytesAcked);
            }
            else
            {
                StatePktsAcked(segsAcked);
                IncreaseWindow(segsAcked);
            }
            NewAck(ackNumber, true);
            NS_LOG_DEBUG("ACK of the recovery point "
                         << ackNumber << " in "
                         << TcpSocketState::TcpCongStateName[m_tcb->m_congState]
                         << ": recovery held");
        }
        else if (m_tcb->m_congState == TcpSocketState::CA_CWR)
        {
            StatePktsAcked(segsAcked);
            // TODO: need to check behavior if marking is compounded by loss
            // and/or packet reordering
            if (!m_congestionControl->HasCongControl() && currentDelivered > 0)
            {
                m_recoveryOps->DoRecovery(m_tcb, currentDelivered, false);
            }
            NewAck(ackNumber, true);
        }
        else
        {
            if (m_tcb->m_congState == TcpSocketState::CA_OPEN)
            {
                StatePktsAcked(segsAcked);
            }
            else if (m_tcb->m_congState == TcpSocketState::CA_DISORDER)
            {
                if (segsAcked >= oldDupAckCount)
                {
                    StatePktsAcked(segsAcked - oldDupAckCount);
                }

                if (!isDupack)
                {
                    // The network reorder packets. Linux changes the counting lost
                    // packet algorithm from FACK to NewReno. We simply go back in Open.
                    m_congestionControl->CongestionStateSet(m_tcb, TcpSocketState::CA_OPEN);
                    m_tcb->m_congState = TcpSocketState::CA_OPEN;
                    NS_LOG_DEBUG(segsAcked << " segments acked in CA_DISORDER, ack of " << ackNumber
                                           << " exiting CA_DISORDER -> CA_OPEN");
                }
                else
                {
                    // The ACK advances SND.UNA and also SACKs new data: it
                    // acknowledges new data, hence it grows the window
                    // (RFC 5681, Section 3.1), by the segments delivered as
                    // Linux tcp_cong_control() with acked_sacked when the
                    // congestion control counts them, and restarts the
                    // retransmission timer (RFC 6298, Section 5.3)
                    IncreaseWindow(segsAcked);
                    m_tcb->m_cWndInfl = m_tcb->m_cWnd;
                    NewAck(ackNumber, true);
                    NS_LOG_DEBUG(segsAcked << " segments acked in CA_DISORDER, ack of " << ackNumber
                                           << " but still in CA_DISORDER");
                }
            }
            // RFC 6675, Section 5:
            // Once a TCP is in the loss recovery phase, the following procedure
            // MUST be used for each arriving ACK:
            // (A) An incoming cumulative ACK for a sequence number greater than
            // RecoveryPoint signals the end of loss recovery, and the loss
            // recovery phase MUST be terminated.  Any information contained in
            // the scoreboard for sequence numbers greater than the new value of
            // HighACK SHOULD NOT be cleared when leaving the loss recovery
            // phase.
            else if (m_tcb->m_congState == TcpSocketState::CA_RECOVERY)
            {
                // PktsAcked() is told of every segment this ACK covers, as
                // Linux tcp_clean_rtx_queue() does, while the window grows
                // below only by the segments newly delivered (Linux
                // acked_sacked in tcp_cong_control()): the segments SACKed
                // during the recovery were already counted as delivered
                StatePktsAcked(static_cast<uint32_t>(ackNumber - oldHeadSequence) /
                               m_tcb->m_segmentSize);
                m_congestionControl->CwndEvent(m_tcb, TcpSocketState::CA_EVENT_COMPLETE_CWR);
                m_congestionControl->CongestionStateSet(m_tcb, TcpSocketState::CA_OPEN);
                m_tcb->m_congState = TcpSocketState::CA_OPEN;
                exitedFastRecovery = true;
                m_dupAckCount = 0; // From recovery to open, reset dupack

                NS_LOG_DEBUG(segsAcked << " segments acked in CA_RECOVER, ack of " << ackNumber
                                       << ", exiting CA_RECOVERY -> CA_OPEN");
            }
            else if (m_tcb->m_congState == TcpSocketState::CA_LOSS)
            {
                // As on the exit from CA_RECOVERY, PktsAcked() is told of
                // every segment this ACK covers, while the window grows only
                // by the segments newly delivered (Linux acked_sacked in
                // tcp_cong_control()): the segments SACKed before the
                // timeout were already counted as delivered
                StatePktsAcked(static_cast<uint32_t>(ackNumber - oldHeadSequence) /
                               m_tcb->m_segmentSize);

                m_congestionControl->CongestionStateSet(m_tcb, TcpSocketState::CA_OPEN);
                m_tcb->m_congState = TcpSocketState::CA_OPEN;
                TryUndoNonSackRecovery();

                // Without SACK, the ACK of the retransmission may cover a
                // whole window held by the receiver: as Linux
                // tcp_slow_start() does, the slow start stops at ssThresh,
                // and the remaining segments go to the congestion
                // avoidance, so that the window does not burst past
                // ssThresh at once (RFC 5681, Section 3.1)
                if (m_tcb->m_cWnd < m_tcb->m_ssThresh)
                {
                    uint32_t toSsThresh =
                        (m_tcb->m_ssThresh - m_tcb->m_cWnd + m_tcb->m_segmentSize - 1) /
                        m_tcb->m_segmentSize;
                    if (segsAcked > toSsThresh)
                    {
                        IncreaseWindow(toSsThresh);
                        segsAcked = m_tcb->m_cWnd >= m_tcb->m_ssThresh ? segsAcked - toSsThresh : 0;
                    }
                }
                NS_LOG_DEBUG(segsAcked << " segments acked in CA_LOSS, ack of" << ackNumber
                                       << ", exiting CA_LOSS -> CA_OPEN");
            }

            if (ackNumber >= m_recover)
            {
                // All lost segments in the congestion event have been
                // retransmitted successfully. The recovery point (m_recover)
                // should be deactivated.
                m_recoverActive = false;
            }

            if (exitedFastRecovery)
            {
                NewAck(ackNumber, true);
                // A congestion control which sets the window itself (e.g.,
                // TcpBbr, which restores the window of before the recovery)
                // keeps it, as Linux tcp_end_cwnd_reduction()
                if (!m_congestionControl->HasCongControl())
                {
                    m_tcb->m_cWnd = m_tcb->m_ssThresh.Get();
                    m_recoveryOps->ExitRecovery(m_tcb);
                }
                TryUndoNonSackRecovery();
                NS_LOG_INFO("Exit CA_RECOVERY -> CA_OPEN: ack "
                            << ackNumber << " reached recover " << m_recover << "; cWnd "
                            << m_tcb->m_cWnd << " ssThresh " << m_tcb->m_ssThresh
                            << " bytesInFlight " << BytesInFlight());
                NS_LOG_DEBUG("Leaving Fast Recovery; BytesInFlight() = "
                             << BytesInFlight() << "; cWnd = " << m_tcb->m_cWnd);
            }
            if (m_recoveryReentry && exitedFastRecovery && m_sackEnabled &&
                m_txBuffer->GetLost() > 0)
            {
                // As Linux tcp_fastretrans_alert(), data still deemed lost
                // when the recovery ends starts a new one on the same ACK
                // (tcp_time_to_recover()). The window is not grown on that
                // ACK, which tcp_cong_control() gives to the reduction only,
                // and as the ACK advances SND.UNA, PRR may send a segment
                // more (tcp_cwnd_reduction(), FLAG_SND_UNA_ADVANCED)
                NS_LOG_DEBUG("Lost data left at the end of the recovery: recovering again");
                m_tcb->m_isRetransDataAcked = true;
                EnterRecovery(currentDelivered);
            }
            if (m_tcb->m_congState == TcpSocketState::CA_OPEN)
            {
                IncreaseWindow(segsAcked);

                m_tcb->m_cWndInfl = m_tcb->m_cWnd;

                NS_LOG_LOGIC("Congestion control called: cWnd: " << m_tcb->m_cWnd
                                                                 << " ssTh: " << m_tcb->m_ssThresh
                                                                 << " segsAcked: " << segsAcked);

                NewAck(ackNumber, true);
            }
        }
    }
    // Update the pacing rate, since m_congestionControl->IncreaseWindow() or
    // m_congestionControl->PktsAcked () may change m_tcb->m_cWnd
    // Make sure that control reaches the end of this function and there is no
    // return in between
    UpdatePacingRate();
}

void
TcpSocketBase::QueueCwr()
{
    NS_LOG_FUNCTION(this);
    if (m_tcb->m_ecnState != TcpSocketState::ECN_DISABLED)
    {
        m_cwrPending = true;
    }
}

void
TcpSocketBase::InitUndo()
{
    NS_LOG_FUNCTION(this);
    if (m_tcb->m_congState == TcpSocketState::CA_OPEN ||
        m_tcb->m_congState == TcpSocketState::CA_DISORDER)
    {
        uint32_t cWnd = m_tcb->m_cWnd.Get();
        m_undoCwnd = cWnd;
        m_undoSsThresh = std::max(m_tcb->m_ssThresh.Get(), cWnd / 2 + cWnd / 4);
    }
    else
    {
        // A window reduction already in progress (CA_CWR) cannot be undone
        m_undoSsThresh = 0;
    }
    m_undoPending = false;
}

void
TcpSocketBase::TryUndoNonSackRecovery()
{
    NS_LOG_FUNCTION(this);
    if (!m_undoPending)
    {
        return;
    }
    // As Linux tcp_undo_cwnd_reduction() with tcp_reno_undo_cwnd()
    m_undoPending = false;
    m_tcb->m_cWnd = std::max(m_tcb->m_cWnd.Get(), m_undoCwnd);
    m_tcb->m_cWndInfl = m_tcb->m_cWnd;
    if (m_undoSsThresh > m_tcb->m_ssThresh)
    {
        m_tcb->m_ssThresh = m_undoSsThresh;
        // The reduction undone is no longer signaled (tcp_ecn_withdraw_cwr())
        m_cwrPending = false;
    }
    m_undoSsThresh = 0;
    NS_LOG_INFO("Window reduction undone: cWnd " << m_tcb->m_cWnd << " ssThresh "
                                                 << m_tcb->m_ssThresh);
}

void
TcpSocketBase::SetLinuxTcpMem(std::string limits)
{
    NS_LOG_FUNCTION(this << limits);
    std::replace(limits.begin(), limits.end(), ',', ' ');
    std::istringstream in(limits);
    std::array<int64_t, 3> values{};
    for (auto& value : values)
    {
        if (!(in >> value))
        {
            NS_ABORT_MSG_IF(!limits.empty() && limits.find_first_not_of(' ') != std::string::npos,
                            "LinuxTcpMem takes three numbers of pages: " << limits);
            m_lxTcpMem = {};
            return;
        }
    }
    m_lxTcpMem = values;
}

std::string
TcpSocketBase::GetLinuxTcpMem() const
{
    if (m_lxTcpMem[2] == 0)
    {
        return "";
    }
    std::ostringstream out;
    out << m_lxTcpMem[0] << " " << m_lxTcpMem[1] << " " << m_lxTcpMem[2];
    return out.str();
}

int64_t
TcpSocketBase::LinuxWinFromSpace(int64_t space) const
{
    // As tcp_win_from_space(): an arithmetic shift, also for a negative space
    int64_t scaled = space * static_cast<int64_t>(m_lxRatio);
    return scaled >= 0 ? scaled >> 8 : -((-scaled + 255) >> 8);
}

TcpSocketBase::LinuxRcvSkb
TcpSocketBase::LinuxNewSkb(SequenceNumber32 seq, uint32_t len, uint32_t ipLen) const
{
    LinuxRcvSkb skb;
    skb.seq = seq;
    skb.end = seq + len;
    if (ipLen <= m_rcvCopyBreak)
    {
        // Copied into the head, the receive buffer being freed
        skb.truesize = m_rcvHeadTruesize;
        skb.tailroom = m_rcvCopyBreak - ipLen;
        skb.headData = len > 0;
        return skb;
    }
    // The receive buffers are taken in turn, one per segment received:
    // the last one carved from a page takes what is left of it
    // (add_recvbuf_mergeable())
    uint32_t frag = m_rcvFragTruesize;
    if (m_rcvFragPageSize >= m_rcvFragTruesize && m_rcvFragTruesize > 0)
    {
        uint64_t perPage = m_rcvFragPageSize / m_rcvFragTruesize;
        uint64_t buffer = m_lxRxPackets + m_rcvBufferOffset - 1;
        if (buffer % perPage == perPage - 1)
        {
            frag += m_rcvFragPageSize - static_cast<uint32_t>(perPage) * m_rcvFragTruesize;
        }
    }
    skb.truesize = m_rcvHeadTruesize + frag;
    skb.frags = 1;
    skb.fragSize = frag;
    return skb;
}

std::optional<uint32_t>
TcpSocketBase::LinuxCoalesce(LinuxRcvSkb& to, const LinuxRcvSkb& from) const
{
    const uint32_t maxSkbFrags = 17;
    const uint32_t len = from.end - from.seq;
    if (to.frags == 0 && len <= to.tailroom)
    {
        // Copied into the room left in the head
        to.tailroom -= len;
        to.end = from.end;
        return 0;
    }
    uint32_t delta;
    if (from.headData)
    {
        // The head, a page fragment, is stolen as one more fragment, the
        // sk_buff freed
        if (to.frags + from.frags >= maxSkbFrags)
        {
            return std::nullopt;
        }
        delta = from.truesize - LINUX_SKB_STRUCT_SIZE;
        to.frags++;
    }
    else
    {
        // The page fragments are moved, the head of the skb freed
        if (to.frags + from.frags > maxSkbFrags)
        {
            return std::nullopt;
        }
        delta = from.truesize - m_rcvHeadTruesize;
    }
    to.frags += from.frags;
    to.tailroom = 0;
    to.truesize += delta;
    to.end = from.end;
    return delta;
}

void
TcpSocketBase::LinuxQueueSegment(const LinuxRcvSkb& newSkb, bool inSequence)
{
    NS_LOG_FUNCTION(this << newSkb.seq << newSkb.end << inSequence);
    LinuxRcvSkb skb = newSkb;
    const SequenceNumber32 seq = skb.seq;
    // truesize_adjust(): the memory of an skb whose data is all in its page
    // fragment, without its head
    const uint32_t adjusted = skb.frags > 0 ? skb.fragSize : skb.truesize;
    if (inSequence)
    {
        // tcp_queue_rcv(): appended to the last skb not read yet
        std::optional<uint32_t> delta;
        if (!m_lxRcvQueue.empty() && m_lxRcvQueue.back().end == seq)
        {
            delta = LinuxCoalesce(m_lxRcvQueue.back(), skb);
        }
        if (delta)
        {
            LinuxMemCharge(*delta);
        }
        else
        {
            m_lxRcvQueue.push_back(skb);
            LinuxMemCharge(skb.truesize);
        }
        LinuxDataReceived(skb, adjusted, true);
        // tcp_ofo_queue(): the out-of-order skbs now in sequence follow
        SequenceNumber32 rcvNxt = m_tcb->m_rxBuffer->NextRxSequence();
        while (!m_lxOfoQueue.empty() && m_lxOfoQueue.begin()->first <= rcvNxt)
        {
            LinuxRcvSkb ofo = m_lxOfoQueue.begin()->second;
            m_lxOfoQueue.erase(m_lxOfoQueue.begin());
            if (ofo.end <= m_lxRcvQueue.back().end)
            {
                LinuxMemUncharge(ofo.truesize);
                continue;
            }
            std::optional<uint32_t> ofoDelta;
            if (m_lxRcvQueue.back().end == ofo.seq)
            {
                ofoDelta = LinuxCoalesce(m_lxRcvQueue.back(), ofo);
            }
            if (ofoDelta)
            {
                LinuxMemCharge(*ofoDelta);
                LinuxMemUncharge(ofo.truesize);
            }
            else
            {
                m_lxRcvQueue.push_back(ofo);
            }
        }
        return;
    }
    // tcp_data_queue_ofo()
    const bool sack = m_sackEnabled;
    auto last = m_lxOfoQueue.find(m_lxOooLast);
    if (last != m_lxOfoQueue.end() && last->second.end == seq)
    {
        if (auto delta = LinuxCoalesce(last->second, skb))
        {
            LinuxMemCharge(*delta);
            if (sack)
            {
                LinuxDataReceived(skb, adjusted, false);
            }
            return;
        }
    }
    auto next = m_lxOfoQueue.lower_bound(seq);
    if (next != m_lxOfoQueue.begin())
    {
        auto prev = std::prev(next);
        if (prev->second.end >= skb.end)
        {
            return; // All the data is present already
        }
        if (prev->second.end == seq && prev->first != m_lxOooLast)
        {
            if (auto delta = LinuxCoalesce(prev->second, skb))
            {
                LinuxMemCharge(*delta);
                if (sack)
                {
                    LinuxDataReceived(skb, adjusted, false);
                }
                return;
            }
        }
    }
    if (next != m_lxOfoQueue.end() && next->first == seq && next->second.end >= skb.end)
    {
        return;
    }
    // Inserted, the skbs it covers removed; the window grows before the skb
    // is charged
    while (next != m_lxOfoQueue.end() && next->second.end <= skb.end)
    {
        LinuxMemUncharge(next->second.truesize);
        next = m_lxOfoQueue.erase(next);
    }
    if (next == m_lxOfoQueue.end())
    {
        m_lxOooLast = seq;
    }
    if (sack)
    {
        LinuxDataReceived(skb, skb.truesize, false);
    }
    m_lxOfoQueue[seq] = skb;
    LinuxMemCharge(skb.truesize);
}

void
TcpSocketBase::LinuxMemCharge(uint32_t size)
{
    m_lxFwdAlloc -= size;
}

void
TcpSocketBase::LinuxMemUncharge(uint32_t size)
{
    // sk_mem_uncharge(), then sk_mem_reclaim() of the whole pages forward
    // allocated
    m_lxFwdAlloc += size;
    if (m_lxFwdAlloc >= LINUX_PAGE_SIZE)
    {
        int64_t pages = m_lxFwdAlloc / LINUX_PAGE_SIZE;
        m_lxFwdAlloc -= pages * LINUX_PAGE_SIZE;
        LinuxMemAllocatedAdd(-pages);
        if (m_lxMemPressure && m_lxMemAllocated < m_lxTcpMem[0])
        {
            NS_LOG_DEBUG("TCP memory pressure left");
            m_lxMemPressure = false;
        }
    }
}

void
TcpSocketBase::LinuxMemAllocatedAdd(int64_t pages)
{
    // sk_memory_allocated_add() and sk_memory_allocated_sub(): the pages
    // reach the counter of the protocol in batches of a per-CPU reserve
    m_lxMemPerCpu += pages;
    if (m_lxMemPerCpu >= LINUX_MEM_PCPU_RESERVE || m_lxMemPerCpu <= -LINUX_MEM_PCPU_RESERVE)
    {
        m_lxMemAllocated += m_lxMemPerCpu;
        m_lxMemPerCpu = 0;
    }
}

bool
TcpSocketBase::LinuxRmemSchedule(uint32_t size)
{
    // sk_rmem_schedule()
    if (m_lxTcpMem[2] == 0)
    {
        return true;
    }
    int64_t delta = static_cast<int64_t>(size) - m_lxFwdAlloc;
    if (delta <= 0)
    {
        return true;
    }
    // __sk_mem_schedule() and __sk_mem_raise_allocated()
    int64_t pages = (delta + LINUX_PAGE_SIZE - 1) / LINUX_PAGE_SIZE;
    m_lxFwdAlloc += pages * LINUX_PAGE_SIZE;
    LinuxMemAllocatedAdd(pages);
    if (m_lxMemAllocated <= m_lxTcpMem[0])
    {
        m_lxMemPressure = false;
        return true;
    }
    if (m_lxMemAllocated > m_lxTcpMem[1] && !m_lxMemPressure)
    {
        NS_LOG_DEBUG("TCP memory pressure entered at " << m_lxMemAllocated << " pages");
        m_lxMemPressure = true;
    }
    if (m_lxMemAllocated <= m_lxTcpMem[2])
    {
        const int64_t rmem = LinuxRmemAlloc();
        if (rmem < LINUX_TCP_RMEM0 || !m_lxMemPressure)
        {
            return true;
        }
        // The sockets below the average usage may still raise it
        int64_t sockets = m_tcp ? std::max<int64_t>(m_tcp->GetNSockets(), 1) : 1;
        int64_t usage = (rmem + m_lxFwdAlloc + LINUX_PAGE_SIZE - 1) / LINUX_PAGE_SIZE;
        if (m_lxTcpMem[2] > sockets * usage)
        {
            return true;
        }
    }
    LinuxMemAllocatedAdd(-pages);
    m_lxFwdAlloc -= pages * LINUX_PAGE_SIZE;
    return false;
}

void
TcpSocketBase::LinuxAdjustRcvSsThresh()
{
    // tcp_adjust_rcv_ssthresh()
    m_lxRcvSsThresh = std::min(m_lxRcvSsThresh, 4 * m_lxAdvMss);
}

bool
TcpSocketBase::LinuxPruneOfoQueue(SequenceNumber32 seq)
{
    NS_LOG_FUNCTION(this << seq);
    // tcp_prune_ofo_queue(): from the last skb, down to the one the
    // incoming segment follows, at least an eighth of the buffer at a time
    if (m_lxOfoQueue.empty())
    {
        return false;
    }
    bool pruned = false;
    int64_t goal = m_lxRcvBuf >> 3;
    auto it = m_lxOfoQueue.find(m_lxOooLast);
    if (it == m_lxOfoQueue.end())
    {
        it = std::prev(m_lxOfoQueue.end());
    }
    while (true)
    {
        if (seq > it->first)
        {
            break;
        }
        pruned = true;
        const LinuxRcvSkb skb = it->second;
        bool first = it == m_lxOfoQueue.begin();
        auto prev = first ? m_lxOfoQueue.end() : std::prev(it);
        m_lxOfoQueue.erase(it);
        while (m_tcb->m_rxBuffer->PruneLastOutOfOrder(skb.seq) > 0)
        {
        }
        NS_LOG_DEBUG("Pruned the out-of-order skb " << skb.seq << " to " << skb.end);
        goal -= skb.truesize;
        LinuxMemUncharge(skb.truesize);
        if (first)
        {
            m_lxOooLast = SequenceNumber32(0);
        }
        else
        {
            m_lxOooLast = prev->first;
        }
        if (first || goal <= 0)
        {
            if (LinuxRmemAlloc() <= m_lxRcvBuf && !m_lxMemPressure)
            {
                break;
            }
            goal = m_lxRcvBuf >> 3;
        }
        if (first)
        {
            break;
        }
        it = prev;
    }
    if (pruned)
    {
        // As Linux, the SACK blocks are forgotten: the sender learns about
        // the data dropped (reneging) at its retransmission timeout
        m_tcb->m_rxBuffer->ResetSackList();
    }
    return pruned;
}

bool
TcpSocketBase::LinuxTryRmemSchedule(SequenceNumber32 seq, uint32_t size)
{
    NS_LOG_FUNCTION(this << seq << size);
    // tcp_try_rmem_schedule()
    if (LinuxRmemAlloc() <= m_lxRcvBuf && LinuxRmemSchedule(size))
    {
        return true;
    }
    // tcp_prune_queue(): the collapse of the queues is not modelled
    if (LinuxRmemAlloc() >= m_lxRcvBuf)
    {
        // tcp_clamp_window()
        m_quickAckCount = 0;
        if (LinuxRmemAlloc() > m_lxRcvBuf)
        {
            m_lxRcvSsThresh = std::min(m_lxWindowClamp, 2 * m_lxAdvMss);
        }
    }
    else if (m_lxMemPressure)
    {
        LinuxAdjustRcvSsThresh();
    }
    if (LinuxRmemAlloc() > m_lxRcvBuf)
    {
        LinuxPruneOfoQueue(seq);
        if (LinuxRmemAlloc() > m_lxRcvBuf)
        {
            return false;
        }
    }
    while (!LinuxRmemSchedule(size))
    {
        if (!LinuxPruneOfoQueue(seq))
        {
            return false;
        }
    }
    return true;
}

int64_t
TcpSocketBase::LinuxRmemAlloc() const
{
    // The skbs read are freed; while a segment is processed, the data that
    // the application reads has not been read yet
    SequenceNumber32 read = m_lxSoftirq ? m_lxSoftirqRead : GetRxReadPoint();
    int64_t rmem = 0;
    for (const auto& skb : m_lxRcvQueue)
    {
        // The skb of a FIN without data stays until the socket is closed
        // (tcp_close() purges the receive queue)
        if (skb.seq == skb.end ? (m_lxSoftirq || m_state == CLOSE_WAIT) : skb.end > read)
        {
            rmem += skb.truesize;
        }
    }
    for (const auto& [seq, skb] : m_lxOfoQueue)
    {
        rmem += skb.truesize;
    }
    return rmem;
}

int64_t
TcpSocketBase::LinuxSpace() const
{
    return LinuxWinFromSpace(static_cast<int64_t>(m_lxRcvBuf) - LinuxRmemAlloc());
}

void
TcpSocketBase::LinuxInitWindow()
{
    NS_LOG_FUNCTION(this);
    if (m_lxInit)
    {
        return;
    }
    m_lxInit = true;
    // As SO_RCVBUF, which doubles the size it is given to account for the
    // overhead of the skbs
    m_lxRcvBuf = 2 * m_tcb->m_rxBuffer->MaxBufferSize();
    m_tcb->m_rxBuffer->SetMaxBufferSize(m_lxRcvBuf);
    m_lxRatio = 128; // TCP_DEFAULT_SCALING_RATIO
    // The MSS that the peer is to use: before the SYN of the peer is
    // processed, the segment size does not discount the timestamps yet
    m_lxAdvMss = m_tcb->m_segmentSize;
    uint32_t options = GetSegmentOptionsSize(m_timestampEnabled);
    if (m_state == CLOSED || m_state == SYN_SENT)
    {
        m_lxAdvMss = m_lxAdvMss > options ? m_lxAdvMss - options : m_lxAdvMss;
        m_lxInitAdvMss = m_lxAdvMss;
    }
    else
    {
        // The socket that a listener creates sets the bound of its window
        // with the advmss of the route, which does not discount the
        // timestamps (tcp_v4_syn_recv_sock())
        m_lxInitAdvMss = m_lxAdvMss + options;
    }
    // tcp_select_initial_window()
    const uint32_t maxClamp = 65535U << 14;
    auto space = static_cast<uint32_t>(
        std::min<int64_t>(LinuxWinFromSpace(m_lxRcvBuf), static_cast<int64_t>(maxClamp)));
    if (space > m_lxAdvMss)
    {
        space = space / m_lxAdvMss * m_lxAdvMss;
    }
    m_lxRcvWnd = space;
    m_lxWScale = 0;
    if (m_winScalingEnabled)
    {
        // Scaled for the largest buffer, at least net.core.rmem_max (212992)
        uint32_t maxSpace = std::max({space, m_lxRcvBuf, 212992U});
        int log2 = 0;
        while ((maxSpace >> (log2 + 1)) != 0)
        {
            ++log2;
        }
        m_lxWScale = static_cast<uint8_t>(std::clamp(log2 - 15, 0, 14));
    }
    m_lxWindowClamp = 65535U << m_lxWScale;
    m_lxRcvSsThresh = m_lxRcvWnd;
    NS_LOG_DEBUG("Initial window " << m_lxRcvWnd << " scale " << +m_lxWScale << " buffer "
                                   << m_lxRcvBuf);
}

void
TcpSocketBase::LinuxInitBufferSpace()
{
    NS_LOG_FUNCTION(this);
    m_lxEstablished = true;
    m_lxAdvMss = m_tcb->m_segmentSize;
    if (m_rcvWindShift == 0)
    {
        m_lxWindowClamp = std::min<uint32_t>(m_lxWindowClamp, 65535);
    }
    // tcp_init_buffer_space(), with tcp_app_win = 31
    const uint32_t advMss = m_lxInitAdvMss > 0 ? m_lxInitAdvMss : m_lxAdvMss;
    auto maxWin = static_cast<uint32_t>(LinuxWinFromSpace(m_lxRcvBuf));
    if (m_lxWindowClamp >= maxWin)
    {
        m_lxWindowClamp = maxWin;
        if (maxWin > 4 * advMss)
        {
            m_lxWindowClamp = std::max(maxWin - (maxWin >> 31), 4 * advMss);
        }
    }
    // One segment is reserved for the application
    if (m_lxWindowClamp > 2 * advMss && m_lxWindowClamp + advMss > maxWin)
    {
        m_lxWindowClamp = std::max(2 * advMss, maxWin - advMss);
    }
    m_lxRcvSsThresh = std::min(m_lxRcvSsThresh, m_lxWindowClamp);
    // tcp_initialize_rcv_mss()
    uint32_t hint = std::min({m_lxAdvMss, m_lxRcvWnd / 2, 536U});
    m_lxRcvMss = std::max(hint, 88U);
    m_lxRcvWup = m_tcb->m_rxBuffer->NextRxSequence();
    NS_LOG_DEBUG("Window bound " << m_lxWindowClamp << " rcv_ssthresh " << m_lxRcvSsThresh);
}

void
TcpSocketBase::LinuxDataReceived(const LinuxRcvSkb& skb, uint32_t truesizeAdjusted, bool inSequence)
{
    const uint32_t len = skb.end - skb.seq;
    NS_LOG_FUNCTION(this << len << truesizeAdjusted << inSequence);
    if (!m_lxEstablished)
    {
        LinuxInitBufferSpace();
    }
    if (inSequence)
    {
        // tcp_measure_rcv_mss()
        if (len >= m_lxRcvMss)
        {
            if (len != m_lxRcvMss)
            {
                auto ratio = static_cast<uint32_t>((static_cast<uint64_t>(len) << 8) /
                                                   std::max<uint32_t>(skb.truesize, 1));
                ratio = std::max<uint32_t>(ratio, 1);
                if (ratio != m_lxRatio)
                {
                    m_lxRatio = ratio;
                    m_lxWindowClamp = static_cast<uint32_t>(LinuxWinFromSpace(m_lxRcvBuf));
                }
            }
            m_lxRcvMss = std::min(len, m_lxAdvMss);
        }
        if (len < 128)
        {
            return;
        }
    }
    else if (!m_sackEnabled)
    {
        // As tcp_data_queue_ofo(), out-of-order data does not grow the window
        // of a connection without SACK, so that its duplicate ACKs, which
        // carry the same window, trigger the fast retransmit of the peer
        return;
    }
    // tcp_grow_window()
    int64_t space = LinuxSpace();
    int64_t room =
        std::min<int64_t>(m_lxWindowClamp, space) - static_cast<int64_t>(m_lxRcvSsThresh);
    if (room <= 0)
    {
        return;
    }
    if (m_lxMemPressure)
    {
        LinuxAdjustRcvSsThresh();
        return;
    }
    int64_t truesize = truesizeAdjusted;
    int64_t incr = 0;
    if (LinuxWinFromSpace(truesize) <= len)
    {
        incr = 2 * static_cast<int64_t>(m_lxAdvMss);
    }
    else
    {
        // __tcp_grow_window()
        int64_t truesizeWin = LinuxWinFromSpace(truesize) >> 1;
        int64_t window = LinuxWinFromSpace(m_lxRcvBuf) >> 1;
        while (m_lxRcvSsThresh <= window)
        {
            if (truesizeWin <= len)
            {
                incr = 2 * static_cast<int64_t>(m_lxRcvMss);
                break;
            }
            truesizeWin >>= 1;
            window >>= 1;
        }
    }
    if (incr > 0)
    {
        incr = std::max<int64_t>(incr, 2 * static_cast<int64_t>(len));
        m_lxRcvSsThresh += static_cast<uint32_t>(std::min(room, incr));
        // The segment is acknowledged at once, so that the sender learns of
        // the larger window. As Linux (icsk_ack.quick |= 1), the count is
        // or-ed with one: an even count left is not consumed while the
        // window grows, and its quick ACKs are sent once it stops growing
        if (m_quickAck)
        {
            m_quickAckCount |= 1;
        }
        NS_LOG_LOGIC("rcv_ssthresh grown to " << m_lxRcvSsThresh);
    }
}

uint32_t
TcpSocketBase::LinuxSelectWindow()
{
    NS_LOG_FUNCTION(this);
    if (!m_lxEstablished)
    {
        LinuxInitBufferSpace();
    }
    int64_t mss = m_lxRcvMss;
    int64_t freeSpace = LinuxSpace();
    int64_t allowedSpace = LinuxWinFromSpace(m_lxRcvBuf);
    int64_t fullSpace = std::min<int64_t>(m_lxWindowClamp, allowedSpace);
    if (mss > fullSpace)
    {
        mss = fullSpace;
        if (mss <= 0)
        {
            return 0;
        }
    }
    int64_t unit = int64_t{1} << m_rcvWindShift;
    if (freeSpace < (fullSpace >> 1))
    {
        // Memory is short: the quick ACKs stop
        m_quickAckCount = 0;
        if (m_lxMemPressure)
        {
            LinuxAdjustRcvSsThresh();
        }
        if (freeSpace > 0)
        {
            freeSpace = freeSpace / unit * unit;
        }
        if (freeSpace < (allowedSpace >> 4) || freeSpace < mss)
        {
            return 0;
        }
    }
    if (freeSpace > m_lxRcvSsThresh)
    {
        freeSpace = m_lxRcvSsThresh;
    }
    int64_t window;
    if (m_rcvWindShift > 0)
    {
        // Rounded up, so that the window is not scaled away
        window = (freeSpace + unit - 1) / unit * unit;
    }
    else
    {
        // A multiple of the MSS, kept if within one MSS of the free space
        window = m_lxRcvWnd;
        if (window <= freeSpace - mss || window > freeSpace)
        {
            window = freeSpace / mss * mss;
        }
        else if (mss == fullSpace && freeSpace > window + (fullSpace >> 1))
        {
            window = freeSpace;
        }
    }
    return static_cast<uint32_t>(std::max<int64_t>(window, 0));
}

uint32_t
TcpSocketBase::LinuxAdvertiseWindow()
{
    NS_LOG_FUNCTION(this);
    SequenceNumber32 rcvNxt = m_tcb->m_rxBuffer->NextRxSequence();
    if (m_lxNoMem)
    {
        // The zero window of an ACK of data that could not be queued is not
        // kept as the window advertised
        m_lxNoMem = false;
        return 0;
    }
    uint32_t newWin = LinuxSelectWindow();
    // tcp_receive_window(): what is left of the window last advertised
    auto curWin = static_cast<int32_t>(m_lxRcvWup.GetValue() + m_lxRcvWnd - rcvNxt.GetValue());
    if (curWin > 0 && newWin < static_cast<uint32_t>(curWin))
    {
        // The window is never shrunk: the right edge stays, rounded up to
        // the unit of the window scale
        uint32_t unit = 1U << m_rcvWindShift;
        newWin = (static_cast<uint32_t>(curWin) + unit - 1) / unit * unit;
    }
    NS_LOG_DEBUG("Window " << newWin << " (memory " << LinuxRmemAlloc() << ", buffered "
                           << m_tcb->m_rxBuffer->Size() << ", rcv_ssthresh " << m_lxRcvSsThresh
                           << ", left " << curWin << ")");
    m_lxRcvWnd = newWin;
    m_lxRcvWup = rcvNxt;
    return std::min(newWin, static_cast<uint32_t>(m_maxWinSize) << m_rcvWindShift);
}

/* Received a packet upon LISTEN state. */
void
TcpSocketBase::ProcessListen(Ptr<Packet> packet,
                             const TcpHeader& tcpHeader,
                             const Address& fromAddress,
                             const Address& toAddress)
{
    NS_LOG_FUNCTION(this << tcpHeader);

    // Extract the flags. PSH, URG, CWR and ECE are disregarded.
    uint8_t tcpflags =
        tcpHeader.GetFlags() & ~(TcpHeader::PSH | TcpHeader::URG | TcpHeader::CWR | TcpHeader::ECE);

    // Fork a socket if received a SYN. Do nothing otherwise.
    // C.f.: the LISTEN part in tcp_v4_do_rcv() in tcp_ipv4.c in Linux kernel
    // The way back recorded by the segment which arrived belongs to the
    // connection forked for it alone, never to the listener, whatever
    // becomes of the segment
    std::vector<Ipv4Address> returnRoute;
    if (!m_appSourceRoute)
    {
        returnRoute.swap(m_sourceRoute);
    }

    if (tcpflags != TcpHeader::SYN)
    {
        return;
    }

    // Call socket's notify function to let the server app know we got a SYN
    // If the server app refuses the connection, do nothing
    if (!NotifyConnectionRequest(fromAddress))
    {
        return;
    }
    // Clone the socket, simulate fork
    Ptr<TcpSocketBase> newSock = Fork();
    NS_LOG_LOGIC("Cloned a TcpSocketBase " << newSock);
    if (!m_appSourceRoute)
    {
        newSock->m_sourceRoute = returnRoute;
    }
    Simulator::ScheduleNow(&TcpSocketBase::CompleteFork,
                           newSock,
                           packet,
                           tcpHeader,
                           fromAddress,
                           toAddress);
}

/* Received a packet upon SYN_SENT */
void
TcpSocketBase::ProcessSynSent(Ptr<Packet> packet, const TcpHeader& tcpHeader)
{
    NS_LOG_FUNCTION(this << tcpHeader);

    // Extract the flags. PSH and URG are disregarded.
    uint8_t tcpflags = tcpHeader.GetFlags() & ~(TcpHeader::PSH | TcpHeader::URG);

    if (tcpflags & TcpHeader::RST)
    {
        // A reset is acceptable in SYN-SENT only if it acknowledges the SYN,
        // in which case the connection was refused; otherwise it is dropped
        // (RFC 9293, Section 3.10.7.3)
        if ((tcpflags & TcpHeader::ACK) &&
            tcpHeader.GetAckNumber() == m_tcb->m_nextTxSequence + SequenceNumber32(1))
        {
            NS_LOG_LOGIC("RST received in SYN_SENT: connection refused");
            m_errno = ERROR_NOTCONN;
            NotifyConnectionFailed();
            m_closeNotified = true;
            CloseAndNotify();
        }
        else
        {
            NS_LOG_LOGIC("Dropping a RST which does not acknowledge the SYN");
        }
        return;
    }

    if (tcpflags == 0)
    { // Bare data, accept it and move to ESTABLISHED state. This is not a normal behaviour. Remove
      // this?
        NS_LOG_DEBUG("SYN_SENT -> ESTABLISHED");
        m_congestionControl->CongestionStateSet(m_tcb, TcpSocketState::CA_OPEN);
        m_tcb->m_congState = TcpSocketState::CA_OPEN;
        m_state = ESTABLISHED;
        m_connected = true;
        m_retxEvent.Cancel();
        m_delAckCount = m_delAckMaxCount;
        ReceivedData(packet, tcpHeader);
        Simulator::ScheduleNow(&TcpSocketBase::ConnectionSucceeded, this);
    }
    else if (tcpflags & TcpHeader::ACK && !(tcpflags & TcpHeader::SYN))
    { // Ignore ACK in SYN_SENT
    }
    else if (tcpflags & TcpHeader::SYN && !(tcpflags & TcpHeader::ACK))
    { // Received SYN, move to SYN_RCVD state and respond with SYN+ACK
        NS_LOG_DEBUG("SYN_SENT -> SYN_RCVD");
        m_state = SYN_RCVD;
        // The SYN+ACK takes over from the SYN (RFC 9293, Section 3.10.7.3,
        // and Figure 7): the timer of the SYN is cancelled, so that the
        // SYN+ACK arms its own and is the segment retransmitted. The SYN+ACK
        // draws from the retries the SYN left, but is always sent once.
        m_retxEvent.Cancel();
        m_synCount = std::max(m_synCount, 1U);
        m_tcb->m_rxBuffer->SetNextRxSequence(tcpHeader.GetSequenceNumber() + SequenceNumber32(1));
        // The marks compared with the sequence numbers of the peer in the
        // wrap aware arithmetic start from its initial sequence number
        m_ecnCESeq = tcpHeader.GetSequenceNumber();
        m_highRxMark = tcpHeader.GetSequenceNumber();
        m_highTxAck = tcpHeader.GetSequenceNumber();
        /* Check if we received an ECN SYN packet. Change the ECN state of receiver to ECN_IDLE if
         * the traffic is ECN capable and sender has sent ECN SYN packet
         */

        if (m_tcb->m_useEcn != TcpSocketState::Off &&
            (tcpflags & (TcpHeader::CWR | TcpHeader::ECE)) == (TcpHeader::CWR | TcpHeader::ECE))
        {
            NS_LOG_INFO("Received ECN SYN packet");
            SendEmptyPacket(TcpHeader::SYN | TcpHeader::ACK | TcpHeader::ECE);
            NS_LOG_DEBUG(TcpSocketState::EcnStateName[m_tcb->m_ecnState] << " -> ECN_IDLE");
            m_tcb->m_ecnState = TcpSocketState::ECN_IDLE;
        }
        else
        {
            m_tcb->m_ecnState = TcpSocketState::ECN_DISABLED;
            SendEmptyPacket(TcpHeader::SYN | TcpHeader::ACK);
        }
    }
    else if (tcpflags & (TcpHeader::SYN | TcpHeader::ACK) &&
             m_tcb->m_nextTxSequence + SequenceNumber32(1) == tcpHeader.GetAckNumber())
    { // Handshake completed
        NS_LOG_DEBUG("SYN_SENT -> ESTABLISHED");
        m_congestionControl->CongestionStateSet(m_tcb, TcpSocketState::CA_OPEN);
        m_tcb->m_congState = TcpSocketState::CA_OPEN;
        m_state = ESTABLISHED;
        m_connected = true;
        m_retxEvent.Cancel();
        m_tcb->m_rxBuffer->SetNextRxSequence(tcpHeader.GetSequenceNumber() + SequenceNumber32(1));
        // See the simultaneous open above
        m_ecnCESeq = tcpHeader.GetSequenceNumber();
        m_highRxMark = tcpHeader.GetSequenceNumber();
        m_highTxAck = tcpHeader.GetSequenceNumber();
        m_tcb->m_highTxMark = ++m_tcb->m_nextTxSequence;
        SetTxHeadSequence(m_tcb->m_nextTxSequence);
        // Before sending packets, update the pacing rate based on RTT measurement so far
        UpdatePacingRate();
        SendEmptyPacket(TcpHeader::ACK);

        /* Check if we received an ECN SYN-ACK packet. Change the ECN state of sender to ECN_IDLE if
         * receiver has sent an ECN SYN-ACK packet and the  traffic is ECN Capable
         */
        if (m_tcb->m_useEcn != TcpSocketState::Off &&
            (tcpflags & (TcpHeader::CWR | TcpHeader::ECE)) == (TcpHeader::ECE))
        {
            NS_LOG_INFO("Received ECN SYN-ACK packet.");
            NS_LOG_DEBUG(TcpSocketState::EcnStateName[m_tcb->m_ecnState] << " -> ECN_IDLE");
            m_tcb->m_ecnState = TcpSocketState::ECN_IDLE;
        }
        else
        {
            m_tcb->m_ecnState = TcpSocketState::ECN_DISABLED;
        }
        SendPendingData(m_connected);
        Simulator::ScheduleNow(&TcpSocketBase::ConnectionSucceeded, this);
        // Always respond to first data packet to speed up the connection.
        // Remove to get the behaviour of old NS-3 code.
        m_delAckCount = m_delAckMaxCount;
    }
    else
    { // Other in-sequence input
        if (!(tcpflags & TcpHeader::RST))
        { // When (1) rx of FIN+ACK; (2) rx of FIN; (3) rx of bad flags
            NS_LOG_LOGIC("Illegal flag combination "
                         << TcpHeader::FlagsToString(tcpHeader.GetFlags())
                         << " received in SYN_SENT. Reset packet is sent.");
            SendRST();
        }
        CloseAndNotify();
    }
}

/* Received a packet upon SYN_RCVD */
void
TcpSocketBase::ProcessSynRcvd(Ptr<Packet> packet,
                              const TcpHeader& tcpHeader,
                              const Address& fromAddress,
                              const Address& /* toAddress */)
{
    NS_LOG_FUNCTION(this << tcpHeader);

    // Extract the flags. PSH, URG, CWR and ECE are disregarded.
    uint8_t tcpflags =
        tcpHeader.GetFlags() & ~(TcpHeader::PSH | TcpHeader::URG | TcpHeader::CWR | TcpHeader::ECE);

    // A SYN+ACK acknowledging our SYN completes a simultaneous open (RFC 9293,
    // Section 3.5, MUST-10): its SYN repeats the one which brought us here,
    // and only an endpoint which opened the connection actively can receive it
    bool simultaneousOpen =
        m_activeOpen && tcpflags == (TcpHeader::SYN | TcpHeader::ACK) &&
        m_tcb->m_nextTxSequence + SequenceNumber32(1) == tcpHeader.GetAckNumber() &&
        tcpHeader.GetSequenceNumber() + SequenceNumber32(1) == m_tcb->m_rxBuffer->NextRxSequence();

    if (tcpflags == 0 || simultaneousOpen ||
        (tcpflags == TcpHeader::ACK &&
         m_tcb->m_nextTxSequence + SequenceNumber32(1) == tcpHeader.GetAckNumber()))
    { // If it is bare data, accept it and move to ESTABLISHED state. This is
        // possibly due to ACK lost in 3WHS. If in-sequence ACK is received, the
        // handshake is completed nicely.
        NS_LOG_DEBUG("SYN_RCVD -> ESTABLISHED");
        m_congestionControl->CongestionStateSet(m_tcb, TcpSocketState::CA_OPEN);
        m_tcb->m_congState = TcpSocketState::CA_OPEN;
        m_state = ESTABLISHED;
        m_connected = true;
        m_retxEvent.Cancel();
        m_tcb->m_highTxMark = ++m_tcb->m_nextTxSequence;
        SetTxHeadSequence(m_tcb->m_nextTxSequence);
        if (m_endPoint)
        {
            m_endPoint->SetPeer(InetSocketAddress::ConvertFrom(fromAddress).GetIpv4(),
                                InetSocketAddress::ConvertFrom(fromAddress).GetPort());
        }
        else if (m_endPoint6)
        {
            m_endPoint6->SetPeer(Inet6SocketAddress::ConvertFrom(fromAddress).GetIpv6(),
                                 Inet6SocketAddress::ConvertFrom(fromAddress).GetPort());
        }
        // Always respond to first data packet to speed up the connection.
        // Remove to get the behaviour of old NS-3 code.
        m_delAckCount = m_delAckMaxCount;
        if (m_activeOpen)
        {
            // The connection was opened actively, so the application which
            // called Connect() is the one to notify (RFC 9293, Section 3.5,
            // MUST-11)
            NotifyConnectionSucceeded();
        }
        else
        {
            NotifyNewConnectionCreated(this, fromAddress);
        }
        // Once ESTABLISHED, the urgent field of the segment completing the
        // handshake is processed before its data (RFC 9293, Section
        // 3.10.7.4, MUST-32): DoForwardUp skipped it in SYN-RECEIVED
        if (tcpHeader.GetFlags() & TcpHeader::URG)
        {
            ProcessUrgentPointer(tcpHeader);
        }
        ReceivedAck(packet, tcpHeader);
        // Update the pacing rate based on RTT measurement so far
        UpdatePacingRate();
        // As this connection is established, the socket is available to send data now
        if (GetTxAvailable() > 0)
        {
            NotifySend(GetTxAvailable());
        }
    }
    else if (tcpflags == TcpHeader::SYN)
    { // Probably the peer lost my SYN+ACK
        m_tcb->m_rxBuffer->SetNextRxSequence(tcpHeader.GetSequenceNumber() + SequenceNumber32(1));
        /* Check if we received an ECN SYN packet. Change the ECN state of receiver to ECN_IDLE if
         * sender has sent an ECN SYN packet and the  traffic is ECN Capable
         */
        if (m_tcb->m_useEcn != TcpSocketState::Off &&
            (tcpHeader.GetFlags() & (TcpHeader::CWR | TcpHeader::ECE)) ==
                (TcpHeader::CWR | TcpHeader::ECE))
        {
            NS_LOG_INFO("Received ECN SYN packet");
            SendEmptyPacket(TcpHeader::SYN | TcpHeader::ACK | TcpHeader::ECE);
            NS_LOG_DEBUG(TcpSocketState::EcnStateName[m_tcb->m_ecnState] << " -> ECN_IDLE");
            m_tcb->m_ecnState = TcpSocketState::ECN_IDLE;
        }
        else
        {
            m_tcb->m_ecnState = TcpSocketState::ECN_DISABLED;
            SendEmptyPacket(TcpHeader::SYN | TcpHeader::ACK);
        }
    }
    else if (tcpflags == (TcpHeader::FIN | TcpHeader::ACK))
    {
        if (tcpHeader.GetSequenceNumber() == m_tcb->m_rxBuffer->NextRxSequence())
        { // In-sequence FIN before connection complete. Set up connection and close.
            m_connected = true;
            m_retxEvent.Cancel();
            m_tcb->m_highTxMark = ++m_tcb->m_nextTxSequence;
            SetTxHeadSequence(m_tcb->m_nextTxSequence);
            if (m_endPoint)
            {
                m_endPoint->SetPeer(InetSocketAddress::ConvertFrom(fromAddress).GetIpv4(),
                                    InetSocketAddress::ConvertFrom(fromAddress).GetPort());
            }
            else if (m_endPoint6)
            {
                m_endPoint6->SetPeer(Inet6SocketAddress::ConvertFrom(fromAddress).GetIpv6(),
                                     Inet6SocketAddress::ConvertFrom(fromAddress).GetPort());
            }
            if (m_activeOpen)
            {
                // The application which called Connect() is the one to
                // notify (RFC 9293, Section 3.5, MUST-11)
                NotifyConnectionSucceeded();
            }
            else
            {
                NotifyNewConnectionCreated(this, fromAddress);
            }
            if (tcpHeader.GetFlags() & TcpHeader::URG)
            {
                ProcessUrgentPointer(tcpHeader);
            }
            PeerClose(packet, tcpHeader);
        }
    }
    else
    { // Other in-sequence input
        if (tcpflags == TcpHeader::RST && m_activeOpen)
        {
            // The connection came from SYN-SENT: a RST means it was refused,
            // which the application which called Connect() is told through
            // the connection failure callback (RFC 9293, Section 3.10.7.4)
            NS_LOG_LOGIC("RST received in SYN_RCVD after an active open: connection refused");
            m_errno = ERROR_NOTCONN;
            NotifyConnectionFailed();
            m_closeNotified = true;
            CloseAndNotify();
            return;
        }
        if (tcpflags != TcpHeader::RST)
        { // When (1) rx of a SYN+ACK not acknowledging our SYN; (2) rx of FIN;
            // (3) rx of bad flags
            NS_LOG_LOGIC("Illegal flag " << TcpHeader::FlagsToString(tcpflags)
                                         << " received. Reset packet is sent.");
            if (m_endPoint)
            {
                m_endPoint->SetPeer(InetSocketAddress::ConvertFrom(fromAddress).GetIpv4(),
                                    InetSocketAddress::ConvertFrom(fromAddress).GetPort());
            }
            else if (m_endPoint6)
            {
                m_endPoint6->SetPeer(Inet6SocketAddress::ConvertFrom(fromAddress).GetIpv6(),
                                     Inet6SocketAddress::ConvertFrom(fromAddress).GetPort());
            }
            SendRST();
        }
        CloseAndNotify();
    }
}

/* Received a packet upon CLOSE_WAIT, FIN_WAIT_1, or FIN_WAIT_2 states */
void
TcpSocketBase::ProcessWait(Ptr<Packet> packet, const TcpHeader& tcpHeader)
{
    NS_LOG_FUNCTION(this << tcpHeader);

    // Extract the flags. PSH, URG, CWR and ECE are disregarded.
    uint8_t tcpflags =
        tcpHeader.GetFlags() & ~(TcpHeader::PSH | TcpHeader::URG | TcpHeader::CWR | TcpHeader::ECE);

    if (packet->GetSize() > 0 && !(tcpflags & TcpHeader::ACK))
    { // Bare data, accept it
        ReceivedData(packet, tcpHeader);
    }
    else if (tcpflags == TcpHeader::ACK)
    { // Process the ACK, and if in FIN_WAIT_1, conditionally move to FIN_WAIT_2
        ReceivedAck(packet, tcpHeader);
        if (m_state == FIN_WAIT_1 && m_txBuffer->Size() == 0 &&
            tcpHeader.GetAckNumber() == m_tcb->m_highTxMark + SequenceNumber32(1))
        { // This ACK corresponds to the FIN sent
            NS_LOG_DEBUG("FIN_WAIT_1 -> FIN_WAIT_2");
            m_state = FIN_WAIT_2;
        }
    }
    else if (tcpflags == TcpHeader::FIN || tcpflags == (TcpHeader::FIN | TcpHeader::ACK))
    { // Got FIN, respond with ACK and move to next state
        if (tcpflags & TcpHeader::ACK)
        { // Process the ACK first
            ReceivedAck(packet, tcpHeader);
        }
        m_tcb->m_rxBuffer->SetFinSequence(tcpHeader.GetSequenceNumber());
    }
    else if (tcpflags == TcpHeader::SYN || tcpflags == (TcpHeader::SYN | TcpHeader::ACK))
    { // Duplicated SYN or SYN+ACK, possibly due to spurious retransmission
        return;
    }
    else
    { // This is a RST or bad flags
        if (tcpflags != TcpHeader::RST)
        {
            NS_LOG_LOGIC("Illegal flag " << TcpHeader::FlagsToString(tcpflags)
                                         << " received. Reset packet is sent.");
            SendRST();
        }
        else
        {
            NotifyReset();
        }
        CloseAndNotify();
        return;
    }

    // Check if the close responder sent an in-sequence FIN, if so, respond ACK
    if ((m_state == FIN_WAIT_1 || m_state == FIN_WAIT_2) && m_tcb->m_rxBuffer->Finished())
    {
        if (m_state == FIN_WAIT_1)
        {
            NS_LOG_DEBUG("FIN_WAIT_1 -> CLOSING");
            m_state = CLOSING;
            if (m_txBuffer->Size() == 0 &&
                tcpHeader.GetAckNumber() == m_tcb->m_highTxMark + SequenceNumber32(1))
            { // This ACK corresponds to the FIN sent
                TimeWait();
            }
        }
        else if (m_state == FIN_WAIT_2)
        {
            TimeWait();
        }
        SendEmptyPacket(TcpHeader::ACK);
        if (!m_shutdownRecv)
        {
            NotifyDataRecv();
        }
    }
}

/* Received a packet upon CLOSING */
void
TcpSocketBase::ProcessClosing(Ptr<Packet> packet, const TcpHeader& tcpHeader)
{
    NS_LOG_FUNCTION(this << tcpHeader);

    // Extract the flags. PSH and URG are disregarded.
    uint8_t tcpflags = tcpHeader.GetFlags() & ~(TcpHeader::PSH | TcpHeader::URG);

    if (tcpflags == TcpHeader::ACK)
    {
        if (tcpHeader.GetSequenceNumber() == m_tcb->m_rxBuffer->NextRxSequence())
        { // This ACK corresponds to the FIN sent
            TimeWait();
        }
    }
    else
    { // CLOSING state means simultaneous close, i.e. no one is sending data to
        // anyone. If anything other than ACK is received, respond with a reset.
        if (tcpflags == TcpHeader::FIN || tcpflags == (TcpHeader::FIN | TcpHeader::ACK))
        { // FIN from the peer as well. We can close immediately.
            SendEmptyPacket(TcpHeader::ACK);
        }
        else if (tcpflags != TcpHeader::RST)
        { // Receive of SYN or SYN+ACK or bad flags or pure data
            NS_LOG_LOGIC("Illegal flag " << TcpHeader::FlagsToString(tcpflags)
                                         << " received. Reset packet is sent.");
            SendRST();
        }
        CloseAndNotify();
    }
}

/* Received a packet upon LAST_ACK */
void
TcpSocketBase::ProcessLastAck(Ptr<Packet> packet, const TcpHeader& tcpHeader)
{
    NS_LOG_FUNCTION(this << tcpHeader);

    // Extract the flags. PSH and URG are disregarded.
    uint8_t tcpflags = tcpHeader.GetFlags() & ~(TcpHeader::PSH | TcpHeader::URG);

    if (tcpflags == 0)
    {
        ReceivedData(packet, tcpHeader);
    }
    else if (tcpflags == TcpHeader::ACK)
    {
        if (tcpHeader.GetSequenceNumber() == m_tcb->m_rxBuffer->NextRxSequence())
        { // This ACK corresponds to the FIN sent. This socket closed peacefully.
            CloseAndNotify();
        }
    }
    else if (tcpflags == TcpHeader::FIN)
    { // Received FIN again, the peer probably lost the FIN+ACK
        SendEmptyPacket(TcpHeader::FIN | TcpHeader::ACK);
    }
    else if (tcpflags == (TcpHeader::FIN | TcpHeader::ACK) || tcpflags == TcpHeader::RST)
    {
        CloseAndNotify();
    }
    else
    { // Received a SYN or SYN+ACK or bad flags
        NS_LOG_LOGIC("Illegal flag " << TcpHeader::FlagsToString(tcpflags)
                                     << " received. Reset packet is sent.");
        SendRST();
        CloseAndNotify();
    }
}

/* Peer sent me a FIN. Remember its sequence in rx buffer. */
void
TcpSocketBase::PeerClose(Ptr<Packet> p, const TcpHeader& tcpHeader)
{
    NS_LOG_FUNCTION(this << tcpHeader);

    // Ignore all out of range packets
    if (tcpHeader.GetSequenceNumber() < m_tcb->m_rxBuffer->NextRxSequence() ||
        tcpHeader.GetSequenceNumber() > m_tcb->m_rxBuffer->MaxRxSequence())
    {
        return;
    }
    // For any case, remember the FIN position in rx buffer first
    m_tcb->m_rxBuffer->SetFinSequence(tcpHeader.GetSequenceNumber() +
                                      SequenceNumber32(p->GetSize()));
    NS_LOG_LOGIC("Accepted FIN at seq "
                 << tcpHeader.GetSequenceNumber() + SequenceNumber32(p->GetSize()));
    // If there is any piggybacked data, process it
    if (p->GetSize())
    {
        ReceivedData(p, tcpHeader);
    }
    // Return if FIN is out of sequence, otherwise move to CLOSE_WAIT state by DoPeerClose
    if (!m_tcb->m_rxBuffer->Finished())
    {
        return;
    }

    // Simultaneous close: Application invoked Close() when we are processing this FIN packet
    if (m_state == FIN_WAIT_1)
    {
        NS_LOG_DEBUG("FIN_WAIT_1 -> CLOSING");
        m_state = CLOSING;
        return;
    }

    if (m_linuxRcvWindow && m_lxInit && p->GetSize() == 0)
    {
        // A FIN without data is queued as an skb of its own (copied into
        // its head), held until the socket is closed
        const uint32_t ipLen = tcpHeader.GetSerializedSize() + (m_endPoint6 != nullptr ? 40 : 20);
        m_lxRcvQueue.push_back(LinuxNewSkb(tcpHeader.GetSequenceNumber(), 0, ipLen));
        m_lxSoftirq = true;
        m_lxSoftirqRead = GetRxReadPoint();
        DoPeerClose(true);
        m_lxSoftirq = false;
        return;
    }
    DoPeerClose(p->GetSize() == 0); // Change state, respond with ACK
}

/* Received a in-sequence FIN. Close down this socket. */
void
TcpSocketBase::DoPeerClose(bool finSegment)
{
    NS_ASSERT(m_state == ESTABLISHED || m_state == SYN_RCVD || m_state == FIN_WAIT_1 ||
              m_state == FIN_WAIT_2);

    // Move the state to CLOSE_WAIT
    NS_LOG_DEBUG(TcpStateName[m_state] << " -> CLOSE_WAIT");
    m_state = CLOSE_WAIT;

    if (!m_closeNotified)
    {
        // The normal behaviour for an application is that, when the peer sent a in-sequence
        // FIN, the app should prepare to close. The app has two choices at this point: either
        // respond with ShutdownSend() call to declare that it has nothing more to send and
        // the socket can be closed immediately; or remember the peer's close request, wait
        // until all its existing data are pushed into the TCP socket, then call Close()
        // explicitly.
        NS_LOG_LOGIC("TCP " << this << " calling NotifyNormalClose");
        NotifyNormalClose();
        m_closeNotified = true;
    }
    if (m_shutdownSend)
    { // The application declares that it would not sent any more, close this socket
        Close();
    }
    else if (finSegment && m_adaptiveDelAck && !m_delAckPending &&
             m_tcb->m_rxBuffer->Size() == m_tcb->m_rxBuffer->Available())
    {
        // As tcp_fin() of Linux: the interactive mode starts, in which no
        // ACK is sent quickly, and the FIN, which does not advance RCV.NXT
        // by more than one segment, is acknowledged on the delayed ACK timer
        m_pingPong = true;
        m_delAckEvent.Cancel();
        m_delAckEvent = Simulator::Schedule(GetDelAckDelay(), &TcpSocketBase::DelAckTimeout, this);
        m_delAckPending = true;
        NS_LOG_LOGIC("FIN acknowledged in " << Simulator::GetDelayLeft(m_delAckEvent).As(Time::MS));
    }
    else
    { // Need to ack, the application will close later
        SendEmptyPacket(TcpHeader::ACK);
    }
    if (m_state == LAST_ACK)
    {
        m_dataRetrCount = m_dataRetries; // prevent endless FINs
        NS_LOG_LOGIC("TcpSocketBase " << this << " scheduling LATO1");
        Time lastRto = m_rtt->GetEstimate() + Max(m_clockGranularity, m_rtt->GetVariation() * 4);
        m_lastAckEvent = Simulator::Schedule(lastRto, &TcpSocketBase::LastAckTimeout, this);
    }
}

/* Kill this socket. This is a callback function configured to m_endpoint in
   SetupCallback(), invoked when the endpoint is destroyed. */
void
TcpSocketBase::Destroy()
{
    NS_LOG_FUNCTION(this);
    m_endPoint = nullptr;
    if (m_tcp)
    {
        m_tcp->RemoveSocket(this);
    }
    NS_LOG_LOGIC(this << " Cancelled ReTxTimeout event which was set to expire at "
                      << (Simulator::Now() + Simulator::GetDelayLeft(m_retxEvent)).GetSeconds());
    CancelAllTimers();
}

/* Kill this socket. This is a callback function configured to m_endpoint in
   SetupCallback(), invoked when the endpoint is destroyed. */
void
TcpSocketBase::Destroy6()
{
    NS_LOG_FUNCTION(this);
    m_endPoint6 = nullptr;
    if (m_tcp)
    {
        m_tcp->RemoveSocket(this);
    }
    NS_LOG_LOGIC(this << " Cancelled ReTxTimeout event which was set to expire at "
                      << (Simulator::Now() + Simulator::GetDelayLeft(m_retxEvent)).GetSeconds());
    CancelAllTimers();
}

/* Send an empty packet with specified TCP flags */
void
TcpSocketBase::SendEmptyPacket(uint8_t flags)
{
    NS_LOG_FUNCTION(this << static_cast<uint32_t>(flags));

    if (m_endPoint == nullptr && m_endPoint6 == nullptr)
    {
        NS_LOG_WARN("Failed to send empty packet due to null endpoint");
        return;
    }

    Ptr<Packet> p = Create<Packet>();
    TcpHeader header;
    SequenceNumber32 s = m_tcb->m_nextTxSequence;
    TcpPacketType_t packetType = INVALID;

    if (flags & TcpHeader::FIN)
    {
        packetType = TcpPacketType_t::FIN;
        flags |= TcpHeader::ACK;
    }
    else if (!(flags & TcpHeader::SYN))
    {
        // A segment without data carries SND.NXT, the highest sequence number
        // sent so far: a retransmission rewinds m_nextTxSequence below the
        // RCV.NXT of the peer, which would find the segment out of its window
        // and drop it with the acknowledgment it carries (RFC 9293, Section
        // 3.10.7.4)
        s = std::max(s, m_tcb->m_highTxMark.Get());
        if (m_state == FIN_WAIT_1 || m_state == LAST_ACK || m_state == CLOSING)
        {
            ++s;
        }
    }

    if (flags & TcpHeader::SYN)
    {
        packetType = TcpPacketType_t::SYN;
        if (flags & TcpHeader::ACK)
        {
            packetType = TcpPacketType_t::SYN_ACK;
        }
    }
    else if (flags & TcpHeader::ACK)
    {
        packetType = TcpPacketType_t::PURE_ACK;
    }

    if (flags & TcpHeader::RST)
    {
        packetType = TcpPacketType_t::RST;
    }

    NS_ASSERT_MSG(packetType != TcpPacketType_t::INVALID, "Invalid TCP packet type");
    AddSocketTags(p, IsEct(packetType));

    header.SetFlags(flags);
    header.SetSequenceNumber(s);
    header.SetAckNumber(m_tcb->m_rxBuffer->NextRxSequence());
    if (m_endPoint != nullptr)
    {
        header.SetSourcePort(m_endPoint->GetLocalPort());
        header.SetDestinationPort(m_endPoint->GetPeerPort());
    }
    else
    {
        header.SetSourcePort(m_endPoint6->GetLocalPort());
        header.SetDestinationPort(m_endPoint6->GetPeerPort());
    }
    AddOptions(header);

    // RFC 6298, clause 2.4
    m_rto =
        Max(m_rtt->GetEstimate() + Max(m_clockGranularity, m_rtt->GetVariation() * 4), m_minRto);

    uint16_t windowSize = AdvertisedWindowSize();
    bool hasSyn = flags & TcpHeader::SYN;
    bool hasFin = flags & TcpHeader::FIN;
    bool isAck = flags == TcpHeader::ACK;
    if (hasSyn)
    {
        // The MSS option should be sent in every SYN segment when the receive
        // MSS differs from the default, and may be sent always (RFC 9293,
        // Section 3.7.1, SHLD-5 and MAY-3): always send it
        AddOptionMss(header);

        if (m_winScalingEnabled)
        { // The window scaling option is set only on SYN packets
            AddOptionWScale(header);
        }

        if (m_sackEnabled)
        {
            AddOptionSackPermitted(header);
        }

        if (m_synCount == 0)
        { // No more connection retries, give up
            NS_LOG_LOGIC("Connection failed.");
            m_rtt->Reset(); // According to recommendation -> RFC 6298
            NotifyConnectionFailed();
            m_state = CLOSED;
            DeallocateEndPoint();
            return;
        }
        else
        { // Exponential backoff of connection time out
            int backoffCount = 0x1 << (m_synRetries - m_synCount);
            m_rto = m_cnTimeout * backoffCount;
            m_synCount--;
        }

        if (m_synRetries - 1 == m_synCount)
        {
            UpdateRttHistory(s, 0, false);
        }
        else
        { // This is SYN retransmission
            UpdateRttHistory(s, 0, true);
        }

        windowSize = AdvertisedWindowSize(false);
    }
    header.SetWindowSize(windowSize);

    if (flags & TcpHeader::ACK)
    { // If sending an ACK, cancel the delay ACK as well
        StopDelAckTimer();
        m_ackNow = false;
        m_delAckCount = 0;
        QuickAckSent();
        m_sackCompressionEvent.Cancel();
        m_compressedAcks = 0;
        if (m_highTxAck < header.GetAckNumber())
        {
            m_highTxAck = header.GetAckNumber();
        }
        if (m_sackEnabled && m_tcb->m_rxBuffer->GetSackListSize() > 0)
        {
            AddOptionSack(header);
        }
        NS_LOG_INFO("Sending a pure ACK, acking seq " << m_tcb->m_rxBuffer->NextRxSequence());
    }

    m_txTrace(p, header, this);

    if (m_endPoint != nullptr)
    {
        m_tcp->SendPacket(p,
                          header,
                          m_endPoint->GetLocalAddress(),
                          m_endPoint->GetPeerAddress(),
                          m_boundnetdevice);
    }
    else
    {
        m_tcp->SendPacket(p,
                          header,
                          m_endPoint6->GetLocalAddress(),
                          m_endPoint6->GetPeerAddress(),
                          m_boundnetdevice);
    }

    if (m_retxEvent.IsExpired() && (hasSyn || hasFin) && !isAck)
    { // Retransmit SYN / SYN+ACK / FIN / FIN+ACK to guard against lost
        NS_LOG_LOGIC("Schedule retransmission timeout at time "
                     << Simulator::Now().GetSeconds() << " to expire at time "
                     << (Simulator::Now() + m_rto.Get()).GetSeconds());
        m_retxEvent =
            Simulator::Schedule(GetRtoTimerDelay(), &TcpSocketBase::SendEmptyPacket, this, flags);
    }
}

/* This function closes the endpoint completely. Called upon RST_TX action. */
void
TcpSocketBase::SendRST()
{
    NS_LOG_FUNCTION(this);
    SendEmptyPacket(TcpHeader::RST);
    if (m_state != CLOSED)
    {
        // The application may call Close() from the callback: the connection
        // is CLOSED first, so that it is neither reset nor notified again
        NS_LOG_DEBUG(TcpStateName[m_state] << " -> CLOSED");
        m_state = CLOSED;
        m_closeNotified = true;
        NotifyErrorClose();
    }
    DeallocateEndPoint();
}

/* Deallocate the end point and cancel all the timers */
void
TcpSocketBase::DeallocateEndPoint()
{
    // note: it shouldn't be necessary to invalidate the callback and manually call
    // TcpL4Protocol::RemoveSocket. Alas, if one relies on the endpoint destruction
    // callback, there's a weird memory access to a free'd area. Harmless, but valgrind
    // considers it an error.

    if (m_endPoint != nullptr)
    {
        CancelAllTimers();
        m_endPoint->SetDestroyCallback(MakeNullCallback<void>());
        m_tcp->DeAllocate(m_endPoint);
        m_endPoint = nullptr;
        m_tcp->RemoveSocket(this);
    }
    else if (m_endPoint6 != nullptr)
    {
        CancelAllTimers();
        m_endPoint6->SetDestroyCallback(MakeNullCallback<void>());
        m_tcp->DeAllocate(m_endPoint6);
        m_endPoint6 = nullptr;
        m_tcp->RemoveSocket(this);
    }
}

/* Configure the endpoint to a local address. Called by Connect() if Bind() didn't specify one. */
int
TcpSocketBase::SetupEndpoint()
{
    NS_LOG_FUNCTION(this);
    Ptr<Ipv4> ipv4 = m_node->GetObject<Ipv4>();
    NS_ASSERT(ipv4);
    if (!ipv4->GetRoutingProtocol())
    {
        NS_FATAL_ERROR("No Ipv4RoutingProtocol in the node");
    }
    // Create a dummy packet, then ask the routing function for the best output
    // interface's address
    Ipv4Header header;
    header.SetDestination(m_endPoint->GetPeerAddress());
    Socket::SocketErrno errno_;
    Ptr<Ipv4Route> route;
    Ptr<NetDevice> oif = m_boundnetdevice;
    route = ipv4->GetRoutingProtocol()->RouteOutput(Ptr<Packet>(), header, oif, errno_);
    if (!route)
    {
        NS_LOG_LOGIC("Route to " << m_endPoint->GetPeerAddress() << " does not exist");
        NS_LOG_ERROR(errno_);
        m_errno = errno_;
        return -1;
    }
    NS_LOG_LOGIC("Route exists");
    if (m_endPoint->GetLocalAddress() == Ipv4Address::GetAny())
    {
        // The application did not specify a local address, so the IP layer is
        // asked to select one (RFC 9293, Section 3.9.1.1, MUST-44). Otherwise
        // the address it bound is the one to use (MUST-43 and MUST-45)
        m_endPoint->SetLocalAddress(route->GetSource());
    }
    return 0;
}

int
TcpSocketBase::SetupEndpoint6()
{
    NS_LOG_FUNCTION(this);
    Ptr<Ipv6L3Protocol> ipv6 = m_node->GetObject<Ipv6L3Protocol>();
    NS_ASSERT(ipv6);
    if (!ipv6->GetRoutingProtocol())
    {
        NS_FATAL_ERROR("No Ipv6RoutingProtocol in the node");
    }
    // Create a dummy packet, then ask the routing function for the best output
    // interface's address
    Ipv6Header header;
    header.SetDestination(m_endPoint6->GetPeerAddress());
    Socket::SocketErrno errno_;
    Ptr<Ipv6Route> route;
    Ptr<NetDevice> oif = m_boundnetdevice;
    route = ipv6->GetRoutingProtocol()->RouteOutput(Ptr<Packet>(), header, oif, errno_);
    if (!route)
    {
        NS_LOG_LOGIC("Route to " << m_endPoint6->GetPeerAddress() << " does not exist");
        NS_LOG_ERROR(errno_);
        m_errno = errno_;
        return -1;
    }
    NS_LOG_LOGIC("Route exists");
    if (m_endPoint6->GetLocalAddress() == Ipv6Address::GetAny())
    {
        // See the IPv4 variant: RFC 9293, Section 3.9.1.1, MUST-43 to MUST-45
        m_endPoint6->SetLocalAddress(route->GetSource());
    }
    return 0;
}

/* This function is called only if a SYN received in LISTEN state. After
   TcpSocketBase cloned, allocate a new end point to handle the incoming
   connection and send a SYN+ACK to complete the handshake. */
void
TcpSocketBase::CompleteFork(Ptr<Packet> p [[maybe_unused]],
                            const TcpHeader& h,
                            const Address& fromAddress,
                            const Address& toAddress)
{
    NS_LOG_FUNCTION(this << p << h << fromAddress << toAddress);
    // Get port and address from peer (connecting host)
    if (InetSocketAddress::IsMatchingType(toAddress))
    {
        m_endPoint = m_tcp->Allocate(GetBoundNetDevice(),
                                     InetSocketAddress::ConvertFrom(toAddress).GetIpv4(),
                                     InetSocketAddress::ConvertFrom(toAddress).GetPort(),
                                     InetSocketAddress::ConvertFrom(fromAddress).GetIpv4(),
                                     InetSocketAddress::ConvertFrom(fromAddress).GetPort());
        m_endPoint6 = nullptr;
    }
    else if (Inet6SocketAddress::IsMatchingType(toAddress))
    {
        m_endPoint6 = m_tcp->Allocate6(GetBoundNetDevice(),
                                       Inet6SocketAddress::ConvertFrom(toAddress).GetIpv6(),
                                       Inet6SocketAddress::ConvertFrom(toAddress).GetPort(),
                                       Inet6SocketAddress::ConvertFrom(fromAddress).GetIpv6(),
                                       Inet6SocketAddress::ConvertFrom(fromAddress).GetPort());
        m_endPoint = nullptr;
    }
    m_tcp->AddSocket(this);

    // Change the cloned socket from LISTEN state to SYN_RCVD
    NS_LOG_DEBUG("LISTEN -> SYN_RCVD");
    m_state = SYN_RCVD;
    // Reached through a passive open (RFC 9293, Section 3.5, MUST-11)
    m_activeOpen = false;
    m_synCount = m_synRetries;
    m_dataRetrCount = m_dataRetries;
    SetupCallback();
    // Set the sequence number and send SYN+ACK
    SelectIsn();
    // The marks compared with the sequence numbers of the peer in the wrap
    // aware arithmetic start from its initial sequence number
    m_tcb->m_rxBuffer->SetNextRxSequence(h.GetSequenceNumber() + SequenceNumber32(1));
    m_ecnCESeq = h.GetSequenceNumber();
    m_highRxMark = h.GetSequenceNumber();
    m_highTxAck = h.GetSequenceNumber();

    // The options of the SYN are negotiated by this socket, not by the
    // listener it was forked from
    ProcessSynOptions(h);

    /* Check if we received an ECN SYN packet. Change the ECN state of receiver to ECN_IDLE if
     * sender has sent an ECN SYN packet and the traffic is ECN Capable
     */
    if (m_tcb->m_useEcn != TcpSocketState::Off &&
        (h.GetFlags() & (TcpHeader::CWR | TcpHeader::ECE)) == (TcpHeader::CWR | TcpHeader::ECE))
    {
        SendEmptyPacket(TcpHeader::SYN | TcpHeader::ACK | TcpHeader::ECE);
        NS_LOG_DEBUG(TcpSocketState::EcnStateName[m_tcb->m_ecnState] << " -> ECN_IDLE");
        m_tcb->m_ecnState = TcpSocketState::ECN_IDLE;
    }
    else
    {
        SendEmptyPacket(TcpHeader::SYN | TcpHeader::ACK);
        m_tcb->m_ecnState = TcpSocketState::ECN_DISABLED;
    }
}

void
TcpSocketBase::ConnectionSucceeded()
{ // Wrapper to protected function NotifyConnectionSucceeded() so that it can
    // be called as a scheduled event
    NotifyConnectionSucceeded();
    // The if-block below was moved from ProcessSynSent() to here because we need
    // to invoke the NotifySend() only after NotifyConnectionSucceeded() to
    // reflect the behaviour in the real world.
    if (GetTxAvailable() > 0)
    {
        NotifySend(GetTxAvailable());
    }
}

void
TcpSocketBase::AddSocketTags(const Ptr<Packet>& p, bool isEct) const
{
    if (!m_sourceRoute.empty() && m_endPoint != nullptr)
    {
        // The route the application asked for, or the one the opening segment
        // of a passively opened connection recorded (RFC 9293, Section
        // 3.9.2.1, MUST-51 and MUST-53). The segment size leaves room for the
        // option (see UpdateSegmentSizeForIpOptions).
        Ipv4SourceRouteTag routeTag;
        routeTag.SetRoute(m_sourceRoute);
        p->AddPacketTag(routeTag);
    }

    /*
     * Add tags for each socket option.
     * Note that currently the socket adds both IPv4 tag and IPv6 tag
     * if both options are set. Once the packet got to layer three, only
     * the corresponding tags will be read.
     */
    if (GetIpTos())
    {
        SocketIpTosTag ipTosTag;
        if (m_tcb->m_ecnState != TcpSocketState::ECN_DISABLED && !CheckNoEcn(GetIpTos()) && isEct)
        {
            ipTosTag.SetTos(MarkEcnCodePoint(GetIpTos(), m_tcb->m_ectCodePoint));
        }
        else
        {
            // Set the last received ipTos
            ipTosTag.SetTos(GetIpTos());
        }
        p->AddPacketTag(ipTosTag);
    }
    else
    {
        if ((m_tcb->m_ecnState != TcpSocketState::ECN_DISABLED && p->GetSize() > 0 && isEct) ||
            m_tcb->m_ecnMode == TcpSocketState::DctcpEcn)
        {
            SocketIpTosTag ipTosTag;
            ipTosTag.SetTos(MarkEcnCodePoint(GetIpTos(), m_tcb->m_ectCodePoint));
            p->AddPacketTag(ipTosTag);
        }
    }

    if (IsManualIpv6Tclass())
    {
        SocketIpv6TclassTag ipTclassTag;
        if (m_tcb->m_ecnState != TcpSocketState::ECN_DISABLED && !CheckNoEcn(GetIpv6Tclass()) &&
            isEct)
        {
            ipTclassTag.SetTclass(MarkEcnCodePoint(GetIpv6Tclass(), m_tcb->m_ectCodePoint));
        }
        else
        {
            // Set the last received ipTos
            ipTclassTag.SetTclass(GetIpv6Tclass());
        }
        p->AddPacketTag(ipTclassTag);
    }
    else
    {
        if ((m_tcb->m_ecnState != TcpSocketState::ECN_DISABLED && p->GetSize() > 0 && isEct) ||
            m_tcb->m_ecnMode == TcpSocketState::DctcpEcn)
        {
            SocketIpv6TclassTag ipTclassTag;
            ipTclassTag.SetTclass(MarkEcnCodePoint(GetIpv6Tclass(), m_tcb->m_ectCodePoint));
            p->AddPacketTag(ipTclassTag);
        }
    }

    if (IsManualIpTtl())
    {
        SocketIpTtlTag ipTtlTag;
        ipTtlTag.SetTtl(GetIpTtl());
        p->AddPacketTag(ipTtlTag);
    }

    if (IsManualIpv6HopLimit())
    {
        SocketIpv6HopLimitTag ipHopLimitTag;
        ipHopLimitTag.SetHopLimit(GetIpv6HopLimit());
        p->AddPacketTag(ipHopLimitTag);
    }

    uint8_t priority = GetPriority();
    if (priority)
    {
        SocketPriorityTag priorityTag;
        priorityTag.SetPriority(priority);
        p->ReplacePacketTag(priorityTag);
    }
}

/* Extract at most maxSize bytes from the TxBuffer at sequence seq, add the
    TCP header, and send to TcpL4Protocol */
uint32_t
TcpSocketBase::SendDataPacket(SequenceNumber32 seq, uint32_t maxSize, bool withAck)
{
    NS_LOG_FUNCTION(this << seq << maxSize << withAck);

    bool isStartOfTransmission = BytesInFlight() == 0U;
    m_lastDataTxTime = Simulator::Now();
    TcpTxItem* outItem = m_txBuffer->CopyFromSequence(maxSize, seq);

    m_rateOps->SkbSent(outItem, isStartOfTransmission);

    bool isRetransmission = outItem->IsRetrans();
    Ptr<Packet> p = outItem->GetPacketCopy();
    uint32_t sz = p->GetSize(); // Size of packet
    uint8_t flags = withAck ? TcpHeader::ACK : 0;
    uint32_t remainingData = m_txBuffer->SizeFromSequence(seq + SequenceNumber32(sz));

    // TCP sender should not send data out of the window advertised by the
    // peer when it is not retransmission.
    NS_ASSERT(isRetransmission ||
              ((m_highRxAckMark + SequenceNumber32(m_rWnd)) >= (seq + SequenceNumber32(maxSize))));

    if (IsLinuxPacing())
    {
        ++m_dataSegsOut;
        if (m_inGsoBurst)
        {
            m_paceSkbBytes += sz;
        }
        else
        {
            LinuxPaceSent(sz);
        }
    }
    else if (IsPacingEnabled())
    {
        NS_LOG_INFO("Pacing is enabled");
        if (m_pacingTimer.IsExpired())
        {
            NS_LOG_DEBUG("Current Pacing Rate " << m_tcb->m_pacingRate);
            NS_LOG_DEBUG("Timer is in expired state, activate it "
                         << m_tcb->m_pacingRate.Get().CalculateBytesTxTime(sz));
            m_pacingTimer.Schedule(m_tcb->m_pacingRate.Get().CalculateBytesTxTime(sz));
        }
        else
        {
            NS_LOG_INFO("Timer is already in running state");
        }
    }
    else
    {
        NS_LOG_INFO("Pacing is disabled");
    }

    if (withAck)
    {
        StopDelAckTimer();
        m_ackNow = false;
        m_delAckCount = 0;
        QuickAckSent();
        m_sackCompressionEvent.Cancel();
        m_compressedAcks = 0;
    }

    if (m_tcb->m_ecnState == TcpSocketState::ECN_ECE_RCVD &&
        m_ecnEchoSeq.Get() > m_ecnCWRSeq.Get() && !isRetransmission)
    {
        NS_LOG_DEBUG(TcpSocketState::EcnStateName[m_tcb->m_ecnState] << " -> ECN_CWR_SENT");
        m_tcb->m_ecnState = TcpSocketState::ECN_CWR_SENT;
        m_ecnCWRSeq = seq;
        flags |= TcpHeader::CWR;
        m_cwrPending = false;
        NS_LOG_INFO("CWR flags set");
    }
    else if (m_cwrPending && !isRetransmission && m_tcb->m_ecnState != TcpSocketState::ECN_DISABLED)
    {
        // RFC 3168, Section 6.1.2: the first new data segment sent after a
        // window reduction for any reason (a fast retransmission, a
        // retransmission timeout or an ECN Echo) carries CWR
        flags |= TcpHeader::CWR;
        m_cwrPending = false;
        NS_LOG_INFO("CWR flag set after a window reduction");
    }

    bool isEct = IsEct(isRetransmission ? TcpPacketType_t::RE_XMT : TcpPacketType_t::DATA);
    AddSocketTags(p, isEct);

    if (m_closeOnEmpty && (remainingData == 0))
    {
        flags |= TcpHeader::FIN;
        if (m_state == ESTABLISHED)
        { // On active close: I am the first one to send FIN
            NS_LOG_DEBUG("ESTABLISHED -> FIN_WAIT_1");
            m_state = FIN_WAIT_1;
        }
        else if (m_state == CLOSE_WAIT)
        { // On passive close: Peer sent me FIN already
            NS_LOG_DEBUG("CLOSE_WAIT -> LAST_ACK");
            m_state = LAST_ACK;
        }
    }
    TcpHeader header;
    // The PSH that Linux sets on the last segment of a GSO packet goes to the
    // copy of the skb sent only, whereas the PSH of the writes stays in the
    // skb, and so in its retransmissions
    bool gsoPush = m_gsoPush;
    m_gsoPush = false;
    flags |= SetPushAndUrgent(header, seq, sz);
    if (!isRetransmission)
    {
        outItem->m_gsoEnd = m_gsoLast;
    }
    if (m_forcedPush)
    {
        if (isRetransmission && outItem->m_pushed)
        {
            flags |= TcpHeader::PSH;
        }
        outItem->m_pushed = outItem->m_pushed || (flags & TcpHeader::PSH);
    }
    if (gsoPush)
    {
        flags |= TcpHeader::PSH;
    }
    header.SetFlags(flags);
    header.SetSequenceNumber(seq);
    header.SetAckNumber(m_tcb->m_rxBuffer->NextRxSequence());
    if (m_endPoint)
    {
        header.SetSourcePort(m_endPoint->GetLocalPort());
        header.SetDestinationPort(m_endPoint->GetPeerPort());
    }
    else
    {
        header.SetSourcePort(m_endPoint6->GetLocalPort());
        header.SetDestinationPort(m_endPoint6->GetPeerPort());
    }
    header.SetWindowSize(AdvertisedWindowSize());
    AddOptions(header);

    if (m_retransmitRestartsRto && isRetransmission && seq == m_txBuffer->HeadSequence() &&
        m_retxEvent.IsPending())
    {
        // As tcp_xmit_retransmit_queue() of Linux
        NS_LOG_LOGIC(this << " Retransmission of the first unacknowledged segment restarts "
                             "the RTO");
        m_retxEvent.Cancel();
    }

    if (m_retxEvent.IsExpired())
    {
        // Schedules retransmit timeout. m_rto should be already doubled.

        NS_LOG_LOGIC(this << " SendDataPacket Schedule ReTxTimeout at time "
                          << Simulator::Now().GetSeconds() << " to expire at time "
                          << (Simulator::Now() + m_rto.Get()).GetSeconds());
        m_retxEvent = Simulator::Schedule(GetRtoTimerDelay(), &TcpSocketBase::ReTxTimeout, this);
    }

    m_txTrace(p, header, this);
    if (isRetransmission)
    {
        if (m_endPoint)
        {
            m_retransmissionTrace(p,
                                  header,
                                  m_endPoint->GetLocalAddress(),
                                  m_endPoint->GetPeerAddress(),
                                  this);
        }
        else
        {
            m_retransmissionTrace(p,
                                  header,
                                  m_endPoint6->GetLocalAddress(),
                                  m_endPoint6->GetPeerAddress(),
                                  this);
        }
    }

    if (m_endPoint)
    {
        m_tcp->SendPacket(p,
                          header,
                          m_endPoint->GetLocalAddress(),
                          m_endPoint->GetPeerAddress(),
                          m_boundnetdevice);
        NS_LOG_DEBUG("Send segment of size "
                     << sz << " with remaining data " << remainingData << " via TcpL4Protocol to "
                     << m_endPoint->GetPeerAddress() << ". Header " << header);
    }
    else
    {
        m_tcp->SendPacket(p,
                          header,
                          m_endPoint6->GetLocalAddress(),
                          m_endPoint6->GetPeerAddress(),
                          m_boundnetdevice);
        NS_LOG_DEBUG("Send segment of size "
                     << sz << " with remaining data " << remainingData << " via TcpL4Protocol to "
                     << m_endPoint6->GetPeerAddress() << ". Header " << header);
    }

    // Signal to congestion control whether the cwnd is fully used
    // This is a simple version of Linux tcp_cwnd_validate() but following
    // the principle implemented in Linux that limits the updating of cwnd
    // (in the congestion controls) when flight size is >= cwnd
    // send will also be cwnd limited if less then one segment of cwnd is available
    m_tcb->m_isCwndLimited = (m_tcb->m_cWnd < BytesInFlight() + m_tcb->m_segmentSize);

    UpdateRttHistory(seq, sz, isRetransmission);

    // Update bytes sent during recovery phase
    if (m_tcb->m_congState == TcpSocketState::CA_RECOVERY ||
        m_tcb->m_congState == TcpSocketState::CA_CWR)
    {
        m_recoveryOps->UpdateBytesSent(sz);
    }

    // Notify the application of the data being sent unless this is a retransmit
    if (!isRetransmission)
    {
        Simulator::ScheduleNow(&TcpSocketBase::NotifyDataSent,
                               this,
                               (seq + sz - m_tcb->m_highTxMark.Get()));
    }
    // Update highTxMark
    if (sz < m_tcb->m_segmentSize && seq + sz > m_tcb->m_highTxMark)
    {
        // tcp_minshall_update(): the end of the last short segment sent
        m_sndSml = seq + sz;
    }
    m_tcb->m_highTxMark = std::max(seq + sz, m_tcb->m_highTxMark.Get());
    while (!m_lxSkbs.empty() && m_lxSkbs.front().end <= m_tcb->m_highTxMark)
    {
        m_lxSkbs.pop_front();
    }
    return sz;
}

void
TcpSocketBase::UpdateRttHistory(const SequenceNumber32& seq, uint32_t sz, bool isRetransmission)
{
    NS_LOG_FUNCTION(this);

    // update the history of sequence numbers used to calculate the RTT
    if (!isRetransmission)
    { // This is the next expected one, just log at end
        m_history.emplace_back(seq, sz, Simulator::Now());
    }
    else
    { // This is a retransmit, find in list and mark as re-tx
        for (auto i = m_history.begin(); i != m_history.end(); ++i)
        {
            if ((seq >= i->seq) && (seq < (i->seq + SequenceNumber32(i->count))))
            { // Found it
                i->retx = true;
                i->count = ((seq + SequenceNumber32(sz)) - i->seq); // And update count in hist
                break;
            }
        }
    }
}

uint8_t
TcpSocketBase::SetPushAndUrgent(TcpHeader& header, SequenceNumber32 seq, uint32_t size)
{
    uint8_t flags = 0;
    if (size > 0 && m_txBuffer->SizeFromSequence(seq + SequenceNumber32(size)) == 0)
    {
        // No data is queued after this segment, so it is the last of the
        // buffer and says so (RFC 9293, Section 3.9.1.3, MUST-61)
        flags |= TcpHeader::PSH;
    }
    if (m_forcedPush && size > 0)
    {
        // As Linux, the segment that ends an skb marked by tcp_push() or
        // forced_push() carries PSH, and so does its retransmission
        SequenceNumber32 end = seq + SequenceNumber32(size);
        while (!m_lxPushEnds.empty() && m_lxPushEnds.front() <= m_txBuffer->HeadSequence())
        {
            m_lxPushEnds.pop_front();
        }
        bool push = false;
        if (seq >= m_tcb->m_highTxMark)
        {
            for (const auto& skb : m_lxSkbs)
            {
                if (skb.end >= end)
                {
                    push = skb.end == end && skb.push;
                    break;
                }
            }
            if (push)
            {
                m_lxPushEnds.push_back(end);
            }
        }
        else
        {
            push = std::find(m_lxPushEnds.begin(), m_lxPushEnds.end(), end) != m_lxPushEnds.end();
        }
        if (push)
        {
            flags |= TcpHeader::PSH;
        }
    }

    if (m_gsoPush)
    {
        // As __tcp_transmit_skb() of Linux, the last segment of a GSO packet
        // carries PSH
        flags |= TcpHeader::PSH;
        m_gsoPush = false;
    }

    if (m_sndUrgentArmed && m_sndUrgentPoint <= m_txBuffer->HeadSequence())
    {
        // The urgent data was acknowledged whole: the urgent point is no
        // longer comparable to the live sequence numbers once they wrap
        // around, so it is disarmed
        m_sndUrgentArmed = false;
    }

    if (m_sndUrgentArmed && m_sndUrgentPoint > seq)
    {
        // Urgent data reaches beyond the start of this segment: flag it and
        // point past its last octet, saturating the 16 bit field for urgent
        // data longer than that (RFC 9293, Section 3.8.5)
        flags |= TcpHeader::URG;
        uint32_t offset = (m_sndUrgentPoint - seq);
        header.SetUrgentPointer(static_cast<uint16_t>(std::min(offset, 65535U)));
    }
    return flags;
}

uint32_t
TcpSocketBase::SendPendingData(bool withAck)
{
    NS_LOG_FUNCTION(this << withAck);
    if (m_txBuffer->Size() == 0)
    {
        return 0; // Nothing to send
    }
    if (m_endPoint == nullptr && m_endPoint6 == nullptr)
    {
        NS_LOG_INFO(
            "TcpSocketBase::SendPendingData: No endpoint; m_shutdownSend=" << m_shutdownSend);
        return 0; // Is this the right way to handle this condition?
    }

    uint32_t nPacketsSent = 0;
    uint32_t availableWindow = AvailableWindow();
    m_maxSndWnd = std::max(m_maxSndWnd, m_rWnd.Get());
    m_lxSentEnd.reset();
    // The GSO packet of Linux being sent: segments left in it, and the
    // sequence number its next segment is to start at
    uint32_t gsoLeft = 0;
    uint32_t gsoSegments = 0;
    SequenceNumber32 gsoNext;
    bool deferCwndLimited = false;

    // RFC 6675, Section (C)
    // If cwnd - pipe >= 1 SMSS, the sender SHOULD transmit one or more
    // segments as follows:
    // (NOTE: We check > 0, and do the checks for segmentSize in the following
    // else branch to control silly window syndrome and Nagle)
    while (availableWindow > 0)
    {
        if (IsLinuxPacing())
        {
            // As tcp_pacing_check() of Linux, a GSO packet waits for the
            // departure time that the previous one set
            if (gsoLeft == 0 && m_paceNext > Simulator::Now())
            {
                NS_LOG_INFO("Pacing until " << m_paceNext.As(Time::S));
                if (!m_pacingTimer.IsRunning())
                {
                    m_pacingTimer.Schedule(m_paceNext - Simulator::Now());
                }
                break;
            }
        }
        else if (IsPacingEnabled())
        {
            NS_LOG_INFO("Pacing is enabled");
            if (m_pacingTimer.IsRunning())
            {
                NS_LOG_INFO("Skipping Packet due to pacing" << m_pacingTimer.GetDelayLeft());
                break;
            }
            NS_LOG_INFO("Timer is not running");
        }

        if (m_tcb->m_congState == TcpSocketState::CA_OPEN && m_state == TcpSocket::FIN_WAIT_1)
        {
            NS_LOG_INFO("FIN_WAIT and OPEN state; no data to transmit");
            break;
        }
        // (C.1) The scoreboard MUST be queried via NextSeg () for the
        //       sequence number range of the next segment to transmit (if
        //       any), and the given segment sent.  If NextSeg () returns
        //       failure (no data to send), return without sending anything
        //       (i.e., terminate steps C.1 -- C.5).
        SequenceNumber32 next;
        SequenceNumber32 nextHigh;
        m_txBuffer->SetLinuxLostRetransmit(!m_forwardRetransmit);
        bool enableRule3 = m_sackEnabled && m_forwardRetransmit &&
                           m_tcb->m_congState == TcpSocketState::CA_RECOVERY;
        if (!m_txBuffer->NextSeg(&next, &nextHigh, enableRule3))
        {
            NS_LOG_INFO("no valid seq to transmit, or no data available");
            break;
        }
        else
        {
            if (m_newDataOnly && next < m_tcb->m_highTxMark)
            {
                // Only new data go: Linux retransmits on the ACKs, and not
                // on the ACK revealing SACK reneging, on which
                // tcp_fastretrans_alert() returns
                if (m_txBuffer->SizeFromSequence(m_tcb->m_highTxMark) == 0)
                {
                    break;
                }
                NS_LOG_LOGIC("No retransmission on the ACK revealing SACK reneging");
                next = m_tcb->m_highTxMark;
                nextHigh = next + SequenceNumber32(m_tcb->m_segmentSize);
            }
            // It's time to transmit, but before do silly window and Nagle's check
            uint32_t availableData = m_txBuffer->SizeFromSequence(next);

            // If there's less app data than the full window, ask the app for more
            // data before trying to send
            if (availableData < availableWindow)
            {
                NotifySend(GetTxAvailable());
            }

            // Stop sending if we need to wait for a larger Tx window (prevent silly window
            // syndrome) but continue if we don't have data
            // As Linux (tcp_snd_wnd_test()), a GSO packet starts only if its
            // first segment fits the window: the short segment filling the
            // window is sent only within the GSO packet being sent
            uint32_t partialAfter = nPacketsSent;
            if (m_forcedPush || m_tsoDefer)
            {
                partialAfter = (gsoLeft > 0 && next == gsoNext) ? 1 : 0;
            }
            if ((m_forcedPush || m_tsoDefer) && partialAfter == 0 && next >= m_tcb->m_highTxMark)
            {
                SequenceNumber32 wndEnd =
                    m_txBuffer->HeadSequence() + SequenceNumber32(m_rWnd.Get());
                uint32_t first = std::min(availableData, m_tcb->m_segmentSize);
                if (wndEnd < next + SequenceNumber32(first))
                {
                    NS_LOG_LOGIC("The first segment of " << next << " exceeds the receive window");
                    break;
                }
            }
            if (availableWindow < m_tcb->m_segmentSize && availableData > availableWindow &&
                !CanSendPartialSegment(availableWindow, partialAfter))
            {
                NS_LOG_LOGIC("Preventing Silly Window Syndrome. Wait to send.");
                break; // No more
            }
            // Nagle's algorithm (RFC896): Hold off sending if there is unacked data
            // in the buffer and the amount of data to send is less than one segment
            // With TsoDefer, as Linux (tcp_nagle_check()), only new data, and
            // only while a short segment sent is not acknowledged (Minshall's
            // variant)
            bool nagleHold = UnAckDataCount() > 0;
            if (m_tsoDefer)
            {
                nagleHold = next >= m_tcb->m_highTxMark && m_sndSml &&
                            *m_sndSml > m_txBuffer->HeadSequence() &&
                            *m_sndSml <= m_tcb->m_highTxMark;
            }
            if (!m_noDelay && nagleHold && availableData < m_tcb->m_segmentSize)
            {
                NS_LOG_DEBUG("Invoking Nagle's algorithm for seq "
                             << next << ", SFS: " << m_txBuffer->SizeFromSequence(next)
                             << ". Wait to send.");
                break;
            }

            uint32_t s = std::min(availableWindow, m_tcb->m_segmentSize);
            // NextSeg () may have further constrained the segment size
            auto maxSizeToSend = static_cast<uint32_t>(nextHigh - next);
            s = std::min(s, maxSizeToSend);

            // (C.2) If any of the data octets sent in (C.1) are below HighData,
            //       HighRxt MUST be set to the highest sequence number of the
            //       retransmitted segment unless NextSeg () rule (4) was
            //       invoked for this retransmission.
            // (C.3) If any of the data octets sent in (C.1) are above HighData,
            //       HighData must be updated to reflect the transmission of
            //       previously unsent data.
            //
            // These steps are done in m_txBuffer with the tags.
            if (m_tcb->m_nextTxSequence != next)
            {
                m_tcb->m_nextTxSequence = next;
            }
            if (m_tcb->m_bytesInFlight.Get() == 0 && !m_lxTxStarted)
            {
                m_congestionControl->CwndEvent(m_tcb, TcpSocketState::CA_EVENT_TX_START);
            }
            m_lxTxStarted = false;
            bool gsoStart = gsoLeft == 0 || next != gsoNext;
            // The skb of Linux the GSO packet starts, and the segments the
            // congestion window lets it take (tcp_cwnd_test())
            uint32_t skbLen = availableData;
            bool skbTail = true;
            uint32_t cwndQuota = 0;
            if ((m_forcedPush || m_tsoDefer) && gsoStart && next >= m_tcb->m_highTxMark)
            {
                cwndQuota = std::max(GetLinuxCwndQuota(), 1U);
                if (uint32_t len = GrowLinuxSkb(next, cwndQuota, &skbTail); len > 0)
                {
                    skbLen = std::min(skbLen, len);
                }
            }
            if (m_tsoDefer && gsoStart && next >= m_tcb->m_highTxMark &&
                ShouldDeferTso(next, skbLen, skbTail))
            {
                NS_LOG_LOGIC("Deferring the transmission of " << next << " as Linux TSO");
                // As Linux, a deferral that the congestion window causes
                // still counts as cwnd-limited, for the window to grow
                SequenceNumber32 wndEnd =
                    m_txBuffer->HeadSequence() + SequenceNumber32(m_rWnd.Get());
                if (wndEnd > next &&
                    m_tcb->m_cWnd.Get() - BytesInFlight() < static_cast<uint32_t>(wndEnd - next))
                {
                    m_tcb->m_isCwndLimited = true;
                    deferCwndLimited = true;
                }
                break;
            }
            if (m_forcedPush || m_tsoDefer)
            {
                if (gsoStart)
                {
                    // tcp_write_xmit(): a GSO packet holds new data up to the
                    // TSO size, the congestion window, the receive window and
                    // the end of the skb (here, of the application write)
                    gsoSegments = 1;
                    if (next >= m_tcb->m_highTxMark)
                    {
                        // tcp_mss_split_point(): the GSO packet takes the
                        // segments the congestion window allows, cut by the
                        // end of the skb and the right edge of the receive
                        // window; with FillReceiveWindow, the short segment
                        // filling the window ends the packet
                        const uint32_t mss = m_tcb->m_segmentSize;
                        SequenceNumber32 wndEnd =
                            m_txBuffer->HeadSequence() + SequenceNumber32(m_rWnd.Get());
                        uint32_t window = wndEnd > next ? static_cast<uint32_t>(wndEnd - next) : 0;
                        uint32_t maxLen = cwndQuota * mss;
                        uint32_t limit = std::min(maxLen, skbLen);
                        if (maxLen > window || skbTail)
                        {
                            uint32_t needed = std::min(skbLen, window);
                            limit = std::min(maxLen, needed);
                            // A short last segment stays out while Nagle
                            // holds it (tcp_nagle_check())
                            bool nagle = m_tsoDefer && !m_noDelay && m_sndSml &&
                                         *m_sndSml > m_txBuffer->HeadSequence() &&
                                         *m_sndSml <= m_tcb->m_highTxMark;
                            if (limit == needed && needed % mss > 0 && (!m_fillRcvWindow || nagle))
                            {
                                limit -= needed % mss;
                            }
                        }
                        uint32_t segments = (limit + mss - 1) / mss;
                        gsoSegments = std::clamp(segments, 1U, cwndQuota);
                    }
                    else
                    {
                        // tcp_retransmit_skb(): the lost part of an skb is
                        // retransmitted as one GSO packet, up to the TSO
                        // size and the congestion window
                        uint32_t limit =
                            std::min(availableWindow, GetLinuxTsoSegments() * m_tcb->m_segmentSize);
                        uint32_t lostBytes = m_txBuffer->LostPacketBytes(next, limit);
                        gsoSegments =
                            std::max(1U,
                                     (lostBytes + m_tcb->m_segmentSize - 1) / m_tcb->m_segmentSize);
                    }
                    gsoLeft = gsoSegments;
                }
                m_gsoPush = m_forcedPush && gsoSegments > 1 && gsoLeft == 1;
                --gsoLeft;
                gsoNext = next + SequenceNumber32(s);
            }
            bool newData = m_tcb->m_nextTxSequence >= m_tcb->m_highTxMark;
            m_inGsoBurst = IsLinuxPacing();
            m_gsoLast = gsoLeft == 0;
            uint32_t sz = SendDataPacket(m_tcb->m_nextTxSequence, s, withAck);
            m_gsoLast = true;
            m_inGsoBurst = false;
            if (gsoLeft == 0 && m_paceSkbBytes > 0)
            {
                LinuxPaceSent(m_paceSkbBytes);
            }
            if (m_sndSkbOverhead > 0 && newData && (m_forcedPush || m_tsoDefer))
            {
                // The GSO packet sent is an skb of the retransmission queue
                SequenceNumber32 end = m_tcb->m_nextTxSequence + SequenceNumber32(sz);
                if (gsoStart || m_lxRtxSkbEnds.empty())
                {
                    m_lxRtxSkbEnds.push_back(end);
                }
                else
                {
                    m_lxRtxSkbEnds.back() = end;
                }
                while (!m_lxRtxSkbEnds.empty() &&
                       m_lxRtxSkbEnds.front() <= m_txBuffer->HeadSequence())
                {
                    m_lxRtxSkbEnds.pop_front();
                }
            }

            NS_LOG_LOGIC(" rxwin " << m_rWnd << " segsize " << m_tcb->m_segmentSize
                                   << " highestRxAck " << m_txBuffer->HeadSequence() << " pd->Size "
                                   << m_txBuffer->Size() << " pd->SFS "
                                   << m_txBuffer->SizeFromSequence(m_tcb->m_nextTxSequence));

            NS_LOG_DEBUG("cWnd: " << m_tcb->m_cWnd << " total unAck: " << UnAckDataCount()
                                  << " sent seq " << m_tcb->m_nextTxSequence << " size " << sz);
            m_tcb->m_nextTxSequence += sz;
            ++nPacketsSent;
            if (!IsLinuxPacing() && IsPacingEnabled())
            {
                NS_LOG_INFO("Pacing is enabled");
                if (m_pacingTimer.IsExpired())
                {
                    NS_LOG_DEBUG("Current Pacing Rate " << m_tcb->m_pacingRate);
                    NS_LOG_DEBUG("Timer is in expired state, activate it "
                                 << m_tcb->m_pacingRate.Get().CalculateBytesTxTime(sz));
                    m_pacingTimer.Schedule(m_tcb->m_pacingRate.Get().CalculateBytesTxTime(sz));
                    break;
                }
            }
        }

        // (C.4) The estimate of the amount of data outstanding in the
        //       network must be updated by incrementing pipe by the number
        //       of octets transmitted in (C.1).
        //
        // Done in BytesInFlight, inside AvailableWindow.
        availableWindow = AvailableWindow();

        // (C.5) If cwnd - pipe >= 1 SMSS, return to (C.1)
        // loop again!
    }

    if (m_paceSkbBytes > 0)
    {
        // The GSO packet ended early
        LinuxPaceSent(m_paceSkbBytes);
    }

    m_newDataOnly = false;

    if (m_linuxCwndLimited)
    {
        ValidateLinuxCwnd(nPacketsSent, deferCwndLimited);
    }

    if (nPacketsSent > 0)
    {
        if (!m_sackEnabled)
        {
            if (!m_limitedTx)
            {
                // We can't transmit in CA_DISORDER without limitedTx active
                NS_ASSERT(m_tcb->m_congState != TcpSocketState::CA_DISORDER);
            }
        }

        NS_LOG_DEBUG("SendPendingData sent " << nPacketsSent << " segments");
    }
    else
    {
        NS_LOG_DEBUG("SendPendingData no segments sent");
    }
    return nPacketsSent;
}

void
TcpSocketBase::IncreaseWindow(uint32_t segmentsAcked)
{
    NS_LOG_FUNCTION(this << segmentsAcked);
    if (m_linuxCwndLimited && !IsLinuxCwndLimited())
    {
        NS_LOG_LOGIC("The congestion window does not limit the transmissions: not grown");
        return;
    }
    m_congestionControl->IncreaseWindow(m_tcb, segmentsAcked);
}

bool
TcpSocketBase::IsLinuxCwndLimited() const
{
    if (m_lxIsCwndLimited)
    {
        return true;
    }
    // In slow start, the window grows up to twice the largest flight
    return m_tcb->m_cWnd < m_tcb->m_ssThresh &&
           m_tcb->m_cWnd.Get() / m_tcb->m_segmentSize < 2 * m_lxMaxPacketsOut;
}

void
TcpSocketBase::ValidateLinuxCwnd(uint32_t sent, bool cwndLimited)
{
    NS_LOG_FUNCTION(this << sent << cwndLimited);
    const uint32_t mss = m_tcb->m_segmentSize;
    uint32_t inFlight = (BytesInFlight() + mss - 1) / mss;
    cwndLimited = cwndLimited || inFlight >= m_tcb->m_cWnd.Get() / mss;
    if (sent == 0 && !cwndLimited)
    {
        return;
    }
    // tcp_cwnd_validate(): the state holds for the window of data sent,
    // until its last byte is acknowledged, unless the window limits the
    // transmissions or the flight grows larger in the meantime
    uint32_t packetsOut = (UnAckDataCount() + mss - 1) / mss;
    if (!m_lxCwndUsageSeq || m_txBuffer->HeadSequence() >= *m_lxCwndUsageSeq || cwndLimited ||
        (!m_lxIsCwndLimited && packetsOut > m_lxMaxPacketsOut))
    {
        m_lxIsCwndLimited = cwndLimited;
        m_lxMaxPacketsOut = packetsOut;
        m_lxCwndUsageSeq = m_tcb->m_highTxMark;
    }
}

void
TcpSocketBase::SendWrittenData()
{
    NS_LOG_FUNCTION(this);
    // As tcp_push() of Linux, which calls tcp_write_xmit(), the data written
    // sends new data only
    m_newDataOnly = m_forcedPush || m_tsoDefer;
    SendPendingData(m_connected);
}

uint32_t
TcpSocketBase::UnAckDataCount() const
{
    return m_tcb->m_highTxMark - m_txBuffer->HeadSequence();
}

uint32_t
TcpSocketBase::BytesInFlight() const
{
    uint32_t bytesInFlight = m_txBuffer->BytesInFlight();
    // Ugly, but we are not modifying the state; m_bytesInFlight is used
    // only for tracing purpose.
    m_tcb->m_bytesInFlight = bytesInFlight;

    NS_LOG_DEBUG("Returning calculated bytesInFlight: " << bytesInFlight);
    return bytesInFlight;
}

uint32_t
TcpSocketBase::FlightSize() const
{
    return m_txBuffer->Size() - m_txBuffer->SizeFromSequence(m_tcb->m_highTxMark);
}

uint32_t
TcpSocketBase::CongestionFlightSize() const
{
    // The pipe is refreshed as well: the recovery operations, which start
    // after the threshold is set, read it from the socket state
    BytesInFlight();
    return std::min(FlightSize(), m_tcb->m_cWnd.Get());
}

uint32_t
TcpSocketBase::Window() const
{
    return std::min(m_rWnd.Get(), m_tcb->m_cWnd.Get());
}

bool
TcpSocketBase::ShouldDeferTso(SequenceNumber32 next, uint32_t skbLen, bool tail) const
{
    NS_LOG_FUNCTION(this << next << skbLen << tail);
    const uint32_t mss = m_tcb->m_segmentSize;
    if (m_tcb->m_congState >= TcpSocketState::CA_RECOVERY)
    {
        return false;
    }
    // Deferring only right after a transmission, within 1 ms
    if (!m_lastDataTxTime || Simulator::Now() - *m_lastDataTxTime > MilliSeconds(1))
    {
        return false;
    }
    if (skbLen <= mss)
    {
        return false;
    }
    uint32_t cwnd = m_tcb->m_cWnd.Get();
    uint32_t inFlight = BytesInFlight();
    if (cwnd <= inFlight)
    {
        return false;
    }
    SequenceNumber32 wndEnd = m_txBuffer->HeadSequence() + SequenceNumber32(m_rWnd.Get());
    uint32_t sendWin = wndEnd > next ? static_cast<uint32_t>(wndEnd - next) : 0;
    uint32_t limit = std::min(sendWin, cwnd - inFlight);
    // A full GSO packet can be sent, or the whole skb, or a third of the
    // window (tcp_tso_win_divisor)
    if (limit >= GetLinuxTsoSegments() * mss || (!tail && limit >= skbLen) ||
        limit >= std::min(m_rWnd.Get(), cwnd) / 3)
    {
        return false;
    }
    // The next ACK is likely to come too late
    Time headSent = m_txBuffer->GetHeadLastSent();
    if (headSent == Time::Max() || Simulator::Now() - headSent < m_tcb->m_srtt.Get() / 2)
    {
        return false;
    }
    // The FIN is not delayed
    if (tail && m_closeOnEmpty)
    {
        return false;
    }
    return true;
}

uint32_t
TcpSocketBase::GetLinuxSizeGoal() const
{
    // tcp_xmit_size_goal(): a GSO packet of 64 KiB, less 320 bytes of
    // MAX_TCP_HEADER, bounded by half the largest window, in whole segments
    const uint32_t mss = m_tcb->m_segmentSize;
    uint32_t maxWnd = std::max(m_maxSndWnd, m_rWnd.Get());
    return std::max(mss, std::min(65536U - 1 - 320, maxWnd / 2) / mss * mss);
}

uint32_t
TcpSocketBase::QueueLinuxWrite(SequenceNumber32 start, SequenceNumber32 end)
{
    NS_LOG_FUNCTION(this << start << end);
    const uint32_t mss = m_tcb->m_segmentSize;
    const uint32_t goal = GetLinuxSizeGoal();
    const uint32_t maxWnd = std::max(m_maxSndWnd, m_rWnd.Get());
    if (!m_lxPushedSeq)
    {
        // tcp_connect_init(): pushed_seq starts at the first data byte
        m_lxPushedSeq = start;
    }
    // The data Linux sends at this instant, as tcp_sendmsg_locked() pushes
    // the skbs it fills: what the windows let go, beyond what is sent
    SequenceNumber32 sent = m_tcb->m_highTxMark;
    if (m_lxSentEnd && *m_lxSentEnd > sent)
    {
        sent = *m_lxSentEnd;
    }
    auto push = [&](SequenceNumber32 written) {
        if (m_state != ESTABLISHED && m_state != CLOSE_WAIT)
        {
            return;
        }
        if (!m_lxTxStarted && BytesInFlight() == 0)
        {
            // Linux starts the transmission within the write
            // (tcp_event_data_sent())
            m_congestionControl->CwndEvent(m_tcb, TcpSocketState::CA_EVENT_TX_START);
            m_lxTxStarted = true;
        }
        SequenceNumber32 limit =
            m_tcb->m_highTxMark + SequenceNumber32(AvailableWindow() / mss * mss);
        sent = std::max(sent, std::min(written, limit));
        m_lxSentEnd = sent;
    };
    // The data of the last skb not yet sent, and whether all the skbs
    // before it are sent (it is the send head)
    auto unsent = [&](uint32_t* len, bool* head) {
        if (m_lxSkbs.empty() || m_lxSkbs.back().end <= sent)
        {
            return false;
        }
        SequenceNumber32 skbStart = sent;
        if (m_lxSkbs.size() > 1 && m_lxSkbs[m_lxSkbs.size() - 2].end > sent)
        {
            skbStart = m_lxSkbs[m_lxSkbs.size() - 2].end;
        }
        *len = m_lxSkbs.back().end - skbStart;
        *head = skbStart == sent;
        return true;
    };

    SequenceNumber32 pos = start;
    bool copied = false;
    while (pos < end)
    {
        uint32_t len = 0;
        bool head = false;
        uint32_t left = end - pos;
        if (unsent(&len, &head) && len < goal)
        {
            // The data is appended to the last skb, which loses the PSH
            // of the previous write
            uint32_t copy = std::min(goal - len, left);
            if (!copied)
            {
                m_lxSkbs.back().push = false;
            }
            m_lxSkbs.back().end += copy;
            pos += copy;
        }
        else
        {
            // sk_stream_memory_free(): no new skb once the buffer is full
            if (m_sndSkbOverhead > 0 &&
                GetLinuxSndBufCost() + (pos - start) >= m_txBuffer->MaxBufferSize())
            {
                break;
            }
            uint32_t copy = std::min(goal, left);
            pos += copy;
            m_lxSkbs.push_back({pos, false});
        }
        copied = true;
        if (pos == end)
        {
            break;
        }
        // The skb is full: forced_push() marks it and sends the pending data,
        // or tcp_push_one() sends from it if it is the send head
        unsent(&len, &head);
        if (pos > *m_lxPushedSeq + SequenceNumber32(maxWnd / 2))
        {
            m_lxSkbs.back().push = true;
            m_lxPushedSeq = pos;
            push(pos);
        }
        else if (head)
        {
            push(pos);
        }
    }
    // tcp_push() marks the last skb and sends, also when the write waits for
    // space
    if (pos > start && !m_lxSkbs.empty() && m_lxSkbs.back().end == pos)
    {
        m_lxSkbs.back().push = true;
        m_lxPushedSeq = pos;
    }
    push(pos);
    return pos - start;
}

uint32_t
TcpSocketBase::GetLinuxCwndQuota() const
{
    // tcp_cwnd_test(): the segments the congestion window allows, at most half
    // of it, and at most a GSO packet
    const uint32_t mss = m_tcb->m_segmentSize;
    uint32_t cwnd = m_tcb->m_cWnd.Get() / mss;
    uint32_t inFlight = BytesInFlight() / mss;
    uint32_t quota = cwnd > inFlight ? std::min(std::max(cwnd / 2, 1U), cwnd - inFlight) : 0;
    return std::min(quota, GetLinuxTsoSegments());
}

uint32_t
TcpSocketBase::GrowLinuxSkb(SequenceNumber32 next, uint32_t quota, bool* tail)
{
    NS_LOG_FUNCTION(this << next << quota);
    while (!m_lxSkbs.empty() && m_lxSkbs.front().end <= next)
    {
        m_lxSkbs.pop_front();
    }
    *tail = true;
    if (m_lxSkbs.empty())
    {
        return 0;
    }
    LinuxSkb& skb = m_lxSkbs.front();
    uint32_t len = skb.end - next;
    NS_LOG_LOGIC("skb ends " << skb.end << (m_lxSkbs.size() > 1 ? " " : "")
                             << (m_lxSkbs.size() > 1 ? m_lxSkbs[1].end : skb.end) << " of "
                             << m_lxSkbs.size());
    uint32_t want = quota * m_tcb->m_segmentSize;
    if (len < want && m_lxSkbs.size() > 1)
    {
        // Data moves in from the next skb, which disappears if it gives it
        // all, its flags joining those of this skb
        uint32_t amount = want - len;
        uint32_t nextLen = m_lxSkbs[1].end - skb.end;
        if (amount >= nextLen)
        {
            skb.end = m_lxSkbs[1].end;
            skb.push = skb.push || m_lxSkbs[1].push;
            m_lxSkbs.erase(m_lxSkbs.begin() + 1);
        }
        else
        {
            skb.end += amount;
        }
    }
    *tail = m_lxSkbs.size() == 1;
    return skb.end - next;
}

uint32_t
TcpSocketBase::GetLinuxTsoSegments() const
{
    const uint32_t mss = m_tcb->m_segmentSize;
    uint64_t rate;
    if (m_congestionControl->HasCongControl())
    {
        // The congestion control sets the pacing rate itself
        rate = m_tcb->m_pacingRate.Get().GetBitRate() / 8;
    }
    else
    {
        // tcp_update_pacing_rate(): 200% of cwnd * mss / srtt in slow start,
        // below half the slow start threshold, and 120% otherwise, with the
        // srtt in eighths of microseconds
        uint64_t cwnd = std::max(m_tcb->m_cWnd.Get(), m_tcb->m_bytesInFlight.Get()) / mss;
        // in segments, as tcp_snd_cwnd() and snd_ssthresh count them
        uint64_t ratio =
            m_tcb->m_cWnd.Get() / mss < (m_tcb->m_ssThresh.Get() / mss) / 2 ? 200 : 120;
        int64_t srtt8 = m_tcb->m_srtt.Get().GetNanoSeconds() * 8 / 1000;
        rate = mss * 80000ULL * ratio * cwnd;
        rate = srtt8 > 0 ? rate / srtt8 : std::numeric_limits<uint32_t>::max();
    }
    // tcp_tso_autosize(): the bytes sent in about 1 ms at that rate
    // (sk_pacing_shift of 10), more with a small minimum RTT, at most a GSO
    // packet of 64 KiB, and at least two segments
    const uint64_t gsoMaxSize = 65536;
    uint64_t bytes = rate >> 10;
    if (m_tcb->m_minRtt != Time::Max())
    {
        uint64_t r = static_cast<uint64_t>(m_tcb->m_minRtt.GetMicroSeconds()) >> 9;
        if (r < 32)
        {
            bytes += gsoMaxSize >> r;
        }
    }
    bytes = std::min(bytes, gsoMaxSize);
    uint32_t minSegs = m_congestionControl->GetMinTsoSegs(m_tcb);
    NS_LOG_DEBUG("TSO segments " << std::max<uint32_t>(bytes / mss, minSegs) << " for a rate of "
                                 << rate << " B/s");
    return std::max<uint32_t>(bytes / mss, minSegs);
}

bool
TcpSocketBase::CanSendPartialSegment(uint32_t availableWindow, uint32_t nPacketsSent) const
{
    NS_LOG_FUNCTION(this << availableWindow << nPacketsSent);
    // Sender SWS avoidance (RFC 9293, Section 3.8.6.2.1): a segment shorter
    // than the MSS is sent when it carries at least a fraction Fs = 1/2 of
    // the largest window the peer has offered, so that a peer whose window
    // never reaches one MSS still receives data
    if (availableWindow >= m_maxSndWnd / 2)
    {
        NS_LOG_LOGIC("Sending " << availableWindow << " bytes, at least half of the maximum window "
                                << m_maxSndWnd);
        return true;
    }
    // As tcp_mss_split_point() of Linux, the last segment of a burst that
    // the right edge of the receive window cuts is sent short, filling the
    // window
    if (m_fillRcvWindow && nPacketsSent > 0 && m_rWnd.Get() <= m_tcb->m_cWnd.Get())
    {
        NS_LOG_LOGIC("Filling the receive window with " << availableWindow << " bytes");
        return true;
    }
    return false;
}

uint32_t
TcpSocketBase::AvailableWindow() const
{
    uint32_t win = Window(); // Number of bytes allowed to be outstanding

    if (m_sackEnabled && m_fackEnabled && win >= m_tcb->m_ssThresh)
    {
        // Update awnd (Data sender's estimate of the actual quantity of data outstanding in the
        // network)
        NS_LOG_DEBUG("FACK is enabled and win >= ssthresh (" << m_tcb->m_ssThresh << ")");

        uint32_t sndNxt = (m_tcb->m_highTxMark.Get().GetValue());
        uint32_t retranData = m_txBuffer->GetRetransmitsCount();
        uint32_t awnd = sndNxt - m_sndFack + retranData;
        m_tcb->m_fackAwnd = awnd;

        NS_LOG_DEBUG("SndNxt: " << sndNxt << ", SndFack: " << m_sndFack << ", AWND: " << awnd
                                << ", RetranData :" << retranData);

        uint32_t awndDiff = (win > awnd) ? (win - awnd) : 0;
        NS_LOG_DEBUG("AWND: " << awnd << ", win: " << win << ", AWND_DIFF: " << awndDiff);
        return awndDiff;
    }

    uint32_t inflight = BytesInFlight();

    if (inflight >= win)
    {
        return 0;
    }

    return win - inflight;
}

uint32_t
TcpSocketBase::GetReceiveWindow() const
{
    uint32_t w;

    // We don't want to advertise 0 after a FIN is received. So, we just use
    // the previous value of the advWnd.
    if (m_tcb->m_rxBuffer->GotFin())
    {
        w = m_advWnd;
    }
    else
    {
        NS_ASSERT_MSG(m_tcb->m_rxBuffer->MaxRxSequence() - m_tcb->m_rxBuffer->NextRxSequence() >= 0,
                      "Unexpected sequence number values");
        w = static_cast<uint32_t>(m_tcb->m_rxBuffer->MaxRxSequence() -
                                  m_tcb->m_rxBuffer->NextRxSequence());
    }
    if (m_rcvSsthresh)
    {
        // The window clamped after the receive buffer overflowed
        w = std::min(w, *m_rcvSsthresh);
    }

    if (!m_advRightEdgeArmed)
    {
        return w;
    }

    SequenceNumber32 next = m_tcb->m_rxBuffer->NextRxSequence();
    uint32_t held = m_advRightEdge > next ? m_advRightEdge - next : 0;
    if (w < held)
    {
        // The window is not shrunk (RFC 9293, Section 3.8.6, SHLD-14)
        return held;
    }

    // Silly window syndrome avoidance in the receiver (RFC 9293, Section
    // 3.8.6.2.2, MUST-39): the right edge RCV.NXT + RCV.WND is held where it
    // is until it can move by min(Fr * RCV.BUFF, Eff.snd.MSS), with Fr = 1/2,
    // whatever the size of the window
    uint32_t threshold = std::min(m_tcb->m_segmentSize, m_tcb->m_rxBuffer->MaxBufferSize() / 2);
    if (w - held < threshold)
    {
        NS_LOG_LOGIC("Not advancing the right edge by " << w - held << ", less than " << threshold);
        return held;
    }
    return w;
}

uint16_t
TcpSocketBase::AdvertisedWindowSize(bool scale) const
{
    NS_LOG_FUNCTION(this << scale);
    if (m_linuxRcvWindow)
    {
        auto self = const_cast<TcpSocketBase*>(this);
        if (!scale)
        {
            // The window of a SYN is not scaled
            self->LinuxInitWindow();
            uint32_t synWin = std::min<uint32_t>(m_lxRcvWnd, m_maxWinSize);
            self->m_advWnd = synWin;
            return static_cast<uint16_t>(synWin);
        }
        if (m_lxInit && m_state != CLOSED && m_state != LISTEN && m_state != SYN_SENT)
        {
            uint32_t win = self->LinuxAdvertiseWindow();
            self->m_advWnd = win;
            self->m_advRightEdge = m_tcb->m_rxBuffer->NextRxSequence() + win;
            self->m_advRightEdgeArmed = true;
            return static_cast<uint16_t>(win >> m_rcvWindShift);
        }
    }
    uint32_t w = GetReceiveWindow();
    SequenceNumber32 next = m_tcb->m_rxBuffer->NextRxSequence();
    uint8_t shift = scale ? m_rcvWindShift : 0;

    uint32_t field = w >> shift;
    if (m_advRightEdgeArmed && next + (field << shift) < m_advRightEdge)
    {
        // Rounded down to the unit of the window scale, the right edge would
        // move left of the one advertised: round it up instead (RFC 7323,
        // Section 2.4)
        field++;
    }
    if (field > m_maxWinSize)
    {
        field = m_maxWinSize;
        NS_LOG_WARN("Adv window size truncated to "
                    << m_maxWinSize << "; possibly to avoid overflow of the 16-bit integer");
    }

    // The method is const because composing a header does not logically alter
    // the socket, but the window actually advertised has to be remembered,
    // both for tracing and for holding the right edge above
    auto self = const_cast<TcpSocketBase*>(this);
    if (w != m_advWnd)
    {
        self->m_advWnd = w;
    }
    if (m_state == CLOSED || m_state == LISTEN || m_state == SYN_SENT)
    {
        // RCV.NXT is set from the SYN of the peer only, so the edge of a
        // window advertised before it came in, such as the one of our SYN,
        // is not to be held
        self->m_advRightEdgeArmed = false;
    }
    else
    {
        // The edge the peer sees, once the field is scaled back
        self->m_advRightEdge = next + (field << shift);
        self->m_advRightEdgeArmed = true;
    }
    NS_LOG_LOGIC("Returning AdvertisedWindowSize of " << field);
    return static_cast<uint16_t>(field);
}

// Receipt of new packet, put into Rx buffer
void
TcpSocketBase::ReceivedData(Ptr<Packet> p, const TcpHeader& tcpHeader)
{
    NS_LOG_FUNCTION(this << tcpHeader);
    NS_LOG_DEBUG("Data segment, seq=" << tcpHeader.GetSequenceNumber()
                                      << " pkt size=" << p->GetSize());

    if (p->GetSize() > 0)
    {
        m_quickAckScheduled = true;
    }
    // As tcp_measure_rcv_mss() of Linux, a full-sized segment with PSH, or a
    // segment shorter than the MSS, likely ends a write of the peer, which
    // may then wait for the ACK
    bool pushed = p->GetSize() > 0 &&
                  (p->GetSize() < m_tcb->m_segmentSize || (tcpHeader.GetFlags() & TcpHeader::PSH));

    // Linux processes the segment, and acknowledges it in most cases, before
    // the application reads the data that it brings: the data read while the
    // segment is processed is still held until the ACK is sent from the read
    // (tcp_cleanup_rbuf())
    struct SoftirqReset
    {
        bool& softirq;   //!< The application has not read yet
        bool& inReceive; //!< Segment being processed

        ~SoftirqReset()
        {
            softirq = false;
            inReceive = false;
        }
    } softirqReset{m_lxSoftirq, m_lxInReceive};

    m_lxInReceive = true;
    if (m_linuxRcvWindow)
    {
        m_lxSoftirqRead = GetRxReadPoint();
        while (!m_lxRcvQueue.empty() && m_lxRcvQueue.front().end <= m_lxSoftirqRead &&
               m_lxRcvQueue.front().seq != m_lxRcvQueue.front().end)
        {
            // Read by the application, the skb was freed
            LinuxMemUncharge(m_lxRcvQueue.front().truesize);
            m_lxRcvQueue.pop_front();
        }
        m_lxSoftirq = true;
    }
    if (m_linuxRcvWindow && m_lxInit && !m_lxEstablished)
    {
        LinuxInitBufferSpace();
    }

    // Put into Rx buffer
    SequenceNumber32 expectedSeq = m_tcb->m_rxBuffer->NextRxSequence();
    const uint32_t ipLen =
        p->GetSize() + tcpHeader.GetSerializedSize() + (m_endPoint6 != nullptr ? 40 : 20);
    std::optional<LinuxRcvSkb> lxSkb;
    if (m_linuxRcvWindow && m_lxInit && p->GetSize() > 0)
    {
        if (!m_lxEstablished)
        {
            LinuxInitBufferSpace();
        }
        SequenceNumber32 seq = tcpHeader.GetSequenceNumber();
        lxSkb = LinuxNewSkb(seq, p->GetSize(), ipLen);
        if (seq + p->GetSize() > expectedSeq && !LinuxTryRmemSchedule(seq, lxSkb->truesize))
        {
            if (seq > expectedSeq)
            {
                // tcp_data_queue_ofo(): out-of-order data dropped, without
                // an ACK
                NS_LOG_LOGIC("No memory for out-of-order data at " << seq);
                return;
            }
            // tcp_data_queue(): the data in sequence is dropped if data is
            // queued, and acknowledged at once with a zero window
            // (ICSK_ACK_NOMEM)
            m_lxNoMem = true;
            if (LinuxRmemAlloc() > 0)
            {
                NS_LOG_LOGIC("No memory for data at " << seq);
                SendEmptyPacket(TcpHeader::ACK);
                return;
            }
            // sk_forced_mem_schedule()
            int64_t delta = static_cast<int64_t>(lxSkb->truesize) - m_lxFwdAlloc;
            if (delta > 0)
            {
                int64_t pages = (delta + LINUX_PAGE_SIZE - 1) / LINUX_PAGE_SIZE;
                m_lxFwdAlloc += pages * LINUX_PAGE_SIZE;
                LinuxMemAllocatedAdd(pages);
            }
        }
    }
    if (!m_linuxRcvWindow && m_pruneOverhead > 0 && p->GetSize() > 0 &&
        tcpHeader.GetSequenceNumber() > expectedSeq &&
        !ScheduleOutOfOrderMemory(tcpHeader.GetSequenceNumber()))
    {
        // As tcp_data_queue_ofo() of Linux, out-of-order data that the
        // memory of the receive buffer cannot hold is dropped
        NS_LOG_LOGIC("Dropping out-of-order data at " << tcpHeader.GetSequenceNumber());
        SendEmptyPacket(TcpHeader::ACK);
        return;
    }
    // As tcp_sack_new_ofo_skb() of Linux, out-of-order data that extends
    // a SACK block other than the first two, or that starts a new block
    // while two are reported already, sends the ACK held back by SACK
    // compression at once, with the SACK blocks of before the segment. The
    // segment itself is then left to the next ACK
    bool compressedAckSent = false;
    if (m_compressedAcks > 0 && p->GetSize() > 0 && tcpHeader.GetSequenceNumber() > expectedSeq)
    {
        SequenceNumber32 head = tcpHeader.GetSequenceNumber();
        SequenceNumber32 tail = head + p->GetSize();
        uint32_t index = 0;
        for (const auto& block : m_tcb->m_rxBuffer->GetSackList())
        {
            if (head <= block.second && block.first <= tail)
            {
                break;
            }
            ++index;
        }
        if (index >= SACK_COMPRESS_FLUSH_BLOCK)
        {
            NS_LOG_LOGIC("SACK block " << index << " changed, sending the ACK held back");
            SackCompressionTimeout();
            compressedAckSent = true;
        }
    }
    if (!m_tcb->m_rxBuffer->Add(p, tcpHeader))
    { // Insert failed: No data or RX buffer full
        if (p->GetSize() > 0 && tcpHeader.GetSequenceNumber() + p->GetSize() <= expectedSeq)
        {
            // Duplicate data: as Linux, stop delaying the ACKs
            EnterQuickAckMode(MAX_QUICK_ACKS);
        }
        if (m_tcb->m_ecnState == TcpSocketState::ECN_CE_RCVD ||
            m_tcb->m_ecnState == TcpSocketState::ECN_SENDING_ECE)
        {
            SendEmptyPacket(TcpHeader::ACK | TcpHeader::ECE);
            NS_LOG_DEBUG(TcpSocketState::EcnStateName[m_tcb->m_ecnState] << " -> ECN_SENDING_ECE");
            m_tcb->m_ecnState = TcpSocketState::ECN_SENDING_ECE;
        }
        else
        {
            SendEmptyPacket(TcpHeader::ACK);
        }
        return;
    }
    const bool inSequence = expectedSeq < m_tcb->m_rxBuffer->NextRxSequence();
    const bool linuxWindow = lxSkb.has_value();
    if (linuxWindow && !inSequence)
    {
        LinuxQueueSegment(*lxSkb, false);
    }
    // Notify app to receive if necessary
    if (inSequence)
    { // NextRxSeq advanced, we have something to send to the app
        UpdateDelAckTimeout();
        // As tcp_event_data_recv() of Linux, the first data, and the first
        // data after an idle period longer than the RTO, are acknowledged
        // quickly, so that the peer opens its window fast
        if (!m_lastDataRxTime || Simulator::Now() - *m_lastDataRxTime > m_rto.Get())
        {
            IncrQuickAck(MAX_QUICK_ACKS);
        }
        // The window grows after the quick ACKs are counted, as Linux
        if (linuxWindow)
        {
            LinuxQueueSegment(*lxSkb, true);
        }
        if (m_rcvSsthresh && p->GetSize() >= 128)
        {
            // As tcp_grow_window() of Linux, the clamped window grows by two
            // segments per segment received in sequence, each acknowledged
            // quickly (the count of quick ACKs or-ed with one)
            *m_rcvSsthresh += std::max(2 * m_tcb->m_segmentSize, 2 * p->GetSize());
            if (*m_rcvSsthresh >= m_tcb->m_rxBuffer->MaxBufferSize())
            {
                m_rcvSsthresh.reset();
            }
            if (m_quickAck)
            {
                m_quickAckCount |= 1;
            }
        }
        m_lastDataRxTime = Simulator::Now();
        if (!m_shutdownRecv)
        {
            NotifyDataRecv();
        }
        // Handle exceptions
        if (m_closeNotified)
        {
            NS_LOG_WARN("Why TCP " << this << " got data after close notification?");
        }
        // If we received FIN before and now completed all "holes" in rx buffer,
        // invoke peer close procedure
        if (m_tcb->m_rxBuffer->Finished() && (tcpHeader.GetFlags() & TcpHeader::FIN) == 0)
        {
            DoPeerClose();
            return;
        }
    }
    // Now send a new ACK packet acknowledging all received and delivered data
    if (m_tcb->m_rxBuffer->Size() > m_tcb->m_rxBuffer->Available() ||
        m_tcb->m_rxBuffer->NextRxSequence() > expectedSeq + p->GetSize())
    { // A gap exists in the buffer, or we filled a gap: Always ACK
        bool hold = false;
        if (m_tcb->m_rxBuffer->NextRxSequence() == expectedSeq)
        {
            hold = compressedAckSent || (!m_ackNow && CompressSackAck());
        }
        else if (m_tcb->m_rxBuffer->Size() > m_tcb->m_rxBuffer->Available() && !m_ackNow &&
                 !LinuxAckDue())
        {
            // As __tcp_ack_snd_check() of Linux, data filling a gap partly,
            // which no other rule acknowledges at once, is acknowledged as
            // out-of-order data: at once, as the first ACK at the new RCV.NXT
            hold = CompressSackAck();
        }
        if (hold)
        {
            m_congestionControl->CwndEvent(m_tcb, TcpSocketState::CA_EVENT_DELAYED_ACK);
            return;
        }
        m_congestionControl->CwndEvent(m_tcb, TcpSocketState::CA_EVENT_NON_DELAYED_ACK);
        if (m_tcb->m_ecnState == TcpSocketState::ECN_CE_RCVD ||
            m_tcb->m_ecnState == TcpSocketState::ECN_SENDING_ECE)
        {
            SendEmptyPacket(TcpHeader::ACK | TcpHeader::ECE);
            NS_LOG_DEBUG(TcpSocketState::EcnStateName[m_tcb->m_ecnState] << " -> ECN_SENDING_ECE");
            m_tcb->m_ecnState = TcpSocketState::ECN_SENDING_ECE;
        }
        else
        {
            SendEmptyPacket(TcpHeader::ACK);
        }
    }
    else
    { // In-sequence packet: ACK if delayed ack count allows, or in quick ACK mode
        // As tcp_cleanup_rbuf() of Linux, a pushed segment is acknowledged at
        // once if the application has read all the data received
        bool pushedAck = m_quickAck && pushed && m_tcb->m_rxBuffer->Available() == 0;
        bool ackNow;
        if (m_linuxRcvWindow && m_lxEstablished)
        {
            // __tcp_ack_snd_check(): more than one segment acknowledged, with
            // a window not smaller than the one advertised, or the quick ACK
            // mode. Otherwise the ACK, if due, follows the read of the
            // application (tcp_cleanup_rbuf()), and advertises the memory
            // that the read freed
            bool due = LinuxAckDue();
            // A DelAckCount of one acknowledges every segment at once
            ackNow = due || m_quickAckCount > 0 || m_lxNoMem || m_ackNow || m_delAckMaxCount <= 1;
            if (!ackNow && DelAckAboutToExpire())
            {
                // tcp_send_delayed_ack(): the timer about to expire, the ACK
                // is sent at once, before the application reads
                ackNow = true;
            }
            if (!ackNow)
            {
                m_lxSoftirq = false;
                ackNow = m_tcb->m_rxBuffer->NextRxSequence() - m_lxRcvWup >
                             static_cast<int32_t>(m_lxRcvMss) ||
                         (m_quickAck && pushed && !m_pingPong && LinuxRmemAlloc() == 0);
            }
        }
        else
        {
            ackNow =
                ++m_delAckCount >= m_delAckMaxCount || m_quickAckCount > 0 || pushedAck || m_ackNow;
        }
        if (ackNow)
        {
            m_delAckCount = 0;
            m_congestionControl->CwndEvent(m_tcb, TcpSocketState::CA_EVENT_NON_DELAYED_ACK);
            if (m_tcb->m_ecnState == TcpSocketState::ECN_CE_RCVD ||
                m_tcb->m_ecnState == TcpSocketState::ECN_SENDING_ECE)
            {
                NS_LOG_DEBUG("Congestion algo " << m_congestionControl->GetName());
                SendEmptyPacket(TcpHeader::ACK | TcpHeader::ECE);
                NS_LOG_DEBUG(TcpSocketState::EcnStateName[m_tcb->m_ecnState]
                             << " -> ECN_SENDING_ECE");
                m_tcb->m_ecnState = TcpSocketState::ECN_SENDING_ECE;
            }
            else
            {
                SendEmptyPacket(TcpHeader::ACK);
            }
        }
        else if (m_delAckPending)
        {
            m_congestionControl->CwndEvent(m_tcb, TcpSocketState::CA_EVENT_DELAYED_ACK);
        }
        else
        {
            m_congestionControl->CwndEvent(m_tcb, TcpSocketState::CA_EVENT_DELAYED_ACK);
            // As sk_reset_timer() of Linux, a timer left running is rearmed
            m_delAckEvent.Cancel();
            m_delAckEvent =
                Simulator::Schedule(GetDelAckDelay(), &TcpSocketBase::DelAckTimeout, this);
            m_delAckPending = true;
            NS_LOG_LOGIC(
                this << " scheduled delayed ACK at "
                     << (Simulator::Now() + Simulator::GetDelayLeft(m_delAckEvent)).GetSeconds());
        }
    }
}

void
TcpSocketBase::SetSackRenegingTimeout(bool enabled)
{
    NS_LOG_FUNCTION(this << enabled);
    m_sackRenegingTimeout = enabled;
    if (m_txBuffer)
    {
        m_txBuffer->SetMarkRenegedHeadLost(!enabled);
    }
}

bool
TcpSocketBase::ScheduleOutOfOrderMemory(SequenceNumber32 seq)
{
    NS_LOG_FUNCTION(this << seq);
    // The memory of the buffered data, as sk_rmem_alloc of Linux, against
    // twice the buffer size, as sk_rcvbuf doubles SO_RCVBUF
    // (LinuxRcvWindow doubles the buffer already)
    const double limit = (m_linuxRcvWindow && m_lxInit) ? static_cast<double>(m_lxRcvBuf)
                                                        : 2.0 * m_tcb->m_rxBuffer->MaxBufferSize();
    auto memory = [this]() { return m_tcb->m_rxBuffer->Size() * m_pruneOverhead; };
    if (memory() <= limit)
    {
        return true;
    }
    // tcp_clamp_window(): the window is clamped to two segments, and no
    // ACK is sent quickly
    NS_LOG_DEBUG("Receive buffer memory " << memory() << " over " << limit);
    if (m_linuxRcvWindow && m_lxInit)
    {
        m_lxRcvSsThresh = std::min(m_lxWindowClamp, 2 * m_lxAdvMss);
    }
    else
    {
        m_rcvSsthresh = 2 * m_tcb->m_segmentSize;
    }
    m_quickAckCount = 0;
    // tcp_prune_ofo_queue(): the out-of-order segments are dropped from the
    // highest, at least an eighth of the memory at a time, until the memory
    // fits the buffer
    double goal = limit / 8;
    bool pruned = false;
    while (uint32_t removed = m_tcb->m_rxBuffer->PruneLastOutOfOrder(seq))
    {
        pruned = true;
        goal -= removed * m_pruneOverhead;
        if (goal <= 0)
        {
            if (memory() <= limit)
            {
                break;
            }
            goal = limit / 8;
        }
    }
    if (pruned)
    {
        // As Linux, the SACK blocks are forgotten: the sender learns about
        // the data dropped (reneging) at its retransmission timeout
        NS_LOG_DEBUG("Pruned the out-of-order queue down to " << m_tcb->m_rxBuffer->Size()
                                                              << " bytes");
        m_tcb->m_rxBuffer->ResetSackList();
    }
    return memory() <= limit;
}

Time
TcpSocketBase::CalculateRttSample(const TcpHeader& tcpHeader, const RttHistory& rttHistory)
{
    NS_LOG_FUNCTION(this);
    SequenceNumber32 ackSeq = tcpHeader.GetAckNumber();
    Time rtt;

    if (ackSeq >= (rttHistory.seq + SequenceNumber32(rttHistory.count)))
    {
        // As per RFC 6298 (Section 3)
        // RTT samples MUST NOT be made using segments that were
        // retransmitted (and thus for which it is ambiguous whether the reply
        // was for the first instance of the packet or a later instance).  The
        // only case when TCP can safely take RTT samples from retransmitted
        // segments is when the TCP timestamp option is employed, since
        // the timestamp option removes the ambiguity regarding which instance
        // of the data segment triggered the acknowledgment.
        if (m_timestampEnabled && tcpHeader.HasOption(TcpOption::TS))
        {
            Ptr<const TcpOptionTS> ts;
            ts = DynamicCast<const TcpOptionTS>(tcpHeader.GetOption(TcpOption::TS));
            rtt = TcpOptionTS::ElapsedTimeFromTsValue(ts->GetEcho() - m_tcb->m_timestampOffset);
            if (rtt.IsZero())
            {
                NS_LOG_LOGIC("TcpSocketBase::EstimateRtt - RTT calculated from TcpOption::TS "
                             "is zero, approximating to 1us.");
                NS_LOG_DEBUG("RTT calculated from TcpOption::TS is zero, updating rtt to 1us.");
                rtt = MicroSeconds(1);
            }
        }
        else if (!rttHistory.retx)
        {
            // Elapsed time since the packet was transmitted
            rtt = Simulator::Now() - rttHistory.time;
        }
    }
    return rtt;
}

void
TcpSocketBase::StatePktsAcked(uint32_t segmentsAcked)
{
    if (!m_linuxPktsAcked)
    {
        m_congestionControl->PktsAcked(m_tcb, segmentsAcked, m_ackRttSample);
    }
}

void
TcpSocketBase::UpdateLinuxCaRtt(const TcpHeader& tcpHeader, bool advanced, bool sacked)
{
    NS_LOG_FUNCTION(this << advanced << sacked);

    // Linux tcp_clean_rtx_queue(): ca_rtt_us from the newest segment
    // cumulatively acknowledged (never retransmitted), then from the newest
    // one newly SACKed; tcp_ack_update_rtt() falls back to the timestamps
    // when there is no such sample and the ACK acknowledges new data
    Time rtt(0);
    Time acked = m_txBuffer->GetNewestCumAckedSentTime();
    if (advanced && acked != Time::Min())
    {
        rtt = Simulator::Now() - acked;
    }
    Time newlySacked = m_txBuffer->GetNewestNewlySackedSentTime();
    if (sacked && newlySacked != Time::Min())
    {
        rtt = Simulator::Now() - newlySacked;
    }
    if (rtt.IsZero() && advanced && m_timestampEnabled && tcpHeader.HasOption(TcpOption::TS))
    {
        auto ts = DynamicCast<const TcpOptionTS>(tcpHeader.GetOption(TcpOption::TS));
        rtt = TcpOptionTS::ElapsedTimeFromTsValue(ts->GetEcho() - m_tcb->m_timestampOffset);
        if (rtt.IsZero())
        {
            rtt = MicroSeconds(1);
        }
    }
    NS_LOG_DEBUG("RTT sample of the ACK for the congestion control: " << rtt);
    m_ackRttSample = rtt;
    if (rtt.IsStrictlyPositive())
    {
        m_tcb->m_lastRtt = rtt;
        m_tcb->m_minRtt = std::min(m_tcb->m_minRtt, rtt);
    }
}

void
TcpSocketBase::EstimateRtt(const TcpHeader& tcpHeader)
{
    NS_LOG_FUNCTION(this);
    SequenceNumber32 ackSeq = tcpHeader.GetAckNumber();
    Time rtt;
    // An ACK which gives no new RTT sample gives none to the congestion
    // control either (Linux passes rtt_us = -1 to pkts_acked())
    m_ackRttSample = Time(0);

    // An ack has been received, calculate rtt and log this measurement
    // Note we use a linear search (O(n)) for this since for the common
    // case the ack'ed packet will be at the head of the list
    if (!m_history.empty())
    {
        RttHistory& earliestTransmittedPktHistory = m_history.front();
        rtt = CalculateRttSample(tcpHeader, earliestTransmittedPktHistory);
        if (m_linuxCaRtt && ackSeq >= earliestTransmittedPktHistory.seq +
                                          SequenceNumber32(earliestTransmittedPktHistory.count))
        {
            // As tcp_clean_rtx_queue() of Linux (seq_rtt_us), the sample is
            // the time since the transmission of the oldest segment newly
            // acknowledged, unless the ACK covers retransmitted data: the
            // timestamps are then the only unambiguous source
            bool retransAcked = false;
            for (const auto& h : m_history)
            {
                if (h.seq + SequenceNumber32(h.count) > ackSeq)
                {
                    break;
                }
                retransAcked = retransAcked || h.retx;
            }
            if (!retransAcked)
            {
                rtt = Simulator::Now() - earliestTransmittedPktHistory.time;
            }
        }

        // Store ACKed packet that has the latest transmission time to update `lastRtt`
        RttHistory latestTransmittedPktHistory = earliestTransmittedPktHistory;

        // Delete all ACK history with seq <= ack
        while (!m_history.empty())
        {
            RttHistory& rttHistory = m_history.front();
            if ((rttHistory.seq + SequenceNumber32(rttHistory.count)) > ackSeq)
            {
                break; // Done removing
            }

            latestTransmittedPktHistory = rttHistory;
            m_history.pop_front(); // Remove
        }

        // In case of multiple packets being ACKed in a single acknowledgement, `m_lastRtt` is
        // RTT of the last (S)ACKed packet calculated using the data packet with the latest
        // transmission time. Its transmission time gives that RTT exactly, unless it was
        // retransmitted: the echoed timestamp has a granularity of one millisecond, and
        // belongs to the oldest segment of a delayed ACK (as Linux, which feeds the
        // congestion control with the timestamps only when no clean sample is available)
        Time lastRtt;
        if (!latestTransmittedPktHistory.retx &&
            ackSeq >= latestTransmittedPktHistory.seq +
                          SequenceNumber32(latestTransmittedPktHistory.count))
        {
            lastRtt = Simulator::Now() - latestTransmittedPktHistory.time;
        }
        else
        {
            lastRtt = CalculateRttSample(tcpHeader, latestTransmittedPktHistory);
        }
        if (!lastRtt.IsZero())
        {
            NS_LOG_DEBUG("Last RTT sample updated to: " << lastRtt);
            m_tcb->m_lastRtt = lastRtt;
            m_ackRttSample = lastRtt;
        }
    }

    if (m_linuxCaRtt && rtt.IsZero() && ackSeq > m_txBuffer->HeadSequence() && m_timestampEnabled &&
        tcpHeader.HasOption(TcpOption::TS))
    {
        // As tcp_ack_update_rtt() of Linux, an ACK of new data without a clean
        // sample, such as after a retransmission timeout, takes the echoed
        // timestamp (tcp_rtt_tsopt_us())
        auto ts = DynamicCast<const TcpOptionTS>(tcpHeader.GetOption(TcpOption::TS));
        rtt = TcpOptionTS::ElapsedTimeFromTsValue(ts->GetEcho() - m_tcb->m_timestampOffset);
        if (!rtt.IsZero() && m_ackRttSample.IsZero())
        {
            m_tcb->m_lastRtt = rtt;
            m_ackRttSample = rtt;
        }
    }

    if (!rtt.IsZero())
    {
        UpdateRttEstimate(rtt);
    }
}

void
TcpSocketBase::UpdateRttEstimate(const Time& rtt)
{
    NS_LOG_FUNCTION(this << rtt);
    m_rtt->Measurement(rtt); // Log the measurement
    // RFC 6298, clause 2.4
    m_rto =
        Max(m_rtt->GetEstimate() + Max(m_clockGranularity, m_rtt->GetVariation() * 4), m_minRto);
    m_tcb->m_srtt = m_rtt->GetEstimate();
    // With LinuxCaRtt, the minimum RTT is the one of the samples, as Linux
    // tcp_update_rtt_min(), rather than the one of the smoothed RTT
    m_tcb->m_minRtt = std::min(m_linuxCaRtt ? rtt : m_tcb->m_srtt.Get(), m_tcb->m_minRtt);
    NS_LOG_INFO(this << m_tcb->m_srtt << m_tcb->m_minRtt);
}

// Called by the ReceivedAck() when new ACK received and by ProcessSynRcvd()
// when the three-way handshake completed. This cancels retransmission timer
// and advances Tx window
void
TcpSocketBase::NewAck(const SequenceNumber32& ack, bool resetRTO)
{
    NS_LOG_FUNCTION(this << ack);

    // Reset the data retransmission count. We got a new ACK!
    m_dataRetrCount = m_dataRetries;

    // Update m_sndFack if possible
    if (m_fackEnabled && ack > SequenceNumber32(m_sndFack))
    {
        NS_LOG_INFO(" m_sndFack " << m_sndFack << " updated by normal ack to " << ack.GetValue());
        m_sndFack = ack.GetValue();
    }

    if (m_state != SYN_RCVD && resetRTO)
    { // Set RTO unless the ACK is received in SYN_RCVD state
        NS_LOG_LOGIC(
            this << " Cancelled ReTxTimeout event which was set to expire at "
                 << (Simulator::Now() + Simulator::GetDelayLeft(m_retxEvent)).GetSeconds());
        m_retxEvent.Cancel();
        // On receiving a "New" ack we restart retransmission timer .. RFC 6298
        // RFC 6298, clause 2.4
        m_rto = Max(m_rtt->GetEstimate() + Max(m_clockGranularity, m_rtt->GetVariation() * 4),
                    m_minRto);

        NS_LOG_LOGIC(this << " Schedule ReTxTimeout at time " << Simulator::Now().GetSeconds()
                          << " to expire at time "
                          << (Simulator::Now() + m_rto.Get()).GetSeconds());
        m_retxEvent = Simulator::Schedule(GetRtoTimerDelay(), &TcpSocketBase::ReTxTimeout, this);
    }
    if (m_sackRenegingTimeout && m_txBuffer->GetRenegingEvents() != m_renegingEvents)
    {
        // As tcp_check_sack_reneging() of Linux, the receiver reneged on its
        // SACKs: the retransmission timer expires after half the smoothed
        // RTT, at least 10 ms, to recover from the timeout
        m_renegingEvents = m_txBuffer->GetRenegingEvents();
        m_newDataOnly = true;
        Time delay = std::max(m_tcb->m_srtt.Get() / 2, MilliSeconds(10));
        NS_LOG_DEBUG("SACK reneging: retransmission timeout in " << delay.As(Time::MS));
        m_retxEvent.Cancel();
        if (m_rtoTimerGranularity.IsStrictlyPositive())
        {
            // On the ticks of the timer wheel, as the retransmission timer
            delay = GetTickTimerDelay(DivideRoundUp(delay, m_rtoTimerGranularity));
        }
        m_retxEvent = Simulator::Schedule(delay, &TcpSocketBase::ReTxTimeout, this);
    }

    // Note the highest ACK and tell app to send more
    NS_LOG_LOGIC("TCP " << this << " NewAck " << ack << " numberAck "
                        << (ack - m_txBuffer->HeadSequence())); // Number bytes ack'ed

    // With the send buffer of Linux, the application waiting for space is
    // woken after the transmissions of the ACK (tcp_check_space())
    if (GetTxAvailable() > 0 && !(m_sndSkbOverhead > 0 && m_lxNoSpace))
    {
        NotifySend(GetTxAvailable());
    }
    if (ack > m_tcb->m_nextTxSequence)
    {
        m_tcb->m_nextTxSequence = ack; // If advanced
    }
    if (m_txBuffer->Size() == 0 && m_state != FIN_WAIT_1 && m_state != CLOSING)
    { // No retransmit timer if no data to retransmit
        NS_LOG_LOGIC(
            this << " Cancelled ReTxTimeout event which was set to expire at "
                 << (Simulator::Now() + Simulator::GetDelayLeft(m_retxEvent)).GetSeconds());
        m_retxEvent.Cancel();
    }
}

// Retransmit timeout
void
TcpSocketBase::ReTxTimeout()
{
    NS_LOG_FUNCTION(this);
    NS_LOG_LOGIC(this << " ReTxTimeout Expired at time " << Simulator::Now().GetSeconds());
    // If erroneous timeout in closed/timed-wait state, just return
    if (m_state == CLOSED || m_state == TIME_WAIT)
    {
        return;
    }

    if (m_state == SYN_SENT)
    {
        NS_ASSERT(m_synCount > 0);
        if (m_tcb->m_useEcn == TcpSocketState::On)
        {
            SendEmptyPacket(TcpHeader::SYN | TcpHeader::ECE | TcpHeader::CWR);
        }
        else
        {
            SendEmptyPacket(TcpHeader::SYN);
        }
        return;
    }

    // Retransmit non-data packet: Only if in FIN_WAIT_1 or CLOSING state
    if (m_txBuffer->Size() == 0)
    {
        if (m_state == FIN_WAIT_1 || m_state == CLOSING)
        { // Must have lost FIN, re-send
            SendEmptyPacket(TcpHeader::FIN);
        }
        return;
    }

    NS_LOG_DEBUG("Checking if Connection is Established");
    // If all data are received (non-closing socket and nothing to send), just return
    if (m_state <= ESTABLISHED && m_txBuffer->HeadSequence() >= m_tcb->m_highTxMark &&
        m_txBuffer->Size() == 0)
    {
        NS_LOG_DEBUG("Already Sent full data" << m_txBuffer->HeadSequence() << " "
                                              << m_tcb->m_highTxMark);
        return;
    }

    if (m_dataRetrCount == 0)
    {
        NS_LOG_INFO("No more data retries available. Dropping connection");
        NotifyErrorClose();
        DeallocateEndPoint();
        return;
    }
    else
    {
        --m_dataRetrCount;
    }

    // RFC 5681, equation 4: the FlightSize when the timer expires, bounded
    // by cwnd (see CongestionFlightSize())
    uint32_t inFlightBeforeRto = CongestionFlightSize();
    if (m_tcb->m_congState != TcpSocketState::CA_RECOVERY &&
        m_tcb->m_congState != TcpSocketState::CA_LOSS)
    {
        InitUndo();
    }
    m_undoPending = false;
    // As Linux tcp_enter_loss(), whose recovery point is high_seq: the
    // window was already reduced for this loss episode
    bool keepSsThresh = m_keepRecoverySsThresh &&
                        (m_tcb->m_congState == TcpSocketState::CA_RECOVERY ||
                         m_tcb->m_congState == TcpSocketState::CA_CWR) &&
                        m_recover > m_txBuffer->HeadSequence();
    bool resetSack = !m_sackEnabled; // Reset SACK information if SACK is not enabled.
                                     // The information in the TcpTxBuffer is guessed, in this case.
    if (m_sackEnabled && m_txBuffer->IsSackReneged())
    {
        // The receiver reneged on its SACKs: they are all discarded (RFC
        // 2018, Section 8, and RFC 6675, Section 5.1), as
        // tcp_timeout_mark_lost() of Linux does
        NS_LOG_DEBUG("Discarding the SACK information after reneging");
        resetSack = true;
    }

    // Reset dupAckCount
    m_dupAckCount = 0;
    if (!m_sackEnabled)
    {
        m_txBuffer->ResetRenoSack();
    }

    // From RFC 6675, Section 5.1
    // [RFC2018] suggests that a TCP sender SHOULD expunge the SACK
    // information gathered from a receiver upon a retransmission timeout
    // (RTO) "since the timeout might indicate that the data receiver has
    // reneged."  Additionally, a TCP sender MUST "ignore prior SACK
    // information in determining which data to retransmit."
    // It has been suggested that, as long as robust tests for
    // reneging are present, an implementation can retain and use SACK
    // information across a timeout event [Errata1610].
    // The head of the sent list will not be marked as sacked, therefore
    // will be retransmitted, if the receiver renegotiate the SACK blocks
    // that we received.
    m_txBuffer->SetSentListLost(resetSack);

    // From RFC 6675, Section 5.1
    // If an RTO occurs during loss recovery as specified in this document,
    // RecoveryPoint MUST be set to HighData.  Further, the new value of
    // RecoveryPoint MUST be preserved and the loss recovery algorithm
    // outlined in this document MUST be terminated.
    m_recover = m_tcb->m_highTxMark;
    m_recoverActive = true;

    // RFC 6298, clause 2.5, double the timer. Linux doubles the RTO it
    // holds in whole ticks
    Time rto = m_rto;
    if (m_rtoTimerGranularity.IsStrictlyPositive())
    {
        rto = m_rtoTimerGranularity * DivideRoundUp(rto, m_rtoTimerGranularity);
    }
    Time doubledRto = rto + rto;
    m_rto = Min(doubledRto, Time::FromDouble(60, Time::S));

    // Empty RTT history
    m_history.clear();

    // Please don't reset highTxMark, it is used for retransmission detection

    // When a TCP sender detects segment loss using the retransmission timer
    // and the given segment has not yet been resent by way of the
    // retransmission timer, decrease ssThresh
    if (keepSsThresh)
    {
        NS_LOG_DEBUG("RTO during the recovery, ssThresh kept at " << m_tcb->m_ssThresh);
    }
    else if (m_tcb->m_congState != TcpSocketState::CA_LOSS || !m_txBuffer->IsHeadRetransmitted())
    {
        m_tcb->m_ssThresh = m_congestionControl->GetSsThresh(m_tcb, inFlightBeforeRto);
    }

    QueueCwr();

    // Cwnd set to 1 MSS
    m_congestionControl->CwndEvent(m_tcb, TcpSocketState::CA_EVENT_LOSS);
    m_congestionControl->CongestionStateSet(m_tcb, TcpSocketState::CA_LOSS);
    m_tcb->m_congState = TcpSocketState::CA_LOSS;
    m_tcb->m_cWnd = m_tcb->m_segmentSize;
    m_tcb->m_cWndInfl = m_tcb->m_cWnd;

    m_pacingTimer.Cancel();

    NS_LOG_DEBUG("RTO. Reset cwnd to " << m_tcb->m_cWnd << ", ssthresh to " << m_tcb->m_ssThresh
                                       << ", restart from seqnum " << m_txBuffer->HeadSequence()
                                       << " doubled rto to " << m_rto.Get().GetSeconds() << " s");

    NS_ASSERT_MSG(BytesInFlight() == 0,
                  "There are some bytes in flight after an RTO: " << BytesInFlight());

    SendPendingData(m_connected);

    NS_ASSERT_MSG(BytesInFlight() <= m_tcb->m_segmentSize,
                  "In flight (" << BytesInFlight() << ") there is more than one segment ("
                                << m_tcb->m_segmentSize << ")");
}

int64_t
TcpSocketBase::DivideRoundUp(const Time& t, const Time& tick)
{
    return (t.GetTimeStep() + tick.GetTimeStep() - 1) / tick.GetTimeStep();
}

Time
TcpSocketBase::GetRtoTimerDelay() const
{
    if (!m_rtoTimerGranularity.IsStrictlyPositive())
    {
        return m_rto;
    }
    return GetTickTimerDelay(DivideRoundUp(m_rto, m_rtoTimerGranularity));
}

Time
TcpSocketBase::GetTickTimerDelay(int64_t ticks) const
{
    NS_ASSERT(m_rtoTimerGranularity.IsStrictlyPositive());
    // As the Linux timer wheel (calc_wheel_index() and calc_index() in
    // kernel/time/timer.c), with the tick counter starting at zero with the
    // simulation: a level covers 63 times its granularity, which is 8 times
    // the one of the level below
    const int64_t now = Simulator::Now().GetTimeStep() / m_rtoTimerGranularity.GetTimeStep();
    uint32_t shift = 0;
    while (shift < 21 && ticks >= (int64_t{63} << shift))
    {
        shift += 3;
    }
    const int64_t expiry = (((now + ticks) >> shift) + 1) << shift;
    return m_rtoTimerGranularity * expiry - Simulator::Now();
}

void
TcpSocketBase::DelAckTimeout()
{
    NS_LOG_FUNCTION(this);
    // As tcp_delack_timer_handler() of Linux, the timer, which an ACK sent
    // leaves running with SackCompression, sends the ACK held back by SACK
    // compression, if any, or else the delayed ACK, if still pending
    if (m_compressedAcks > 0)
    {
        SackCompressionTimeout();
        return;
    }
    if (!m_delAckPending)
    {
        return;
    }
    m_delAckPending = false;
    if (m_pingPong)
    {
        // As tcp_delack_timer_handler() of Linux: the interactive mode ends
        m_pingPong = false;
        m_ato = MilliSeconds(ATO_MIN_MS);
    }
    else if (m_adaptiveDelAck && !m_ato.IsZero())
    {
        // As tcp_delack_timer_handler() of Linux: the timeout was too short
        m_ato = std::min(2 * m_ato, m_rto.Get());
    }
    m_delAckCount = 0;
    m_congestionControl->CwndEvent(m_tcb, TcpSocketState::CA_EVENT_DELAYED_ACK);
    if (m_tcb->m_ecnState == TcpSocketState::ECN_CE_RCVD ||
        m_tcb->m_ecnState == TcpSocketState::ECN_SENDING_ECE)
    {
        SendEmptyPacket(TcpHeader::ACK | TcpHeader::ECE);
        m_tcb->m_ecnState = TcpSocketState::ECN_SENDING_ECE;
    }
    else
    {
        SendEmptyPacket(TcpHeader::ACK);
    }
}

void
TcpSocketBase::EnterQuickAckMode(uint32_t maxQuickAcks)
{
    NS_LOG_FUNCTION(this << maxQuickAcks);
    IncrQuickAck(maxQuickAcks);
    if (m_ato > MilliSeconds(ATO_MIN_MS))
    {
        m_ato = MilliSeconds(ATO_MIN_MS);
    }
}

void
TcpSocketBase::IncrQuickAck(uint32_t maxQuickAcks)
{
    NS_LOG_FUNCTION(this << maxQuickAcks);
    if (!m_quickAck)
    {
        return;
    }
    // The window last advertised, as rcv_wnd of Linux, and the segment size
    // of the connection standing for the receive MSS that Linux estimates
    uint32_t rcvWnd =
        std::min(m_advWnd.Get(), static_cast<uint32_t>(m_maxWinSize) << m_rcvWindShift);
    uint32_t quickAcks = rcvWnd / (2 * m_tcb->m_segmentSize);
    if (quickAcks == 0)
    {
        quickAcks = 2;
    }
    quickAcks = std::min(quickAcks, maxQuickAcks);
    if (quickAcks > m_quickAckCount)
    {
        NS_LOG_DEBUG("Quick ACKs left: " << m_quickAckCount << " -> " << quickAcks);
        m_quickAckCount = quickAcks;
    }
}

void
TcpSocketBase::QuickAckSent()
{
    NS_LOG_FUNCTION(this);
    if (!m_quickAckScheduled)
    {
        return;
    }
    m_quickAckScheduled = false;
    if (m_quickAckCount > 0)
    {
        --m_quickAckCount;
        NS_LOG_LOGIC("Quick ACKs left: " << m_quickAckCount);
        if (m_quickAckCount == 0 && m_ato > MilliSeconds(ATO_MIN_MS))
        {
            // As tcp_dec_quickack_mode() of Linux
            m_ato = MilliSeconds(ATO_MIN_MS);
        }
    }
}

bool
TcpSocketBase::LinuxAckDue()
{
    NS_LOG_FUNCTION(this);
    if (!m_linuxRcvWindow || !m_lxInit)
    {
        return true;
    }
    SequenceNumber32 rcvNxt = m_tcb->m_rxBuffer->NextRxSequence();
    if (rcvNxt - m_lxRcvWup <= static_cast<int32_t>(m_lxRcvMss))
    {
        return false;
    }
    SequenceNumber32 read = m_lxSoftirq ? m_lxSoftirqRead : GetRxReadPoint();
    if (rcvNxt - read < 1)
    {
        // The application has read all the data (SO_RCVLOWAT of one byte)
        return true;
    }
    return LinuxSelectWindow() >= m_lxRcvWnd;
}

bool
TcpSocketBase::CompressSackAck()
{
    NS_LOG_FUNCTION(this);
    if (!m_sackCompression || !m_sackEnabled || m_quickAckCount > 0 ||
        m_tcb->m_ecnState == TcpSocketState::ECN_CE_RCVD || m_compressedAcks >= SACK_COMPRESS_MAX)
    {
        return false;
    }
    SequenceNumber32 rcvNxt = m_tcb->m_rxBuffer->NextRxSequence();
    if (m_compressRcvNxt != rcvNxt)
    {
        m_compressRcvNxt = rcvNxt;
        m_compressDupAcks = 0;
    }
    if (m_compressDupAcks < SACK_COMPRESS_DUPACKS)
    {
        ++m_compressDupAcks;
        return false;
    }
    ++m_compressedAcks;
    if (m_sackCompressionEvent.IsPending())
    {
        NS_LOG_LOGIC("ACK of out-of-order data merged, " << m_compressedAcks << " held back");
        return true;
    }
    Time delay = m_sackCompressionDelay;
    if (m_tcb->m_srtt.Get().IsStrictlyPositive())
    {
        delay = std::min(delay, m_tcb->m_srtt.Get() / 20);
    }
    // The range timer of Linux expires at the end of its range
    delay += m_sackCompressionSlack;
    NS_LOG_LOGIC("ACK of out-of-order data held back for " << delay.As(Time::US));
    m_sackCompressionEvent =
        Simulator::Schedule(delay, &TcpSocketBase::SackCompressionTimeout, this);
    return true;
}

void
TcpSocketBase::SackCompressionTimeout()
{
    NS_LOG_FUNCTION(this);
    if (m_tcb->m_ecnState == TcpSocketState::ECN_CE_RCVD ||
        m_tcb->m_ecnState == TcpSocketState::ECN_SENDING_ECE)
    {
        SendEmptyPacket(TcpHeader::ACK | TcpHeader::ECE);
        m_tcb->m_ecnState = TcpSocketState::ECN_SENDING_ECE;
    }
    else
    {
        SendEmptyPacket(TcpHeader::ACK);
    }
}

void
TcpSocketBase::UpdateDelAckTimeout()
{
    NS_LOG_FUNCTION(this);
    if (!m_adaptiveDelAck)
    {
        return;
    }
    if (!m_lastDataRxTime || m_ato.IsZero())
    {
        m_ato = MilliSeconds(ATO_MIN_MS);
    }
    else if (m_rtoTimerGranularity.IsStrictlyPositive())
    {
        // As Linux, in whole ticks (jiffies): the interval is the number of
        // ticks since the last data, and the halved timeout is rounded down
        const Time& tick = m_rtoTimerGranularity;
        const int64_t step = tick.GetTimeStep();
        int64_t interval =
            Simulator::Now().GetTimeStep() / step - m_lastDataRxTime->GetTimeStep() / step;
        int64_t ato = DivideRoundUp(m_ato, tick);
        if (interval <= DivideRoundUp(MilliSeconds(ATO_MIN_MS), tick) / 2)
        {
            ato = ato / 2 + DivideRoundUp(MilliSeconds(ATO_MIN_MS), tick) / 2;
        }
        else if (interval < ato)
        {
            ato = std::min(ato / 2 + interval, DivideRoundUp(m_rto, tick));
        }
        m_ato = tick * ato;
    }
    else
    {
        Time interval = Simulator::Now() - *m_lastDataRxTime;
        if (interval <= MilliSeconds(ATO_MIN_MS / 2))
        {
            m_ato = m_ato / 2 + MilliSeconds(ATO_MIN_MS / 2);
        }
        else if (interval < m_ato)
        {
            m_ato = std::min(m_ato / 2 + interval, m_rto.Get());
        }
    }
    NS_LOG_LOGIC("Delayed ACK timeout " << m_ato.As(Time::MS));
}

Time
TcpSocketBase::GetDelAckAto() const
{
    const bool ticks = m_rtoTimerGranularity.IsStrictlyPositive();
    Time ato = m_ato;
    if (ato > MilliSeconds(ATO_MIN_MS))
    {
        // TCP_DELACK_MAX (200 ms) in the interactive mode
        Time maxAto = MilliSeconds(m_pingPong ? 200 : ATO_MAX_MS);
        if (m_tcb->m_srtt.Get().IsStrictlyPositive())
        {
            // usecs_to_jiffies() rounds the smoothed RTT up to whole ticks
            Time srtt = ticks ? m_rtoTimerGranularity *
                                    DivideRoundUp(m_tcb->m_srtt.Get(), m_rtoTimerGranularity)
                              : m_tcb->m_srtt.Get();
            maxAto = std::min(maxAto, std::max(srtt, MilliSeconds(ATO_MIN_MS)));
        }
        ato = std::min(ato, maxAto);
    }
    ato = std::min(ato, m_delAckTimeout);
    if (ticks)
    {
        ato = m_rtoTimerGranularity * DivideRoundUp(ato, m_rtoTimerGranularity);
    }
    return ato;
}

Time
TcpSocketBase::GetDelAckDelay()
{
    if (!m_adaptiveDelAck || m_ato.IsZero())
    {
        return m_delAckTimeout;
    }
    Time ato = GetDelAckAto();
    if (m_rtoTimerGranularity.IsStrictlyPositive())
    {
        // The timeout requested, counted from the last tick
        const int64_t step = m_rtoTimerGranularity.GetTimeStep();
        m_delAckExpiry = TimeStep(Simulator::Now().GetTimeStep() / step * step) + ato;
        return GetTickTimerDelay(DivideRoundUp(ato, m_rtoTimerGranularity));
    }
    m_delAckExpiry = Simulator::Now() + ato;
    return ato;
}

bool
TcpSocketBase::DelAckAboutToExpire() const
{
    if (!m_adaptiveDelAck || !m_delAckPending || m_ato.IsZero())
    {
        return false;
    }
    // As tcp_send_delayed_ack() of Linux: within a quarter of the timeout
    Time ato = GetDelAckAto();
    if (m_rtoTimerGranularity.IsStrictlyPositive())
    {
        const int64_t step = m_rtoTimerGranularity.GetTimeStep();
        const int64_t now = Simulator::Now().GetTimeStep() / step;
        return m_delAckExpiry.GetTimeStep() / step <= now + (ato.GetTimeStep() / step >> 2);
    }
    return m_delAckExpiry <= Simulator::Now() + ato / 4;
}

void
TcpSocketBase::StopDelAckTimer()
{
    m_delAckPending = false;
    if (!m_sackCompression)
    {
        m_delAckEvent.Cancel();
    }
}

void
TcpSocketBase::LastAckTimeout()
{
    NS_LOG_FUNCTION(this);

    m_lastAckEvent.Cancel();
    if (m_state == LAST_ACK)
    {
        if (m_dataRetrCount == 0)
        {
            NS_LOG_INFO("LAST-ACK: No more data retries available. Dropping connection");
            NotifyErrorClose();
            DeallocateEndPoint();
            return;
        }
        m_dataRetrCount--;
        SendEmptyPacket(TcpHeader::FIN | TcpHeader::ACK);
        NS_LOG_LOGIC("TcpSocketBase " << this << " rescheduling LATO1");
        Time lastRto = m_rtt->GetEstimate() + Max(m_clockGranularity, m_rtt->GetVariation() * 4);
        m_lastAckEvent = Simulator::Schedule(lastRto, &TcpSocketBase::LastAckTimeout, this);
    }
}

// Send 1-byte data (or a zero-length segment, if there is no unsent data) to
// probe for the window size at the receiver when the local knowledge tells
// that the receiver has zero window size
// C.f.: RFC793 p.42, RFC1122 sec.4.2.2.17, RFC9293 sec.3.8.6.1
void
TcpSocketBase::PersistTimeout()
{
    NS_LOG_LOGIC("PersistTimeout expired at " << Simulator::Now().GetSeconds());
    m_persistTimeout =
        std::min(Seconds(60), Time(2 * m_persistTimeout)); // max persist timeout = 60s
    // CopyFromSequence() returns nullptr when there is no unsent data. In that
    // case, send a zero-length probe with an already acknowledged sequence
    // number. The receiver finds it unacceptable and must answer with an ACK
    // carrying its current window (@RFC{9293} sec 3.10.7.4, @issueid{1326}).
    Ptr<Packet> p;
    SequenceNumber32 probeSeq = m_tcb->m_nextTxSequence;
    TcpTxItem* outItem = m_txBuffer->CopyFromSequence(1, m_tcb->m_nextTxSequence);
    if (outItem != nullptr)
    {
        p = outItem->GetPacketCopy();
        // The probe byte is not accounted as sent; return it to the unsent data
        m_txBuffer->ResetLastSegmentSent();
    }
    else
    {
        p = Create<Packet>();
        probeSeq = m_txBuffer->HeadSequence() - 1;
    }
    TcpHeader tcpHeader;
    // Like every segment of a synchronized connection, the probe carries an
    // acknowledgment; the peer discards a segment without the ACK bit
    // (RFC 9293, Section 3.10.7.4). It also carries the urgent field while
    // urgent data is ahead, and PSH when its octet is the last one queued,
    // like any other segment (RFC 9293, Section 3.8.5 and MUST-61)
    tcpHeader.SetFlags(TcpHeader::ACK | SetPushAndUrgent(tcpHeader, probeSeq, p->GetSize()));
    tcpHeader.SetSequenceNumber(probeSeq);
    tcpHeader.SetAckNumber(m_tcb->m_rxBuffer->NextRxSequence());
    tcpHeader.SetWindowSize(AdvertisedWindowSize());
    if (m_endPoint != nullptr)
    {
        tcpHeader.SetSourcePort(m_endPoint->GetLocalPort());
        tcpHeader.SetDestinationPort(m_endPoint->GetPeerPort());
    }
    else
    {
        tcpHeader.SetSourcePort(m_endPoint6->GetLocalPort());
        tcpHeader.SetDestinationPort(m_endPoint6->GetPeerPort());
    }
    AddOptions(tcpHeader);
    // The probe follows the source route of the connection, and carries the
    // IP settings of the socket, like the other segments
    AddSocketTags(p, IsEct(TcpPacketType_t::WINDOW_PROBE));
    m_txTrace(p, tcpHeader, this);

    if (m_endPoint != nullptr)
    {
        m_tcp->SendPacket(p,
                          tcpHeader,
                          m_endPoint->GetLocalAddress(),
                          m_endPoint->GetPeerAddress(),
                          m_boundnetdevice);
    }
    else
    {
        m_tcp->SendPacket(p,
                          tcpHeader,
                          m_endPoint6->GetLocalAddress(),
                          m_endPoint6->GetPeerAddress(),
                          m_boundnetdevice);
    }

    NS_LOG_LOGIC("Schedule persist timeout at time "
                 << Simulator::Now().GetSeconds() << " to expire at time "
                 << (Simulator::Now() + m_persistTimeout).GetSeconds());
    m_persistEvent = Simulator::Schedule(m_persistTimeout, &TcpSocketBase::PersistTimeout, this);
}

void
TcpSocketBase::DoRetransmit()
{
    NS_LOG_FUNCTION(this);
    bool res;
    SequenceNumber32 seq;
    SequenceNumber32 seqHigh;
    uint32_t maxSizeToSend;

    // Find the first segment marked as lost and not retransmitted. With Reno,
    // that should be the head
    res = m_txBuffer->NextSeg(&seq, &seqHigh, false);
    if (!res)
    {
        // We have already retransmitted the head. However, we still received
        // three dupacks, or the RTO expired, but no data to transmit.
        // Therefore, re-send again the head.
        seq = m_txBuffer->HeadSequence();
        maxSizeToSend = m_tcb->m_segmentSize;
    }
    else
    {
        // NextSeg() may constrain the segment size when res is true
        maxSizeToSend = static_cast<uint32_t>(seqHigh - seq);
    }
    NS_ASSERT(m_sackEnabled || seq == m_txBuffer->HeadSequence());

    NS_LOG_INFO("Retransmitting " << seq);
    // Update the trace and retransmit the segment
    m_tcb->m_nextTxSequence = seq;
    uint32_t sz = SendDataPacket(m_tcb->m_nextTxSequence, maxSizeToSend, true);

    NS_ASSERT(sz > 0);
}

void
TcpSocketBase::CancelAllTimers()
{
    m_retxEvent.Cancel();
    m_persistEvent.Cancel();
    m_delAckEvent.Cancel();
    m_delAckPending = false;
    m_sackCompressionEvent.Cancel();
    m_lastAckEvent.Cancel();
    m_timewaitEvent.Cancel();
    m_sendPendingDataEvent.Cancel();
    m_pacingTimer.Cancel();
    m_keepAliveEvent.Cancel();
}

/* Move TCP to Time_Wait state and schedule a transition to Closed state */
void
TcpSocketBase::TimeWait()
{
    NS_LOG_DEBUG(TcpStateName[m_state] << " -> TIME_WAIT");
    m_state = TIME_WAIT;
    CancelAllTimers();
    if (!m_closeNotified)
    {
        // Technically the connection is not fully closed, but we notify now
        // because an implementation (real socket) would behave as if closed.
        // Notify normal close when entering TIME_WAIT or leaving LAST_ACK.
        NotifyNormalClose();
        m_closeNotified = true;
    }
    // Move from TIME_WAIT to CLOSED after 2*MSL. Max segment lifetime is 2 min
    // according to RFC793, p.28
    m_timewaitEvent = Simulator::Schedule(Seconds(2 * m_msl), &TcpSocketBase::CloseAndNotify, this);
}

/* Below are the attribute get/set functions */

void
TcpSocketBase::SetSndBufSize(uint32_t size)
{
    NS_LOG_FUNCTION(this << size);
    m_txBuffer->SetMaxBufferSize(size);
}

uint32_t
TcpSocketBase::GetSndBufSize() const
{
    return m_txBuffer->MaxBufferSize();
}

uint32_t
TcpSocketBase::GetSndFack() const
{
    return m_sndFack;
}

bool
TcpSocketBase::GetFackEnabled() const
{
    return m_fackEnabled;
}

void
TcpSocketBase::SetRcvBufSize(uint32_t size)
{
    NS_LOG_FUNCTION(this << size);
    uint32_t oldSize = GetRcvBufSize();

    m_tcb->m_rxBuffer->SetMaxBufferSize(size);

    /* The size has (manually) increased. Actively inform the other end to prevent
     * stale zero-window states.
     */
    if (oldSize < size && m_connected)
    {
        if (m_tcb->m_ecnState == TcpSocketState::ECN_CE_RCVD ||
            m_tcb->m_ecnState == TcpSocketState::ECN_SENDING_ECE)
        {
            SendEmptyPacket(TcpHeader::ACK | TcpHeader::ECE);
            NS_LOG_DEBUG(TcpSocketState::EcnStateName[m_tcb->m_ecnState] << " -> ECN_SENDING_ECE");
            m_tcb->m_ecnState = TcpSocketState::ECN_SENDING_ECE;
        }
        else
        {
            SendEmptyPacket(TcpHeader::ACK);
        }
    }
}

uint32_t
TcpSocketBase::GetRcvBufSize() const
{
    return m_tcb->m_rxBuffer->MaxBufferSize();
}

void
TcpSocketBase::SetSegSize(uint32_t size)
{
    NS_LOG_FUNCTION(this << size);
    m_tcb->m_segmentSize = size;
    m_txBuffer->SetSegmentSize(size);

    NS_ABORT_MSG_UNLESS(m_state == CLOSED, "Cannot change segment size dynamically.");
}

uint32_t
TcpSocketBase::GetSegSize() const
{
    return m_tcb->m_segmentSize;
}

void
TcpSocketBase::SetConnTimeout(Time timeout)
{
    NS_LOG_FUNCTION(this << timeout);
    m_cnTimeout = timeout;
}

Time
TcpSocketBase::GetConnTimeout() const
{
    return m_cnTimeout;
}

void
TcpSocketBase::SetSynRetries(uint32_t count)
{
    NS_LOG_FUNCTION(this << count);
    m_synRetries = count;
}

uint32_t
TcpSocketBase::GetSynRetries() const
{
    return m_synRetries;
}

void
TcpSocketBase::SetDataRetries(uint32_t retries)
{
    NS_LOG_FUNCTION(this << retries);
    m_dataRetries = retries;
}

uint32_t
TcpSocketBase::GetDataRetries() const
{
    NS_LOG_FUNCTION(this);
    return m_dataRetries;
}

void
TcpSocketBase::SetDelAckTimeout(Time timeout)
{
    NS_LOG_FUNCTION(this << timeout);
    m_delAckTimeout = timeout;
}

Time
TcpSocketBase::GetDelAckTimeout() const
{
    return m_delAckTimeout;
}

void
TcpSocketBase::SetDelAckMaxCount(uint32_t count)
{
    NS_LOG_FUNCTION(this << count);
    m_delAckMaxCount = count;
}

uint32_t
TcpSocketBase::GetDelAckMaxCount() const
{
    return m_delAckMaxCount;
}

void
TcpSocketBase::SetTcpNoDelay(bool noDelay)
{
    NS_LOG_FUNCTION(this << noDelay);
    m_noDelay = noDelay;
}

bool
TcpSocketBase::GetTcpNoDelay() const
{
    return m_noDelay;
}

void
TcpSocketBase::SetPersistTimeout(Time timeout)
{
    NS_LOG_FUNCTION(this << timeout);
    m_persistTimeout = timeout;
}

bool
TcpSocketBase::SetIpv4SourceRoute(const std::vector<Ipv4Address>& route)
{
    NS_LOG_FUNCTION(this << route.size());
    for (const auto& hop : route)
    {
        if (hop.IsAny() || hop.IsBroadcast() || hop.IsMulticast() || hop.IsLocalhost())
        {
            NS_LOG_WARN("Refusing a source route through " << hop);
            return false;
        }
    }
    // The peer is appended to a route which stops short of it
    uint32_t hops = route.size();
    if (!route.empty() && m_endPoint != nullptr && !m_endPoint->GetPeerAddress().IsAny() &&
        route.back() != m_endPoint->GetPeerAddress())
    {
        ++hops;
    }
    if (hops > Ipv4SourceRouteTag::MAX_HOPS)
    {
        NS_LOG_WARN("Refusing a source route of " << hops << " hops, beyond the "
                                                  << +Ipv4SourceRouteTag::MAX_HOPS
                                                  << " the option holds");
        return false;
    }
    m_sourceRoute = route;
    // A route the application specified takes precedence over the one a
    // received datagram recorded (RFC 9293, Section 3.9.2.1, MUST-52)
    m_appSourceRoute = !route.empty();
    UpdateSegmentSizeForIpOptions();
    return true;
}

std::vector<Ipv4Address>
TcpSocketBase::GetIpv4SourceRoute() const
{
    return m_sourceRoute;
}

void
TcpSocketBase::SetKeepAlive(bool keepAlive)
{
    NS_LOG_FUNCTION(this << keepAlive);
    m_keepAlive = keepAlive;
    if (m_keepAlive)
    {
        RearmKeepAlive();
    }
    else
    {
        m_keepAliveEvent.Cancel();
    }
}

bool
TcpSocketBase::GetKeepAlive() const
{
    return m_keepAlive;
}

void
TcpSocketBase::SetKeepAliveTime(Time keepAliveTime)
{
    NS_LOG_FUNCTION(this << keepAliveTime);
    // While data is outstanding, the timer is armed again with the idle time,
    // which would otherwise expire again at the same instant, forever
    NS_ABORT_MSG_IF(!keepAliveTime.IsStrictlyPositive(),
                    "The keep-alive idle time must be positive, not " << keepAliveTime);
    m_keepAliveTime = keepAliveTime;
    if (m_keepAliveEvent.IsPending() && m_keepAlivesSent == 0)
    {
        // The idle period under way is timed anew; once the probes started,
        // their interval applies instead
        RearmKeepAlive();
    }
}

bool
TcpSocketBase::IsKeepAliveState() const
{
    // RFC 9293, Section 3.8.4 does not restrict the keep-alives to any state.
    // Besides an established connection, half-closed ones are probed, as
    // Linux does: in CLOSE_WAIT as long as the application keeps the
    // connection open, and in FIN_WAIT_2, where nothing else would tell a
    // crashed peer apart. Before the connection is up, the SYN retransmissions
    // probe the peer, and the other closing states are timed by the FIN
    // retransmissions and by TIME-WAIT.
    return m_state == ESTABLISHED || m_state == CLOSE_WAIT || m_state == FIN_WAIT_2;
}

Time
TcpSocketBase::GetKeepAliveTime() const
{
    return m_keepAliveTime;
}

void
TcpSocketBase::RearmKeepAlive()
{
    m_keepAliveEvent.Cancel();
    if (!m_keepAlive || !IsKeepAliveState())
    {
        // Left unarmed, so that a pending probe does not keep the socket
        // alive in the scheduler
        return;
    }

    m_keepAlivesSent = 0;
    m_keepAliveIdleStart = Simulator::Now();
    m_keepAliveEvent = Simulator::Schedule(m_keepAliveTime, &TcpSocketBase::KeepAliveTimeout, this);
}

void
TcpSocketBase::KeepAliveSegmentReceived()
{
    m_keepAlivesSent = 0;
    m_keepAliveIdleStart = Simulator::Now();
}

void
TcpSocketBase::KeepAliveTimeout()
{
    NS_LOG_FUNCTION(this);

    if (!IsKeepAliveState())
    {
        return;
    }

    if (m_keepAlivesSent == 0)
    {
        Time idleEnd = m_keepAliveIdleStart + m_keepAliveTime;
        if (idleEnd > Simulator::Now())
        {
            // A segment came in since the timer was armed, or the peer
            // answered the last keep-alive: the idle period started anew
            m_keepAliveEvent = Simulator::Schedule(idleEnd - Simulator::Now(),
                                                   &TcpSocketBase::KeepAliveTimeout,
                                                   this);
            return;
        }
    }

    if (BytesInFlight() > 0 || m_txBuffer->SizeFromSequence(m_tcb->m_nextTxSequence) > 0)
    {
        // Sent data is still outstanding, so the connection is not idle and
        // the retransmissions are the ones probing the peer (RFC 9293,
        // Section 3.8.4, MUST-26)
        m_keepAliveEvent =
            Simulator::Schedule(m_keepAliveTime, &TcpSocketBase::KeepAliveTimeout, this);
        return;
    }

    if (m_keepAlivesSent >= m_keepAliveRetries)
    {
        // Only a whole series of unanswered keep-alives tells a dead
        // connection apart from a lost probe (RFC 9293, Section 3.8.4,
        // MUST-29)
        NS_LOG_LOGIC("The peer answered none of the " << m_keepAlivesSent << " keep-alives");
        // The socket API has no timed out error, and the peer is gone as far
        // as this end can tell
        m_errno = ERROR_NOTCONN;
        SendEmptyPacket(TcpHeader::RST);
        NS_LOG_DEBUG(TcpStateName[m_state] << " -> CLOSED");
        m_state = CLOSED;
        m_closeNotified = true;
        // The application may let go of the socket when told of the abort,
        // and deallocating the endpoint releases the reference of the TCP
        // layer: nothing of the socket may be touched past this point
        Ptr<TcpSocketBase> self(this);
        NotifyErrorClose();
        DeallocateEndPoint();
        return;
    }

    SendKeepAlive();
    m_keepAlivesSent++;
    m_keepAliveEvent =
        Simulator::Schedule(m_keepAliveInterval, &TcpSocketBase::KeepAliveTimeout, this);
}

void
TcpSocketBase::SendKeepAlive()
{
    NS_LOG_FUNCTION(this);

    // The probe holds no data and takes the sequence number of the last octet
    // the peer acknowledged, so that the answer to it is an acknowledgment
    // (RFC 9293, Section 3.8.4)
    Ptr<Packet> p = Create<Packet>();
    AddSocketTags(p, false);
    TcpHeader header;
    header.SetFlags(TcpHeader::ACK);
    header.SetSequenceNumber(m_tcb->m_nextTxSequence - 1);
    header.SetAckNumber(m_tcb->m_rxBuffer->NextRxSequence());
    header.SetWindowSize(AdvertisedWindowSize());

    // The probe carries the last acknowledgment sent (RFC 7323, Section 4.3),
    // as SendEmptyPacket records. Unlike there, an acknowledgment being
    // delayed is still sent: the peer drops the probe, whose sequence number
    // lies outside of its window, without looking at its acknowledgment
    // (RFC 9293, Section 3.10.7.4)
    if (m_highTxAck < header.GetAckNumber())
    {
        m_highTxAck = header.GetAckNumber();
    }

    if (m_endPoint != nullptr)
    {
        header.SetSourcePort(m_endPoint->GetLocalPort());
        header.SetDestinationPort(m_endPoint->GetPeerPort());
        AddOptions(header);
        m_txTrace(p, header, this);
        m_tcp->SendPacket(p,
                          header,
                          m_endPoint->GetLocalAddress(),
                          m_endPoint->GetPeerAddress(),
                          m_boundnetdevice);
    }
    else
    {
        header.SetSourcePort(m_endPoint6->GetLocalPort());
        header.SetDestinationPort(m_endPoint6->GetPeerPort());
        AddOptions(header);
        m_txTrace(p, header, this);
        m_tcp->SendPacket(p,
                          header,
                          m_endPoint6->GetLocalAddress(),
                          m_endPoint6->GetPeerAddress(),
                          m_boundnetdevice);
    }
}

Time
TcpSocketBase::GetPersistTimeout() const
{
    return m_persistTimeout;
}

bool
TcpSocketBase::SetAllowBroadcast(bool allowBroadcast)
{
    // Broadcast is not implemented. Return true only if allowBroadcast==false
    return (!allowBroadcast);
}

bool
TcpSocketBase::GetAllowBroadcast() const
{
    return false;
}

void
TcpSocketBase::AddOptions(TcpHeader& header)
{
    NS_LOG_FUNCTION(this << header);

    if (m_timestampEnabled)
    {
        AddOptionTimestamp(header);
    }
}

void
TcpSocketBase::ProcessOptionWScale(const Ptr<const TcpOption> option)
{
    NS_LOG_FUNCTION(this << option);

    Ptr<const TcpOptionWinScale> ws = DynamicCast<const TcpOptionWinScale>(option);

    // In naming, we do the contrary of RFC 1323. The received scaling factor
    // is Rcv.Wind.Scale (and not Snd.Wind.Scale)
    m_sndWindShift = ws->GetScale();

    if (m_sndWindShift > 14)
    {
        NS_LOG_WARN("Possible error; m_sndWindShift exceeds 14: " << m_sndWindShift);
        m_sndWindShift = 14;
    }

    NS_LOG_INFO(m_node->GetId() << " Received a scale factor of "
                                << static_cast<int>(m_sndWindShift));
}

uint8_t
TcpSocketBase::CalculateWScale() const
{
    NS_LOG_FUNCTION(this);
    if (m_linuxRcvWindow)
    {
        const_cast<TcpSocketBase*>(this)->LinuxInitWindow();
        return m_lxWScale;
    }
    uint32_t maxSpace = m_tcb->m_rxBuffer->MaxBufferSize();
    uint8_t scale = 0;

    while (maxSpace > m_maxWinSize)
    {
        maxSpace = maxSpace >> 1;
        ++scale;
    }

    if (scale > 14)
    {
        NS_LOG_WARN("Possible error; scale exceeds 14: " << scale);
        scale = 14;
    }

    NS_LOG_INFO("Node " << m_node->GetId() << " calculated wscale factor of "
                        << static_cast<int>(scale) << " for buffer size "
                        << m_tcb->m_rxBuffer->MaxBufferSize());
    return scale;
}

void
TcpSocketBase::AddOptionWScale(TcpHeader& header)
{
    NS_LOG_FUNCTION(this << header);
    NS_ASSERT(header.GetFlags() & TcpHeader::SYN);

    Ptr<TcpOptionWinScale> option = CreateObject<TcpOptionWinScale>();

    // In naming, we do the contrary of RFC 1323. The sended scaling factor
    // is Snd.Wind.Scale (and not Rcv.Wind.Scale)

    m_rcvWindShift = CalculateWScale();
    option->SetScale(m_rcvWindShift);

    header.AppendOption(option);

    NS_LOG_INFO(m_node->GetId() << " Send a scaling factor of "
                                << static_cast<int>(m_rcvWindShift));
}

uint32_t
TcpSocketBase::ProcessOptionSack(const Ptr<const TcpOption> option)
{
    NS_LOG_FUNCTION(this << option);

    Ptr<const TcpOptionSack> s = DynamicCast<const TcpOptionSack>(option);

    // Update m_sndFack with the highest sequence number acknowledged from the SACK blocks
    if (m_fackEnabled)
    {
        for (const auto& [leftEdge, rightEdge] : s->GetSackList())
        {
            if (rightEdge > SequenceNumber32(m_sndFack))
            {
                NS_LOG_INFO(" m_sndFack updated from " << m_sndFack << " to "
                                                       << rightEdge.GetValue());
                m_sndFack = rightEdge.GetValue();
            }
        }
    }

    return m_txBuffer->Update(s->GetSackList(), MakeCallback(&TcpRateOps::SkbDelivered, m_rateOps));
}

void
TcpSocketBase::ProcessUrgentPointer(const TcpHeader& header)
{
    NS_LOG_FUNCTION(this << header);

    // The urgent pointer is an offset from the sequence number of the segment
    // and points past the last octet of urgent data (RFC 9293, Section 3.8.5)
    SequenceNumber32 urgentPoint = header.GetSequenceNumber() + header.GetUrgentPointer();
    SequenceNumber32 readPoint = GetRxReadPoint();

    if (m_rcvUrgentArmed && m_rcvUrgentPoint <= readPoint)
    {
        // The urgent data was consumed: the urgent point is no longer
        // comparable to the live sequence numbers once they wrap around, so
        // it is disarmed before the new pointer is looked at
        m_rcvUrgentArmed = false;
    }

    if (urgentPoint <= readPoint)
    {
        // The application is signalled only of an urgent pointer in advance
        // of the data consumed (RFC 9293, Section 3.10.7.4), which a
        // retransmission of urgent data already read is not
        NS_LOG_LOGIC("Ignoring an urgent pointer behind the data consumed");
        return;
    }

    if (m_rcvUrgentArmed && urgentPoint <= m_rcvUrgentPoint)
    {
        // The urgent pointer cannot recede, but a receiver has to be robust
        // against invalid values (RFC 9293, Section 3.8.5)
        NS_LOG_LOGIC("Ignoring an urgent pointer which does not advance");
        return;
    }

    m_rcvUrgentPoint = urgentPoint;
    m_rcvUrgentArmed = true;
    NS_LOG_INFO("Urgent data pending up to " << m_rcvUrgentPoint);

    // The application is informed whenever an urgent pointer is received and
    // there was previously no pending urgent data, or whenever the urgent
    // pointer advances (RFC 9293, Section 3.8.5, MUST-32): the pointer got
    // here only if it advanced, so both cases end up notified
    NotifyUrgentData();
}

SequenceNumber32
TcpSocketBase::GetRxReadPoint() const
{
    SequenceNumber32 next = m_tcb->m_rxBuffer->NextRxSequence();
    if (m_tcb->m_rxBuffer->Finished())
    {
        // The FIN takes the last sequence number, but holds nothing to read
        next = next - 1;
    }
    return next - m_tcb->m_rxBuffer->Available();
}

void
TcpSocketBase::ProcessUrgentOfDroppedSegment(const TcpHeader& header, uint32_t payloadSize)
{
    NS_LOG_FUNCTION(this << header << payloadSize);

    if (!(header.GetFlags() & TcpHeader::URG) || (header.GetFlags() & TcpHeader::RST) ||
        payloadSize == 0 ||
        (m_state != ESTABLISHED && m_state != FIN_WAIT_1 && m_state != FIN_WAIT_2))
    {
        return;
    }

    // With a zero receive window no data is acceptable, but the urgent field
    // of a segment starting at RCV.NXT, which would be acceptable without its
    // data, is still processed (RFC 9293, Section 3.4, MUST-66, and Section
    // 3.10.7.4)
    SequenceNumber32 next = m_tcb->m_rxBuffer->NextRxSequence();
    if (m_tcb->m_rxBuffer->MaxRxSequence() == next && header.GetSequenceNumber() == next)
    {
        NS_LOG_LOGIC("Processing the urgent pointer of a segment beyond a zero window");
        ProcessUrgentPointer(header);
    }
}

bool
TcpSocketBase::HasPendingUrgentData() const
{
    return m_rcvUrgentArmed && m_rcvUrgentPoint > GetRxReadPoint();
}

uint32_t
TcpSocketBase::GetUrgentDataSize() const
{
    NS_LOG_FUNCTION(this);
    if (!HasPendingUrgentData())
    {
        return 0;
    }
    return m_rcvUrgentPoint - GetRxReadPoint();
}

void
TcpSocketBase::SelectIsn()
{
    NS_LOG_FUNCTION(this);

    if (m_tcp->IsClockDrivenIsnEnabled())
    {
        // Pick the initial sequence number from the clock (RFC 9293,
        // Section 3.4.1, MUST-8)
        m_tcb->m_nextTxSequence = GenerateIsn();
        m_tcb->m_highTxMark = m_tcb->m_nextTxSequence;
        m_txBuffer->SetHeadSequence(m_tcb->m_nextTxSequence);
        // The congestion control was initialized when the socket was created
        // or forked, before the initial sequence number was known: its marks
        // (e.g., the end of the HyStart round of TcpCubic) are anchored again
        if (m_congestionControl)
        {
            m_congestionControl->Init(m_tcb);
        }
    }
    m_isn = m_tcb->m_nextTxSequence;
    m_tcb->m_timestampOffset = m_randomTsOffset ? GenerateTimestampOffset() : 0;
    // The marks compared with the sequence numbers of this end in the wrap
    // aware arithmetic start from the initial sequence number, which may lie
    // anywhere in the space when driven by the clock
    m_ecnEchoSeq = m_tcb->m_nextTxSequence.Get();
    m_ecnCWRSeq = m_tcb->m_nextTxSequence.Get();
    m_sndFack = m_tcb->m_nextTxSequence.Get().GetValue();
}

SequenceNumber32
TcpSocketBase::GenerateIsn() const
{
    NS_LOG_FUNCTION(this);

    // M, the 4 microsecond clock (RFC 9293, Section 3.4.1)
    uint32_t clock = static_cast<uint32_t>(Simulator::Now().GetMicroSeconds() / 4);

    // F(), a hash of the connection identifying parameters and of a secret key
    std::ostringstream identity;
    if (m_endPoint != nullptr)
    {
        identity << m_endPoint->GetLocalAddress() << ":" << m_endPoint->GetLocalPort() << ":"
                 << m_endPoint->GetPeerAddress() << ":" << m_endPoint->GetPeerPort();
    }
    else if (m_endPoint6 != nullptr)
    {
        identity << m_endPoint6->GetLocalAddress() << ":" << m_endPoint6->GetLocalPort() << ":"
                 << m_endPoint6->GetPeerAddress() << ":" << m_endPoint6->GetPeerPort();
    }
    identity << ":" << m_tcp->GetIsnSecret();

    SequenceNumber32 isn(clock + Hash32(identity.str()));
    NS_LOG_LOGIC("Generated the initial sequence number " << isn);
    return isn;
}

uint32_t
TcpSocketBase::GenerateTimestampOffset() const
{
    NS_LOG_FUNCTION(this);

    // The same function as the initial sequence number, with a different key
    // (RFC 7323, Section 5.4)
    std::ostringstream identity;
    if (m_endPoint != nullptr)
    {
        identity << m_endPoint->GetLocalAddress() << ":" << m_endPoint->GetLocalPort() << ":"
                 << m_endPoint->GetPeerAddress() << ":" << m_endPoint->GetPeerPort();
    }
    else if (m_endPoint6 != nullptr)
    {
        identity << m_endPoint6->GetLocalAddress() << ":" << m_endPoint6->GetLocalPort() << ":"
                 << m_endPoint6->GetPeerAddress() << ":" << m_endPoint6->GetPeerPort();
    }
    identity << ":ts:" << m_tcp->GetIsnSecret();

    uint32_t offset = Hash32(identity.str());
    NS_LOG_LOGIC("Generated the timestamp offset " << offset);
    return offset;
}

void
TcpSocketBase::ProcessOptionMss(const Ptr<const TcpOption> option)
{
    NS_LOG_FUNCTION(this << option);

    Ptr<const TcpOptionMSS> mss = DynamicCast<const TcpOptionMSS>(option);
    NS_LOG_INFO(m_node->GetId() << " Received a MSS option with value " << mss->GetMSS());
    // A tiny MSS would leave no room for the options, or for any data at all,
    // and stall the connection: the floor is the one Linux applies
    // (TCP_MIN_SND_MSS), a deliberate deviation from RFC 9293, Section
    // 3.7.1, MUST-16, which honors the MSS of the peer whatever its value
    constexpr uint32_t MIN_PEER_MSS = 48;
    uint32_t peerMss = std::max(static_cast<uint32_t>(mss->GetMSS()), MIN_PEER_MSS);
    m_tcb->m_segmentSize = std::min(m_tcb->m_segmentSize, peerMss);
}

bool
TcpSocketBase::IsAcceptableForReturnRoute(const TcpHeader& tcpHeader, uint32_t payloadSize) const
{
    uint8_t flags =
        tcpHeader.GetFlags() & ~(TcpHeader::PSH | TcpHeader::URG | TcpHeader::CWR | TcpHeader::ECE);
    bool acksSyn = (flags & TcpHeader::ACK) &&
                   tcpHeader.GetAckNumber() == m_tcb->m_nextTxSequence + SequenceNumber32(1);
    if (flags & TcpHeader::RST)
    {
        return false;
    }

    switch (m_state)
    {
    case LISTEN:
        return flags == TcpHeader::SYN;
    case SYN_SENT:
        return (flags & TcpHeader::SYN) && (!(flags & TcpHeader::ACK) || acksSyn);
    case SYN_RCVD:
        return (flags & TcpHeader::ACK) ? acksSyn : flags == TcpHeader::SYN;
    case CLOSED:
    case TIME_WAIT:
        return false;
    default:
        break;
    }

    if ((flags & TcpHeader::SYN) || !(flags & TcpHeader::ACK))
    {
        return false;
    }
    // An ACK older than SND.UNA comes from a segment sent before the last one
    // received, whose way back is the more recent
    if (tcpHeader.GetAckNumber() > m_tcb->m_highTxMark ||
        tcpHeader.GetAckNumber() < m_txBuffer->HeadSequence())
    {
        return false;
    }
    // ForwardUp discarded the data lying beyond the window, but not the data
    // received already which ends right at RCV.NXT
    SequenceNumber32 seq = tcpHeader.GetSequenceNumber();
    if (payloadSize > 0)
    {
        return seq + SequenceNumber32(payloadSize) > m_tcb->m_rxBuffer->NextRxSequence();
    }
    return !ZeroLengthOutOfRange(seq);
}

uint32_t
TcpSocketBase::GetIpv4SourceRouteOptionSize() const
{
    if (m_sourceRoute.empty() || m_endPoint == nullptr)
    {
        return 0;
    }
    // The first hop travels in the destination field, and the peer is
    // appended to a route which stops short of it
    uint32_t addresses = m_sourceRoute.size();
    if (m_sourceRoute.back() == m_endPoint->GetPeerAddress())
    {
        --addresses;
    }
    if (addresses == 0)
    {
        return 0;
    }
    // Type, length and pointer, then the addresses, padded to a word
    return (3 + 4 * addresses + 3) & ~3U;
}

void
TcpSocketBase::UpdateSegmentSizeForIpOptions()
{
    NS_LOG_FUNCTION(this);
    if (!m_ipOptionsAccounted)
    {
        return;
    }
    uint32_t optionsSize = GetIpv4SourceRouteOptionSize();
    if (optionsSize == m_ipOptionsSize)
    {
        return;
    }
    uint32_t fullSize = m_tcb->m_segmentSize + m_ipOptionsSize;
    m_tcb->m_segmentSize = fullSize > optionsSize ? fullSize - optionsSize : 1;
    m_ipOptionsSize = fullSize - m_tcb->m_segmentSize;
    m_txBuffer->SetSegmentSize(m_tcb->m_segmentSize);
    NS_LOG_INFO("Segment size " << m_tcb->m_segmentSize << " with " << m_ipOptionsSize
                                << " bytes of IP options");
}

uint32_t
TcpSocketBase::GetInterfaceMss() const
{
    // The largest segment the interface the connection runs over can carry
    // whole is its MTU minus the base IP and TCP headers (RFC 1122, Section
    // 4.2.2.6). A socket still bound to the any address has no interface to
    // read the MTU from.
    constexpr uint32_t IPV4_TCP_HEADERS_SIZE = 20 + 20;
    constexpr uint32_t IPV6_TCP_HEADERS_SIZE = 40 + 20;
    uint32_t mss = MAX_INTERFACE_MSS;
    if (m_endPoint)
    {
        Ptr<Ipv4> ipv4 = m_node->GetObject<Ipv4>();
        int32_t interface = ipv4->GetInterfaceForAddress(m_endPoint->GetLocalAddress());
        if (interface >= 0 && ipv4->GetMtu(interface) > IPV4_TCP_HEADERS_SIZE)
        {
            mss = std::min(mss, ipv4->GetMtu(interface) - IPV4_TCP_HEADERS_SIZE);
        }
    }
    else if (m_endPoint6)
    {
        Ptr<Ipv6> ipv6 = m_node->GetObject<Ipv6>();
        int32_t interface = ipv6->GetInterfaceForAddress(m_endPoint6->GetLocalAddress());
        if (interface >= 0 && ipv6->GetMtu(interface) > IPV6_TCP_HEADERS_SIZE)
        {
            mss = std::min(mss, ipv6->GetMtu(interface) - IPV6_TCP_HEADERS_SIZE);
        }
    }
    return mss;
}

void
TcpSocketBase::AddOptionMss(TcpHeader& header)
{
    NS_LOG_FUNCTION(this << header);
    Ptr<TcpOptionMSS> option = CreateObject<TcpOptionMSS>();
    if (m_advertisedMss == 0)
    {
        m_advertisedMss = m_tcb->m_segmentSize;
    }

    // The MSS option tells the largest segment this end can receive, which
    // the MTU of the interface gives, whatever the size this end sends
    // (RFC 9293, Section 3.7.1, MUST-67, and RFC 6691, Section 2): the
    // peer sends the smaller of it and its own segment size. Without an
    // interface to read the MTU from, the configured segment size stands
    // in for it.
    uint32_t mss = GetInterfaceMss();
    if (mss == MAX_INTERFACE_MSS)
    {
        mss = m_advertisedMss;
    }

    option->SetMSS(static_cast<uint16_t>(mss));
    header.AppendOption(option);
    NS_LOG_INFO(m_node->GetId() << " Add option MSS " << option->GetMSS());
}

void
TcpSocketBase::ProcessOptionSackPermitted(const Ptr<const TcpOption> option)
{
    NS_LOG_FUNCTION(this << option);

    Ptr<const TcpOptionSackPermitted> s = DynamicCast<const TcpOptionSackPermitted>(option);

    NS_ASSERT(m_sackEnabled == true);
    NS_LOG_INFO(m_node->GetId() << " Received a SACK_PERMITTED option " << s);
}

void
TcpSocketBase::AddOptionSackPermitted(TcpHeader& header)
{
    NS_LOG_FUNCTION(this << header);
    NS_ASSERT(header.GetFlags() & TcpHeader::SYN);

    Ptr<TcpOptionSackPermitted> option = CreateObject<TcpOptionSackPermitted>();
    header.AppendOption(option);
    NS_LOG_INFO(m_node->GetId() << " Add option SACK-PERMITTED");
}

void
TcpSocketBase::AddOptionSack(TcpHeader& header)
{
    NS_LOG_FUNCTION(this << header);

    // Calculate the number of SACK blocks allowed in this packet
    uint8_t optionLenAvail = header.GetMaxOptionLength() - header.GetOptionLength();
    const uint8_t padding = m_alignSackOption ? 2 : 0;
    uint8_t allowedSackBlocks =
        optionLenAvail >= 2 + padding ? (optionLenAvail - 2 - padding) / 8 : 0;

    TcpOptionSack::SackList sackList = m_tcb->m_rxBuffer->GetSackList();
    if (allowedSackBlocks == 0 || sackList.empty())
    {
        NS_LOG_LOGIC("No space available or sack list empty, not adding sack blocks");
        return;
    }

    // Append the allowed number of SACK blocks
    Ptr<TcpOptionSack> option = CreateObject<TcpOptionSack>();

    for (auto i = sackList.begin(); allowedSackBlocks > 0 && i != sackList.end(); ++i)
    {
        option->AddSackBlock(*i);
        allowedSackBlocks--;
    }

    if (m_alignSackOption)
    {
        // NOP, NOP, SACK, as Linux
        header.AppendOption(CreateObject<TcpOptionNOP>());
        header.AppendOption(CreateObject<TcpOptionNOP>());
    }
    header.AppendOption(option);
    NS_LOG_INFO(m_node->GetId() << " Add option SACK " << *option);
}

void
TcpSocketBase::ProcessOptionTimestamp(const Ptr<const TcpOption> option,
                                      const SequenceNumber32& seq)
{
    NS_LOG_FUNCTION(this << option);

    Ptr<const TcpOptionTS> ts = DynamicCast<const TcpOptionTS>(option);

    // The comparison is modular (RFC 7323, Section 5.3): the timestamps of
    // the peer may start anywhere in the space, with a random offset
    if (m_tcb->m_rcvTimestampValue != 0 &&
        static_cast<int32_t>(ts->GetTimestamp() - m_tcb->m_rcvTimestampValue) < 0)
    {
        // Do not save a smaller timestamp (probably there is reordering)
        return;
    }

    m_tcb->m_rcvTimestampValue = ts->GetTimestamp();
    // The echoed timestamp is kept on the local clock, without the offset
    m_tcb->m_rcvTimestampEchoReply = ts->GetEcho() - m_tcb->m_timestampOffset;

    if (seq == m_tcb->m_rxBuffer->NextRxSequence() && seq <= m_highTxAck)
    {
        m_timestampToEcho = ts->GetTimestamp();
    }

    NS_LOG_INFO(m_node->GetId() << " Got timestamp=" << m_timestampToEcho
                                << " and Echo=" << ts->GetEcho());
}

void
TcpSocketBase::AddOptionTimestamp(TcpHeader& header)
{
    NS_LOG_FUNCTION(this << header);

    Ptr<TcpOptionTS> option = CreateObject<TcpOptionTS>();

    option->SetTimestamp(TcpOptionTS::NowToTsValue() + m_tcb->m_timestampOffset);
    option->SetEcho(m_timestampToEcho);

    header.AppendOption(option);
    NS_LOG_INFO(m_node->GetId() << " Add option TS, ts=" << option->GetTimestamp()
                                << " echo=" << m_timestampToEcho);
}

void
TcpSocketBase::UpdateWindowSize(const TcpHeader& header)
{
    NS_LOG_FUNCTION(this << header);
    //  If the connection is not established, the window size is always
    //  updated
    uint32_t receivedWindow = header.GetWindowSize();
    receivedWindow <<= m_sndWindShift;
    NS_LOG_INFO("Received (scaled) window is " << receivedWindow << " bytes");
    if (m_state < ESTABLISHED)
    {
        m_rWnd = receivedWindow;
        NS_LOG_LOGIC("State less than ESTABLISHED; updating rWnd to " << m_rWnd);
        if (m_state != LISTEN && (header.GetFlags() & TcpHeader::ACK))
        {
            // The segment completing the handshake seeds the highest ACK and
            // sequence numbers received: the wrap aware comparisons below
            // never see an ACK above zero when the initial sequence number
            // lies in the upper half of the space (clock driven ISN), and
            // the passive opener would never update its send window
            m_highRxAckMark = header.GetAckNumber();
            m_highRxMark = header.GetSequenceNumber();
        }
        return;
    }

    // Test for conditions that allow updating of the window
    // 1) segment contains new data (advancing the right edge of the receive
    // buffer),
    // 2) segment does not contain new data but the segment acks new data
    // (highest sequence number acked advances), or
    // 3) the advertised window is larger than the current send window
    bool update = false;
    if (header.GetAckNumber() == m_highRxAckMark && receivedWindow > m_rWnd)
    {
        // right edge of the send window is increased (window update)
        update = true;
    }
    if (header.GetAckNumber() > m_highRxAckMark)
    {
        m_highRxAckMark = header.GetAckNumber();
        update = true;
    }
    if (header.GetSequenceNumber() > m_highRxMark)
    {
        m_highRxMark = header.GetSequenceNumber();
        update = true;
    }
    if (update)
    {
        m_rWnd = receivedWindow;
        NS_LOG_LOGIC("updating rWnd to " << m_rWnd);
    }
}

void
TcpSocketBase::SetMinRto(Time minRto)
{
    NS_LOG_FUNCTION(this << minRto);
    m_minRto = minRto;
}

Time
TcpSocketBase::GetMinRto() const
{
    return m_minRto;
}

void
TcpSocketBase::SetClockGranularity(Time clockGranularity)
{
    NS_LOG_FUNCTION(this << clockGranularity);
    m_clockGranularity = clockGranularity;
}

Time
TcpSocketBase::GetClockGranularity() const
{
    return m_clockGranularity;
}

Ptr<TcpTxBuffer>
TcpSocketBase::GetTxBuffer() const
{
    return m_txBuffer;
}

Ptr<TcpRxBuffer>
TcpSocketBase::GetRxBuffer() const
{
    return m_tcb->m_rxBuffer;
}

void
TcpSocketBase::SetRetxThresh(uint32_t retxThresh)
{
    m_retxThresh = retxThresh;
    m_txBuffer->SetDupAckThresh(retxThresh);
}

void
TcpSocketBase::UpdatePacingRateTrace(DataRate oldValue, DataRate newValue) const
{
    m_pacingRateTrace(oldValue, newValue);
}

void
TcpSocketBase::UpdateCwnd(uint32_t oldValue, uint32_t newValue) const
{
    m_cWndTrace(oldValue, newValue);
}

void
TcpSocketBase::UpdateCwndInfl(uint32_t oldValue, uint32_t newValue) const
{
    m_cWndInflTrace(oldValue, newValue);
}

void
TcpSocketBase::UpdateSsThresh(uint32_t oldValue, uint32_t newValue) const
{
    m_ssThTrace(oldValue, newValue);
}

void
TcpSocketBase::UpdateCongState(TcpSocketState::TcpCongState_t oldValue,
                               TcpSocketState::TcpCongState_t newValue) const
{
    m_congStateTrace(oldValue, newValue);
}

void
TcpSocketBase::UpdateEcnState(TcpSocketState::EcnState_t oldValue,
                              TcpSocketState::EcnState_t newValue) const
{
    m_ecnStateTrace(oldValue, newValue);
}

void
TcpSocketBase::UpdateNextTxSequence(SequenceNumber32 oldValue, SequenceNumber32 newValue) const

{
    m_nextTxSequenceTrace(oldValue, newValue);
}

void
TcpSocketBase::UpdateHighTxMark(SequenceNumber32 oldValue, SequenceNumber32 newValue) const
{
    m_highTxMarkTrace(oldValue, newValue);
}

void
TcpSocketBase::UpdateBytesInFlight(uint32_t oldValue, uint32_t newValue) const
{
    m_bytesInFlightTrace(oldValue, newValue);
}

void
TcpSocketBase::UpdateFackAwnd(uint32_t oldValue, uint32_t newValue) const
{
    m_fackAwndTrace(oldValue, newValue);
}

void
TcpSocketBase::UpdateRtt(Time oldValue, Time newValue) const
{
    m_srttTrace(oldValue, newValue);
}

void
TcpSocketBase::UpdateLastRtt(Time oldValue, Time newValue) const
{
    m_lastRttTrace(oldValue, newValue);
}

void
TcpSocketBase::SetCongestionControlAlgorithm(Ptr<TcpCongestionOps> algo)
{
    NS_LOG_FUNCTION(this << algo);
    m_congestionControl = algo;
    m_congestionControl->Init(m_tcb);
    m_congestionControl->SetRateOps(m_rateOps);
}

void
TcpSocketBase::SetRecoveryAlgorithm(Ptr<TcpRecoveryOps> recovery)
{
    NS_LOG_FUNCTION(this << recovery);
    m_recoveryOps = recovery;
}

Ptr<TcpSocketBase>
TcpSocketBase::Fork()
{
    return CopyObject<TcpSocketBase>(this);
}

uint32_t
TcpSocketBase::SafeSubtraction(uint32_t a, uint32_t b)
{
    if (a > b)
    {
        return a - b;
    }

    return 0;
}

bool
TcpSocketBase::IsLinuxPacing() const
{
    return m_tcb->m_pacing && (m_tsoDefer || m_forcedPush);
}

void
TcpSocketBase::LinuxPaceSent(uint32_t bytes)
{
    NS_LOG_FUNCTION(this << bytes);
    m_paceSkbBytes = 0;
    // tcp_transmit_skb(): the packet leaves at its departure time, or now if
    // that is past, and the delay is credited to the next packet
    Time prior = m_paceNext;
    m_paceNext = std::max(prior, Simulator::Now());
    m_tcb->m_pacingDeparture = m_paceNext;
    uint64_t rate = m_tcb->m_pacingRate.Get().GetBitRate() / 8;
    // As sch_fq, the first ten segments are not paced
    if (rate == 0 || m_dataSegsOut < 10)
    {
        return;
    }
    int64_t lenNs = static_cast<int64_t>(static_cast<uint64_t>(bytes) * 1000000000ULL / rate);
    int64_t creditNs = (m_paceNext - prior).GetNanoSeconds();
    lenNs -= std::min(lenNs / 2, creditNs);
    m_paceNext += NanoSeconds(lenNs);
    m_tcb->m_pacingDeparture = m_paceNext;
    NS_LOG_DEBUG("Next GSO packet not before " << m_paceNext.As(Time::S));
}

void
TcpSocketBase::NotifyPacingPerformed()
{
    NS_LOG_FUNCTION(this);
    NS_LOG_INFO("Performing Pacing");
    SendPendingData(m_connected);
}

bool
TcpSocketBase::IsPacingEnabled() const
{
    if (!m_tcb->m_pacing)
    {
        return false;
    }
    else
    {
        if (m_tcb->m_paceInitialWindow)
        {
            return true;
        }
        // The octets sent so far, counted from the initial sequence number
        uint32_t sent = m_tcb->m_highTxMark.Get().GetValue() - m_isn.GetValue();
        if (sent > (GetInitialCwnd() * m_tcb->m_segmentSize))
        {
            return true;
        }
    }
    return false;
}

void
TcpSocketBase::UpdatePacingRate()
{
    NS_LOG_FUNCTION(this << m_tcb);

    // According to Linux, set base pacing rate to (cwnd * mss) / srtt
    //
    // In (early) slow start, multiply base by the slow start factor.
    // In late slow start and congestion avoidance, multiply base by
    // the congestion avoidance factor.
    // Comment from Linux code regarding early/late slow start:
    // Normal Slow Start condition is (tp->snd_cwnd < tp->snd_ssthresh)
    // If snd_cwnd >= (tp->snd_ssthresh / 2), we are approaching
    // end of slow start and should slow down.

    // Similar to Linux, do not update pacing rate here if the
    // congestion control implements TcpCongestionOps::CongControl ()
    if (m_congestionControl->HasCongControl() || !m_tcb->m_pacing)
    {
        return;
    }

    double factor;
    if (m_tcb->m_cWnd < m_tcb->m_ssThresh / 2)
    {
        NS_LOG_DEBUG("Pacing according to slow start factor; " << m_tcb->m_cWnd << " "
                                                               << m_tcb->m_ssThresh);
        factor = static_cast<double>(m_tcb->m_pacingSsRatio) / 100;
    }
    else
    {
        NS_LOG_DEBUG("Pacing according to congestion avoidance factor; " << m_tcb->m_cWnd << " "
                                                                         << m_tcb->m_ssThresh);
        factor = static_cast<double>(m_tcb->m_pacingCaRatio) / 100;
    }
    Time srtt = m_tcb->m_srtt.Get(); // Get underlying Time value
    NS_LOG_DEBUG("Smoothed RTT is " << srtt.GetSeconds());

    // Multiply by 8 to convert from bytes per second to bits per second
    DataRate pacingRate((std::max(m_tcb->m_cWnd, m_tcb->m_bytesInFlight) * 8 * factor) /
                        srtt.GetSeconds());
    if (pacingRate < m_tcb->m_maxPacingRate)
    {
        NS_LOG_DEBUG("Pacing rate updated to: " << pacingRate);
        m_tcb->m_pacingRate = pacingRate;
    }
    else
    {
        NS_LOG_DEBUG("Pacing capped by max pacing rate: " << m_tcb->m_maxPacingRate);
        m_tcb->m_pacingRate = m_tcb->m_maxPacingRate;
    }
}

void
TcpSocketBase::SetPacingStatus(bool pacing)
{
    NS_LOG_FUNCTION(this << pacing);
    m_tcb->m_pacing = pacing;
}

void
TcpSocketBase::SetPaceInitialWindow(bool paceWindow)
{
    NS_LOG_FUNCTION(this << paceWindow);
    m_tcb->m_paceInitialWindow = paceWindow;
}

bool
TcpSocketBase::IsEct(TcpPacketType_t packetType) const
{
    NS_LOG_FUNCTION(this << packetType);
    NS_ASSERT_MSG(packetType != TcpPacketType_t::INVALID, "Invalid TCP packet type");
    if (m_tcb->m_ecnState == TcpSocketState::ECN_DISABLED)
    {
        return false;
    }

    NS_ABORT_MSG_IF(!ECN_RESTRICTION_MAP.contains(std::make_pair(packetType, m_tcb->m_ecnMode)),
                    "Invalid packetType and ecnMode");

    return ECN_RESTRICTION_MAP.at(std::make_pair(packetType, m_tcb->m_ecnMode));
}

void
TcpSocketBase::SetUseEcn(TcpSocketState::UseEcn_t useEcn)
{
    NS_LOG_FUNCTION(this << useEcn);
    m_tcb->m_useEcn = useEcn;
}

void
TcpSocketBase::SetUseAbe(bool useAbe)
{
    NS_LOG_FUNCTION(this << useAbe);
    if (m_tcb->m_useEcn == TcpSocketState::Off && useAbe)
    {
        NS_LOG_INFO("Enabling ECN along with ABE");
        m_tcb->m_useEcn = TcpSocketState::On;
    }
    m_tcb->m_abeEnabled = useAbe;
}

bool
TcpSocketBase::GetUseAbe() const
{
    return m_tcb->m_abeEnabled;
}

uint32_t
TcpSocketBase::GetRWnd() const
{
    return m_rWnd.Get();
}

SequenceNumber32
TcpSocketBase::GetHighRxAck() const
{
    return m_highRxAckMark.Get();
}

// RttHistory methods
RttHistory::RttHistory(SequenceNumber32 s, uint32_t c, Time t)
    : seq(s),
      count(c),
      time(t),
      retx(false)
{
}

RttHistory::RttHistory(const RttHistory& h)
    : seq(h.seq),
      count(h.count),
      time(h.time),
      retx(h.retx)
{
}

} // namespace ns3
