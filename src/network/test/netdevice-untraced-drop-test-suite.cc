/*
 * Copyright (c) 2026 ns-3 contributors
 *
 * SPDX-License-Identifier: GPL-2.0-only
 */

#include "ns3/address.h"
#include "ns3/callback.h"
#include "ns3/mac48-address.h"
#include "ns3/node.h"
#include "ns3/packet.h"
#include "ns3/queue.h"
#include "ns3/simple-channel.h"
#include "ns3/simple-net-device.h"
#include "ns3/simulator.h"
#include "ns3/string.h"
#include "ns3/test.h"

using namespace ns3;

/**
 * @file
 *
 * KNOWN-FAILING regression test for ns-3 issue #370.
 *
 * Issue #370: a packet received by a NetDevice can be dropped without being
 * traced when no registered ProtocolHandler matches the protocol number.
 *
 * Root cause: NetDevices (e.g. SimpleNetDevice, PointToPointNetDevice) invoke
 * their receive callback @c m_rxCallback(...) and ignore its @c bool return
 * value. The receive callback is bound to Node::NonPromiscReceiveFromDevice,
 * which delegates to Node::ReceiveFromDevice (src/network/model/node.cc). That
 * method returns @c false when no registered ProtocolHandler matches the
 * protocol number of the incoming frame. Because the device ignores this
 * return value, the frame is silently discarded and NO drop trace is fired.
 * SimpleNetDevice exposes a single drop trace source, "PhyRxDrop", but on the
 * no-handler path that trace is never reached: it is only fired from
 * SimpleNetDevice::Receive() when the optional receive ErrorModel marks the
 * packet as corrupt.
 *
 * This test builds a minimal two-node SimpleNetDevice link. The receiver's
 * device is installed on a Node (so the receive callback is the real
 * Node::NonPromiscReceiveFromDevice), but NO protocol handler is registered on
 * that Node for the chosen protocol number. A single unicast packet addressed
 * to the receiver's MAC address is sent. The receiver therefore drops the
 * packet for lack of a matching handler.
 *
 * The test connects to every drop trace source SimpleNetDevice provides
 * ("PhyRxDrop") and asserts that at least one drop is traced. On the current
 * (pre-fix) tree the packet is dropped silently: "PhyRxDrop" never fires, the
 * drop counter remains 0, and the assertion FAILS BY DESIGN. Once the device
 * honors the receive-callback return value and fires a drop trace on the
 * no-handler path, this test will pass.
 */

/**
 * @ingroup network-test
 * @ingroup tests
 *
 * @brief KNOWN-FAILING regression test for issue #370: an untraced packet drop
 *        on the "no matching protocol handler" receive path.
 *
 * The frame is delivered to the receiver device and discarded by
 * Node::ReceiveFromDevice (which returns false), but no drop trace is fired.
 */
class NetDeviceUntracedDropTestCase : public TestCase
{
  public:
    NetDeviceUntracedDropTestCase();
    ~NetDeviceUntracedDropTestCase() override;

  private:
    void DoRun() override;

    /**
     * Drop trace sink connected to the receiver device's "PhyRxDrop" trace.
     *
     * Counts the number of packets the receiver device reports as dropped
     * during reception.
     *
     * @param p The dropped packet.
     */
    void DropEvent(Ptr<const Packet> p);

    uint32_t m_drops; //!< Number of packet drops traced on the receiver device.
};

NetDeviceUntracedDropTestCase::NetDeviceUntracedDropTestCase()
    : TestCase("Packet dropped for missing protocol handler must be traced (issue #370)"),
      m_drops(0)
{
}

NetDeviceUntracedDropTestCase::~NetDeviceUntracedDropTestCase()
{
}

void
NetDeviceUntracedDropTestCase::DropEvent(Ptr<const Packet> p)
{
    m_drops++;
}

void
NetDeviceUntracedDropTestCase::DoRun()
{
    // Two nodes, two SimpleNetDevices, one SimpleChannel. Fully deterministic:
    // no random variables and no error model are involved.
    Ptr<Node> a = CreateObject<Node>();
    Ptr<Node> b = CreateObject<Node>();

    Ptr<SimpleNetDevice> sender = CreateObject<SimpleNetDevice>();
    Ptr<SimpleNetDevice> receiver = CreateObject<SimpleNetDevice>();
    Ptr<SimpleChannel> channel = CreateObject<SimpleChannel>();

    ObjectFactory queueFactory;
    queueFactory.SetTypeId("ns3::DropTailQueue<Packet>");
    queueFactory.Set("MaxSize", StringValue("100p"));
    sender->SetQueue(queueFactory.Create<Queue<Packet>>());
    receiver->SetQueue(queueFactory.Create<Queue<Packet>>());

    // AddDevice() binds the device's receive callback to
    // Node::NonPromiscReceiveFromDevice. We deliberately do NOT call
    // RegisterProtocolHandler() on node b, so no handler matches any protocol.
    a->AddDevice(sender);
    b->AddDevice(receiver);

    sender->SetNode(a);
    sender->SetChannel(channel);
    sender->SetAddress(Mac48Address::Allocate());

    receiver->SetNode(b);
    receiver->SetChannel(channel);
    receiver->SetAddress(Mac48Address::Allocate());

    // Connect to the only drop trace source SimpleNetDevice exposes. We expect
    // THIS trace ("PhyRxDrop") to fire once when the no-handler drop occurs.
    receiver->TraceConnectWithoutContext(
        "PhyRxDrop",
        MakeCallback(&NetDeviceUntracedDropTestCase::DropEvent, this));

    // Send exactly one unicast packet to the receiver's own MAC address, so the
    // receiver classifies it as PACKET_HOST and invokes its receive callback.
    // Protocol number 0x8888 has no registered handler on node b, so
    // Node::ReceiveFromDevice returns false and the packet is discarded.
    Ptr<Packet> pkt = Create<Packet>(64);
    Simulator::Schedule(Seconds(0),
                        &SimpleNetDevice::Send,
                        sender,
                        pkt,
                        receiver->GetAddress(),
                        static_cast<uint16_t>(0x8888));

    Simulator::Run();
    Simulator::Destroy();

    // The packet was delivered to the receiver device and discarded for lack of
    // a matching protocol handler. A correct implementation must trace this
    // drop, so we require at least one drop to be reported.
    //
    // PRE-FIX OBSERVATION: on the current tree no drop trace fires on the
    // no-handler path (SimpleNetDevice::Receive ignores the bool returned by
    // m_rxCallback / Node::ReceiveFromDevice), so m_drops == 0 here and this
    // assertion FAILS BY DESIGN. This is the intended failing regression test
    // for issue #370.
    NS_TEST_ASSERT_MSG_GT_OR_EQ(m_drops,
                                1,
                                "Packet dropped due to missing protocol handler was not traced "
                                "(issue #370): expected the device PhyRxDrop trace to fire.");
}

/**
 * @ingroup network-test
 * @ingroup tests
 *
 * @brief Test suite for the untraced no-handler NetDevice drop (issue #370).
 */
class NetDeviceUntracedDropTestSuite : public TestSuite
{
  public:
    NetDeviceUntracedDropTestSuite();
};

NetDeviceUntracedDropTestSuite::NetDeviceUntracedDropTestSuite()
    : TestSuite("netdevice-untraced-drop", Type::UNIT)
{
    AddTestCase(new NetDeviceUntracedDropTestCase, TestCase::Duration::QUICK);
}

/// Static variable for test initialization.
static NetDeviceUntracedDropTestSuite g_netdeviceUntracedDropTestSuite;
