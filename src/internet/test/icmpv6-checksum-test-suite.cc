/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

#include "ns3/icmpv6-header.h"
#include "ns3/icmpv6-l4-protocol.h"
#include "ns3/ipv6-address.h"
#include "ns3/packet.h"
#include "ns3/test.h"

#include <vector>

using namespace ns3;

/**
 * @ingroup internet-test
 *
 * @brief Verify that Icmpv6L4Protocol::CheckIcmpv6Checksum accepts a correctly
 * formed ICMPv6 message and rejects a corrupted one (issue #1036).
 *
 * Before the fix, the ICMPv6 receive path never validated the checksum, so
 * packets with a wrong checksum were processed instead of being discarded.
 */
class Icmpv6ChecksumTestCase : public TestCase
{
  public:
    Icmpv6ChecksumTestCase()
        : TestCase("ICMPv6 checksum verification")
    {
    }

  private:
    void DoRun() override;
};

void
Icmpv6ChecksumTestCase::DoRun()
{
    Ipv6Address src("2001:db8::1");
    Ipv6Address dst("2001:db8::2");

    // Build a valid ICMPv6 Echo Request with a non-trivial payload and a
    // checksum computed exactly as on the transmission path.
    uint8_t payload[16];
    for (uint8_t i = 0; i < sizeof(payload); ++i)
    {
        payload[i] = i;
    }
    auto packet = Create<Packet>(payload, sizeof(payload));

    Icmpv6Echo echo(true);
    echo.SetId(1234);
    echo.SetSeq(1);
    echo.CalculatePseudoHeaderChecksum(src,
                                       dst,
                                       packet->GetSize() + echo.GetSerializedSize(),
                                       Icmpv6L4Protocol::PROT_NUMBER);
    packet->AddHeader(echo);

    // A correctly formed packet must pass verification.
    NS_TEST_ASSERT_MSG_EQ(Icmpv6L4Protocol::CheckIcmpv6Checksum(packet, src, dst),
                          true,
                          "Valid ICMPv6 checksum was rejected");

    // Corrupt a single byte of the serialized message: the checksum must now
    // fail to verify.
    uint32_t size = packet->GetSize();
    std::vector<uint8_t> bytes(size);
    packet->CopyData(bytes.data(), size);
    bytes[size - 1] ^= 0xFF;
    auto corrupted = Create<Packet>(bytes.data(), size);
    NS_TEST_ASSERT_MSG_EQ(Icmpv6L4Protocol::CheckIcmpv6Checksum(corrupted, src, dst),
                          false,
                          "Corrupted ICMPv6 packet was accepted as valid");

    // Corrupting the checksum field itself must also be detected.
    bytes[size - 1] ^= 0xFF; // restore payload
    bytes[2] ^= 0xFF;        // checksum field is at offset 2 of the ICMPv6 header
    auto badChecksum = Create<Packet>(bytes.data(), size);
    NS_TEST_ASSERT_MSG_EQ(Icmpv6L4Protocol::CheckIcmpv6Checksum(badChecksum, src, dst),
                          false,
                          "ICMPv6 packet with tampered checksum field was accepted");
}

/**
 * @ingroup internet-test
 *
 * @brief ICMPv6 checksum test suite.
 */
class Icmpv6ChecksumTestSuite : public TestSuite
{
  public:
    Icmpv6ChecksumTestSuite()
        : TestSuite("icmpv6-checksum", Type::UNIT)
    {
        AddTestCase(new Icmpv6ChecksumTestCase, TestCase::Duration::QUICK);
    }
};

static Icmpv6ChecksumTestSuite g_icmpv6ChecksumTestSuite; //!< Static variable for test
                                                          //!< initialization
