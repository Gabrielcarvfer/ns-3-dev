/*
 * Copyright (c) 2026 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
 */

#include "ns3/buffer.h"
#include "ns3/he-capabilities.h"
#include "ns3/test.h"

#include <vector>

using namespace ns3;

/**
 * @ingroup wifi-test
 * @ingroup tests
 *
 * @brief Test of the Supported HE-MCS And NSS Set field of the HE Capabilities
 * element (IEEE 802.11ax-2021 9.4.2.248.4): an element sent by Linux (mac80211),
 * and the elements of ns-3 with and without 160 MHz support.
 */
class HeCapabilitiesMcsMapTest : public TestCase
{
  public:
    HeCapabilitiesMcsMapTest();

  private:
    void DoRun() override;

    /**
     * Serialize an element.
     *
     * @param element the element
     * @return its octets
     */
    static std::vector<uint8_t> Serialize(const HeCapabilities& element);

    /**
     * Deserialize an element.
     *
     * @param octets the octets of the element, from the Element ID
     * @return the element
     */
    static HeCapabilities Deserialize(const std::vector<uint8_t>& octets);
};

HeCapabilitiesMcsMapTest::HeCapabilitiesMcsMapTest()
    : TestCase("Supported HE-MCS And NSS Set field of the HE Capabilities element")
{
}

std::vector<uint8_t>
HeCapabilitiesMcsMapTest::Serialize(const HeCapabilities& element)
{
    Buffer buffer;
    buffer.AddAtStart(element.GetSerializedSize());
    element.Serialize(buffer.Begin());
    std::vector<uint8_t> octets(buffer.GetSize());
    buffer.CopyData(octets.data(), octets.size());
    return octets;
}

HeCapabilities
HeCapabilitiesMcsMapTest::Deserialize(const std::vector<uint8_t>& octets)
{
    Buffer buffer;
    buffer.AddAtStart(octets.size());
    buffer.Begin().Write(octets.data(), octets.size());
    HeCapabilities element;
    element.Deserialize(buffer.Begin());
    return element;
}

void
HeCapabilitiesMcsMapTest::DoRun()
{
    // Sent by Linux 6.12 (mac80211) in a Probe Request: 40/80 MHz in 5 GHz, Rx and
    // Tx HE-MCS Maps 0xfffa (HE-MCS 0-11 for one and two spatial streams)
    const std::vector<uint8_t> linux{0xff, 0x16, 0x23, 0x01, 0x00, 0x00, 0x02, 0x00,
                                     0x00, 0x04, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00,
                                     0x00, 0x00, 0x00, 0x00, 0xfa, 0xff, 0xfa, 0xff};
    auto linuxCaps = Deserialize(linux);
    NS_TEST_EXPECT_MSG_EQ(+linuxCaps.GetChannelWidthSet(), 0x02, "Wrong channel width set");
    NS_TEST_EXPECT_MSG_EQ(+linuxCaps.GetHighestNssSupported(), 2, "Wrong highest NSS");
    NS_TEST_EXPECT_MSG_EQ(+linuxCaps.GetHighestMcsSupported(), 11, "Wrong highest MCS");
    NS_TEST_EXPECT_MSG_EQ(linuxCaps.IsSupportedRxMcs(11), true, "HE-MCS 11 not supported");
    auto octets = Serialize(linuxCaps);
    NS_TEST_EXPECT_MSG_EQ(octets.size(), linux.size(), "Wrong size of the re-serialized element");
    NS_TEST_EXPECT_MSG_EQ((std::vector<uint8_t>(octets.end() - 4, octets.end()) ==
                           std::vector<uint8_t>(linux.end() - 4, linux.end())),
                          true,
                          "Wrong re-serialized HE-MCS Maps");

    // ns-3: HE-MCS 0-9 for one spatial stream, with and without 160 MHz
    for (uint8_t channelWidthSet : {0x02, 0x06})
    {
        HeCapabilities caps;
        caps.SetChannelWidthSet(channelWidthSet);
        caps.SetHighestMcsSupported(9);
        caps.SetHighestNssSupported(1);
        octets = Serialize(caps);
        const std::size_t maps = (channelWidthSet & 0x04) ? 4 : 2;
        NS_TEST_EXPECT_MSG_EQ(octets.size(), 20 + 2 * maps, "Wrong size of the element");
        for (std::size_t i = 0; i < maps; ++i)
        {
            NS_TEST_EXPECT_MSG_EQ(+octets[20 + 2 * i], 0xfd, "Wrong HE-MCS Map " << i);
            NS_TEST_EXPECT_MSG_EQ(+octets[21 + 2 * i], 0xff, "Wrong HE-MCS Map " << i);
        }
        auto roundTrip = Deserialize(octets);
        NS_TEST_EXPECT_MSG_EQ(+roundTrip.GetHighestNssSupported(), 1, "Wrong highest NSS");
        NS_TEST_EXPECT_MSG_EQ(+roundTrip.GetHighestMcsSupported(), 9, "Wrong highest MCS");
        NS_TEST_EXPECT_MSG_EQ(roundTrip.IsSupportedRxMcs(10), false, "HE-MCS 10 supported");
    }

    // highest MCSs that cannot be encoded are rounded down: 10 to 9, 8 to 7
    HeCapabilities caps;
    caps.SetHighestNssSupported(4);
    caps.SetHighestMcsSupported(10);
    NS_TEST_EXPECT_MSG_EQ(caps.GetSupportedMcsAndNss(), 0xff55, "Wrong HE-MCS Map for MCS 10");
    caps.SetHighestMcsSupported(8);
    NS_TEST_EXPECT_MSG_EQ(caps.GetSupportedMcsAndNss(), 0xff00, "Wrong HE-MCS Map for MCS 8");
}

/**
 * @ingroup wifi-test
 * @ingroup tests
 *
 * @brief HE Capabilities element test suite
 */
class HeCapabilitiesTestSuite : public TestSuite
{
  public:
    HeCapabilitiesTestSuite();
};

HeCapabilitiesTestSuite::HeCapabilitiesTestSuite()
    : TestSuite("wifi-he-capabilities", Type::UNIT)
{
    AddTestCase(new HeCapabilitiesMcsMapTest, TestCase::Duration::QUICK);
}

static HeCapabilitiesTestSuite g_heCapabilitiesTestSuite; ///< the test suite
