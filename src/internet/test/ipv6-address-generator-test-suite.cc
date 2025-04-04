/*
 * Copyright (c) 2008 University of Washington
 * Copyright (c) 2011 Atishay Jain
 *
 * SPDX-License-Identifier: GPL-2.0-only
 */

#include "ns3/ipv6-address-generator.h"
#include "ns3/simulation-singleton.h"
#include "ns3/test.h"

using namespace ns3;

/**
 * @ingroup internet-test
 *
 * @brief IPv6 network number allocator Test
 */
class NetworkNumber6AllocatorTestCase : public TestCase
{
  public:
    NetworkNumber6AllocatorTestCase();
    void DoRun() override;
    void DoTeardown() override;
};

NetworkNumber6AllocatorTestCase::NetworkNumber6AllocatorTestCase()
    : TestCase("Make sure the network number allocator is working on some of network prefixes.")
{
}

void
NetworkNumber6AllocatorTestCase::DoTeardown()
{
    Ipv6AddressGenerator::Reset();
}

void
NetworkNumber6AllocatorTestCase::DoRun()
{
    Ipv6Address network;

    Ipv6AddressGenerator::Init(Ipv6Address("1::0:0:0"), Ipv6Prefix("FFFF::0"), Ipv6Address("::"));
    network = Ipv6AddressGenerator::GetNetwork(Ipv6Prefix("FFFF::0"));
    NS_TEST_EXPECT_MSG_EQ(network,
                          Ipv6Address("1::0:0:0"),
                          "network should equal the initialized network for given prefix");
    network = Ipv6AddressGenerator::NextNetwork(Ipv6Prefix("FFFF::0"));
    NS_TEST_EXPECT_MSG_EQ(network, Ipv6Address("2::0:0:0"), "network should equal next network");

    Ipv6AddressGenerator::Init(Ipv6Address("0:1::0:0"),
                               Ipv6Prefix("FFFF:FFFF::0"),
                               Ipv6Address("::"));
    network = Ipv6AddressGenerator::GetNetwork(Ipv6Prefix("FFFF:FFFF::0"));
    NS_TEST_EXPECT_MSG_EQ(network,
                          Ipv6Address("0:1::0"),
                          "network should equal the initialized network for given prefix");
    network = Ipv6AddressGenerator::NextNetwork(Ipv6Prefix("FFFF:FFFF::0"));
    NS_TEST_EXPECT_MSG_EQ(network, Ipv6Address("0:2::0"), "network should equal next network");

    Ipv6AddressGenerator::Init(Ipv6Address("0:0:1::0"),
                               Ipv6Prefix("FFFF:FFFF:FFFF::0"),
                               Ipv6Address("::0"));
    network = Ipv6AddressGenerator::GetNetwork(Ipv6Prefix("FFFF:FFFF:FFFF::0"));
    NS_TEST_EXPECT_MSG_EQ(network,
                          Ipv6Address("0:0:1::0"),
                          "network should equal the initialized network for given prefix");
    network = Ipv6AddressGenerator::NextNetwork(Ipv6Prefix("FFFF:FFFF:FFFF::0"));
    NS_TEST_EXPECT_MSG_EQ(network, Ipv6Address("0:0:2::0"), "network should equal next network");
}

/**
 * @ingroup internet-test
 *
 * @brief IPv6 address allocator Test
 */
class AddressAllocator6TestCase : public TestCase
{
  public:
    AddressAllocator6TestCase();

  private:
    void DoRun() override;
    void DoTeardown() override;
};

AddressAllocator6TestCase::AddressAllocator6TestCase()
    : TestCase("Sanity check on allocation of addresses")
{
}

void
AddressAllocator6TestCase::DoRun()
{
    Ipv6Address address;

    Ipv6AddressGenerator::Init(Ipv6Address("2001::0"), Ipv6Prefix(64));
    address = Ipv6AddressGenerator::GetNetwork(Ipv6Prefix(64));
    NS_TEST_EXPECT_MSG_EQ(address,
                          Ipv6Address("2001::0"),
                          "address should equal the initialized address for given prefix");
    Ipv6AddressGenerator::NextNetwork(Ipv6Prefix(64));
    address = Ipv6AddressGenerator::GetNetwork(Ipv6Prefix(64));
    NS_TEST_EXPECT_MSG_EQ(address,
                          Ipv6Address("2001:0:0:1::0"),
                          "address should equal the initialized address for given prefix");
    address = Ipv6AddressGenerator::GetAddress(Ipv6Prefix(64));
    NS_TEST_EXPECT_MSG_EQ(address,
                          Ipv6Address("2001:0:0:1::1"),
                          "address should equal the initialized address for given prefix");
    address = Ipv6AddressGenerator::NextAddress(Ipv6Prefix(64));
    NS_TEST_EXPECT_MSG_EQ(address,
                          Ipv6Address("2001:0:0:1::1"),
                          "address should equal the initialized address for given prefix");
    address = Ipv6AddressGenerator::NextAddress(Ipv6Prefix(64));
    NS_TEST_EXPECT_MSG_EQ(address,
                          Ipv6Address("2001:0:0:1::2"),
                          "address should equal the initialized address for given prefix");

    Ipv6AddressGenerator::Init(Ipv6Address("1::"), Ipv6Prefix("FFFF::"), Ipv6Address("::3"));
    address = Ipv6AddressGenerator::NextAddress(Ipv6Prefix(16));
    NS_TEST_EXPECT_MSG_EQ(address,
                          Ipv6Address("1::3"),
                          "address should equal initialized address for given prefix");
    address = Ipv6AddressGenerator::NextAddress(Ipv6Prefix(16));
    NS_TEST_EXPECT_MSG_EQ(address, Ipv6Address("1::4"), "address should equal next address");
}

void
AddressAllocator6TestCase::DoTeardown()
{
    Ipv6AddressGenerator::Reset();
    Simulator::Destroy();
}

/**
 * @ingroup internet-test
 *
 * @brief IPv6 network number and address allocator Test
 */
class NetworkAndAddress6TestCase : public TestCase
{
  public:
    NetworkAndAddress6TestCase();
    void DoRun() override;
    void DoTeardown() override;
};

NetworkAndAddress6TestCase::NetworkAndAddress6TestCase()
    : TestCase("Make sure Network and address allocation play together.")
{
}

void
NetworkAndAddress6TestCase::DoTeardown()
{
    Ipv6AddressGenerator::Reset();
    Simulator::Destroy();
}

void
NetworkAndAddress6TestCase::DoRun()
{
    Ipv6Address address;
    Ipv6Address network;

    Ipv6AddressGenerator::Init(Ipv6Address("3::"), Ipv6Prefix("FFFF::"), Ipv6Address("::3"));
    address = Ipv6AddressGenerator::NextAddress(Ipv6Prefix(16));
    NS_TEST_EXPECT_MSG_EQ(address,
                          Ipv6Address("3::3"),
                          "address should equal initialized address for given prefix");
    address = Ipv6AddressGenerator::NextAddress(Ipv6Prefix(16));
    NS_TEST_EXPECT_MSG_EQ(address,
                          Ipv6Address("3::4"),
                          "address should equal next address for given prefix");

    network = Ipv6AddressGenerator::NextNetwork(Ipv6Prefix("FFFF::"));
    NS_TEST_EXPECT_MSG_EQ(network,
                          Ipv6Address("4::0"),
                          "address should equal next address for given prefix");
    address = Ipv6AddressGenerator::NextAddress(Ipv6Prefix(16));
    NS_TEST_EXPECT_MSG_EQ(address,
                          Ipv6Address("4::3"),
                          "address should equal next address for given prefix");
}

/**
 * @ingroup internet-test
 *
 * @brief IPv6 example of an address generator Test
 */
class ExampleAddress6GeneratorTestCase : public TestCase
{
  public:
    ExampleAddress6GeneratorTestCase();

  private:
    void DoRun() override;
    void DoTeardown() override;
};

ExampleAddress6GeneratorTestCase::ExampleAddress6GeneratorTestCase()
    : TestCase("A typical real-world example")
{
}

void
ExampleAddress6GeneratorTestCase::DoTeardown()
{
    Ipv6AddressGenerator::Reset();
}

void
ExampleAddress6GeneratorTestCase::DoRun()
{
    Ipv6Address address;

    Ipv6AddressGenerator::Init(Ipv6Address("2001:0AB8::"),
                               Ipv6Prefix("FFFF:FFFF:FFFF::0"),
                               Ipv6Address("::3"));
    address = Ipv6AddressGenerator::NextAddress(Ipv6Prefix(48));
    NS_TEST_EXPECT_MSG_EQ(address,
                          Ipv6Address("2001:0AB8::0:3"),
                          "address should equal initialized address for given prefix");
    address = Ipv6AddressGenerator::NextAddress(Ipv6Prefix(48));
    NS_TEST_EXPECT_MSG_EQ(address,
                          Ipv6Address("2001:0AB8::0:4"),
                          "address should equal next address for given prefix");
    address = Ipv6AddressGenerator::NextAddress(Ipv6Prefix(48));
    NS_TEST_EXPECT_MSG_EQ(address,
                          Ipv6Address("2001:0AB8::0:5"),
                          "address should equal next address for given prefix");
    //
    // Allocate the next network based on the prefix passed in, which should
    // be 2001:0AB0:0001
    //
    Ipv6AddressGenerator::NextNetwork(Ipv6Prefix("FFFF:FFFF:FFFF::0"));
    //
    // reset first address to be allocated back to ::0:3
    //
    Ipv6AddressGenerator::InitAddress(Ipv6Address("::3"), Ipv6Prefix(48));
    //
    // The first address we should get is the network and address ORed
    //
    address = Ipv6AddressGenerator::NextAddress(Ipv6Prefix(48));
    NS_TEST_EXPECT_MSG_EQ(address,
                          Ipv6Address("2001:0AB8:1::3"),
                          "address should equal initialized address for given prefix");
}

/**
 * @ingroup internet-test
 *
 * @brief IPv6 address collision Test
 */
class AddressCollision6TestCase : public TestCase
{
  public:
    AddressCollision6TestCase();

  private:
    void DoRun() override;
    void DoTeardown() override;
};

AddressCollision6TestCase::AddressCollision6TestCase()
    : TestCase("Make sure that the address collision logic works.")
{
}

void
AddressCollision6TestCase::DoTeardown()
{
    Ipv6AddressGenerator::Reset();
    Simulator::Destroy();
}

void
AddressCollision6TestCase::DoRun()
{
    Ipv6AddressGenerator::AddAllocated("0::0:5");
    Ipv6AddressGenerator::AddAllocated("0::0:10");
    Ipv6AddressGenerator::AddAllocated("0::0:15");
    Ipv6AddressGenerator::AddAllocated("0::0:20");

    Ipv6AddressGenerator::AddAllocated("0::0:4");
    Ipv6AddressGenerator::AddAllocated("0::0:3");
    Ipv6AddressGenerator::AddAllocated("0::0:2");
    Ipv6AddressGenerator::AddAllocated("0::0:1");

    Ipv6AddressGenerator::AddAllocated("0::0:6");
    Ipv6AddressGenerator::AddAllocated("0::0:7");
    Ipv6AddressGenerator::AddAllocated("0::0:8");
    Ipv6AddressGenerator::AddAllocated("0::0:9");

    Ipv6AddressGenerator::AddAllocated("0::0:11");
    Ipv6AddressGenerator::AddAllocated("0::0:12");
    Ipv6AddressGenerator::AddAllocated("0::0:13");
    Ipv6AddressGenerator::AddAllocated("0::0:14");

    Ipv6AddressGenerator::AddAllocated("0::0:19");
    Ipv6AddressGenerator::AddAllocated("0::0:18");
    Ipv6AddressGenerator::AddAllocated("0::0:17");
    Ipv6AddressGenerator::AddAllocated("0::0:16");

    Ipv6AddressGenerator::TestMode();
    bool allocated = Ipv6AddressGenerator::IsAddressAllocated("0::0:21");
    NS_TEST_EXPECT_MSG_EQ(allocated, false, "0::0:21 should not be already allocated");
    bool added = Ipv6AddressGenerator::AddAllocated("0::0:21");
    NS_TEST_EXPECT_MSG_EQ(added, true, "address should get allocated");

    allocated = Ipv6AddressGenerator::IsAddressAllocated("0::0:4");
    NS_TEST_EXPECT_MSG_EQ(allocated, true, "0::0:4 should be already allocated");
    added = Ipv6AddressGenerator::AddAllocated("0::0:4");
    NS_TEST_EXPECT_MSG_EQ(added, false, "address should not get allocated");

    allocated = Ipv6AddressGenerator::IsAddressAllocated("0::0:9");
    NS_TEST_EXPECT_MSG_EQ(allocated, true, "0::0:9 should be already allocated");
    added = Ipv6AddressGenerator::AddAllocated("0::0:9");
    NS_TEST_EXPECT_MSG_EQ(added, false, "address should not get allocated");

    allocated = Ipv6AddressGenerator::IsAddressAllocated("0::0:16");
    NS_TEST_EXPECT_MSG_EQ(allocated, true, "0::0:16 should be already allocated");
    added = Ipv6AddressGenerator::AddAllocated("0::0:16");
    NS_TEST_EXPECT_MSG_EQ(added, false, "address should not get allocated");

    allocated = Ipv6AddressGenerator::IsAddressAllocated("0::0:21");
    NS_TEST_EXPECT_MSG_EQ(allocated, true, "0::0:21 should be already allocated");
    added = Ipv6AddressGenerator::AddAllocated("0::0:21");
    NS_TEST_EXPECT_MSG_EQ(added, false, "address should not get allocated");
}

/**
 * @ingroup internet-test
 *
 * @brief IPv6 address generator TestSuite
 */
class Ipv6AddressGeneratorTestSuite : public TestSuite
{
  public:
    Ipv6AddressGeneratorTestSuite()
        : TestSuite("ipv6-address-generator")
    {
        AddTestCase(new NetworkNumber6AllocatorTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new AddressAllocator6TestCase(), TestCase::Duration::QUICK);
        AddTestCase(new NetworkAndAddress6TestCase(), TestCase::Duration::QUICK);
        AddTestCase(new ExampleAddress6GeneratorTestCase(), TestCase::Duration::QUICK);
        AddTestCase(new AddressCollision6TestCase(), TestCase::Duration::QUICK);
    }
};

static Ipv6AddressGeneratorTestSuite
    g_ipv6AddressGeneratorTestSuite; //!< Static variable for test initialization
