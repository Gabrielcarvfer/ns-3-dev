/*
 * Copyright (c) 2026 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Unit tests for FdNetDevice.
 *
 * These tests use a pipe pair to exercise the read/write path of FdNetDevice
 * without requiring any real network interface, root privileges, or
 * platform-specific TAP/TUN drivers.  They run on all supported platforms
 * (Linux, macOS, Windows).
 *
 * Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
 */

#include "ns3/fd-net-device.h"
#include "ns3/global-value.h"
#include "ns3/node.h"
#include "ns3/simulator.h"
#include "ns3/string.h"
#include "ns3/test.h"

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>

// Map POSIX names to their Windows CRT equivalents so the rest of the file
// uses the portable names (pipe/read/write/close) unchanged.
inline int
pipe(int fds[2])
{
    return _pipe(fds, 65536, _O_BINARY);
}

#define close _close
#define read _read
#define write(fd, buf, len) _write((fd), (buf), static_cast<unsigned int>(len))
#else
#include <unistd.h>
#endif

using namespace ns3;

namespace
{

/**
 * Build the smallest valid Ethernet II frame containing a dummy IPv4 payload.
 * dst and src are 6-byte arrays in network order.
 */
static std::vector<uint8_t>
BuildEthernetFrame(const uint8_t dst[6], const uint8_t src[6], const uint8_t* payload, size_t len)
{
    std::vector<uint8_t> frame;
    frame.insert(frame.end(), dst, dst + 6);
    frame.insert(frame.end(), src, src + 6);
    frame.push_back(0x08); // EtherType = IPv4 (big-endian)
    frame.push_back(0x00);
    frame.insert(frame.end(), payload, payload + len);
    // Minimum Ethernet payload is 46 bytes; pad with zeros if needed.
    while (frame.size() < 14 + 46)
    {
        frame.push_back(0x00);
    }
    return frame;
}

} // anonymous namespace

// ==========================================================================
// Test 1: FdNetDevice receives a packet injected via a pipe
// ==========================================================================

/**
 * @ingroup fd-net-device
 * @brief Test suite for fd-net-device
 */
class FdNetDeviceReceiveTest : public TestCase
{
  public:
    FdNetDeviceReceiveTest()
        : TestCase("FdNetDevice receives a packet injected via a pipe"),
          m_received(false),
          m_protocol(0),
          m_pktSize(0)
    {
    }

  private:
    void DoRun() override;

    bool DoReceive(Ptr<NetDevice>, Ptr<const Packet> packet, uint16_t protocol, const Address&)
    {
        NS_LOG_UNCOND("FdNetDeviceReceiveTest: received packet, protocol=0x" << std::hex
                                                                             << protocol);
        m_received = true;
        m_protocol = protocol;
        m_pktSize = packet->GetSize();
        m_pktBuf.resize(m_pktSize);
        packet->CopyData(m_pktBuf.data(), m_pktSize);
        Simulator::Stop();
        return true;
    }

    bool m_received;               //!< set to true once a packet is forwarded up
    uint16_t m_protocol;           //!< EtherType from the forwarded packet
    uint32_t m_pktSize;            //!< size of the received network-layer PDU
    std::vector<uint8_t> m_pktBuf; //!< copy of the received bytes
};

void
FdNetDeviceReceiveTest::DoRun()
{
    // FdNetDevice's FdReader runs in a separate OS thread and interacts with
    // the simulator via ScheduleWithContext.  The DefaultSimulatorImpl
    // processes events with no wall-clock delay, so the FdReader thread
    // never gets CPU time between events.  RealtimeSimulatorImpl enforces
    // real wall-clock time, giving the FdReader thread proper time to start,
    // select(), read, and call ScheduleWithContext before the sim moves on.
    GlobalValue::Bind("SimulatorImplementationType", StringValue("ns3::RealtimeSimulatorImpl"));

    int fds[2];
    NS_TEST_ASSERT_MSG_EQ(pipe(fds), 0, "pipe() failed");

    Ptr<Node> node = CreateObject<Node>();
    Ptr<FdNetDevice> device = CreateObject<FdNetDevice>();

    Mac48Address devAddr("11:22:33:44:55:66");
    device->SetAddress(devAddr);
    device->SetEncapsulationMode(FdNetDevice::DIX);
    device->SetFileDescriptor(fds[0]); // FdNetDevice reads from fds[0]
    node->AddDevice(device);

    device->SetReceiveCallback(MakeCallback(&FdNetDeviceReceiveTest::DoReceive, this));

    device->Start(Seconds(0.0));

    // Inject an Ethernet frame into the write end of the pipe.
    uint8_t dst[6] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
    uint8_t src[6] = {0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF};
    uint8_t payload[20] = {}; // dummy IPv4 header-sized blob
    payload[0] = 0x45;        // IPv4, IHL=5

    auto frame = BuildEthernetFrame(dst, src, payload, sizeof(payload));
    ssize_t written = write(fds[1], frame.data(), static_cast<unsigned int>(frame.size()));
    NS_TEST_ASSERT_MSG_EQ((size_t)written, frame.size(), "write to pipe failed");

    // Safety net: DoReceive() calls Simulator::Stop() as soon as the packet
    // arrives, so the simulation normally ends well before this deadline.
    Simulator::Stop(Seconds(5));
    Simulator::Run();
    Simulator::Destroy();

    close(fds[1]);

    NS_TEST_ASSERT_MSG_EQ(m_received, true, "Packet was not received by the FdNetDevice");
    if (!m_received)
    {
        return; // m_pktBuf is empty; avoid out-of-bounds access below
    }

    // Verify protocol and payload bytes — FdNetDevice strips the 14-byte
    // Ethernet header so the callback sees only the network-layer PDU.
    NS_TEST_ASSERT_MSG_EQ(m_protocol, (uint16_t)0x0800, "Protocol (EtherType) mismatch");

    // Expected payload size: 20 bytes payload + 26 bytes zero-padding (min frame)
    constexpr uint32_t expectedPayloadSize = 46;
    NS_TEST_ASSERT_MSG_EQ(m_pktSize, expectedPayloadSize, "Received payload size mismatch");

    // First byte of the IPv4 header must survive the pipe round-trip intact.
    NS_TEST_ASSERT_MSG_EQ((uint32_t)m_pktBuf[0], (uint32_t)0x45, "IPv4 header first byte mismatch");

    // All 20 injected payload bytes must be intact (bytes 0..19 of the PDU).
    // The test payload has only the first byte set (0x45); the rest are zero.
    for (uint32_t i = 1; i < 20; ++i)
    {
        NS_TEST_ASSERT_MSG_EQ((uint32_t)m_pktBuf[i], (uint32_t)0x00, "Payload byte mismatch");
    }

    GlobalValue::Bind("SimulatorImplementationType", StringValue("ns3::DefaultSimulatorImpl"));
}

// ==========================================================================
// Test 2: FdNetDevice sends a packet out via the fd (read from the pipe)
// ==========================================================================

class FdNetDeviceSendTest : public TestCase
{
  public:
    FdNetDeviceSendTest()
        : TestCase("FdNetDevice writes a sent packet to the fd")
    {
    }

  private:
    void DoRun() override;
};

void
FdNetDeviceSendTest::DoRun()
{
    int fds[2];
    NS_TEST_ASSERT_MSG_EQ(pipe(fds), 0, "pipe() failed");

    Ptr<Node> node = CreateObject<Node>();
    Ptr<FdNetDevice> device = CreateObject<FdNetDevice>();

    Mac48Address devAddr("AA:BB:CC:DD:EE:FF");
    device->SetAddress(devAddr);
    device->SetEncapsulationMode(FdNetDevice::DIX);
    device->SetFileDescriptor(fds[1]); // FdNetDevice writes to fds[1]
    node->AddDevice(device);
    device->Start(Seconds(0.0));

    // Build a small dummy packet and send it
    Ptr<Packet> packet = Create<Packet>(64);
    Mac48Address dest("11:22:33:44:55:66");

    Simulator::Schedule(Seconds(0.0), [&]() { device->Send(packet, dest, 0x0800 /* IPv4 */); });

    Simulator::Stop(MilliSeconds(100));
    Simulator::Run();
    Simulator::Destroy();

    // The FdNetDevice should have written an Ethernet frame to fds[1].
    // Read it back from fds[0] and verify every field.
    uint8_t buf[256] = {};
    ssize_t bytesRead = read(fds[0], buf, sizeof(buf));
    // Frame = 6 dst + 6 src + 2 EtherType + 64 payload = 78 bytes
    NS_TEST_ASSERT_MSG_EQ((int)bytesRead, 78, "Frame size mismatch");

    // Destination MAC: 11:22:33:44:55:66
    const uint8_t expectedDst[6] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
    for (int i = 0; i < 6; ++i)
    {
        NS_TEST_ASSERT_MSG_EQ((uint32_t)buf[i], (uint32_t)expectedDst[i], "dst MAC byte mismatch");
    }

    // Source MAC: AA:BB:CC:DD:EE:FF (device address)
    const uint8_t expectedSrc[6] = {0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF};
    for (int i = 0; i < 6; ++i)
    {
        NS_TEST_ASSERT_MSG_EQ((uint32_t)buf[6 + i],
                              (uint32_t)expectedSrc[i],
                              "src MAC byte mismatch");
    }

    // EtherType: 0x0800 (IPv4)
    NS_TEST_ASSERT_MSG_EQ((uint32_t)buf[12], (uint32_t)0x08, "EtherType high byte mismatch");
    NS_TEST_ASSERT_MSG_EQ((uint32_t)buf[13], (uint32_t)0x00, "EtherType low byte mismatch");

    // Payload: Create<Packet>(64) produces 64 zero bytes — verify they arrive intact.
    for (int i = 0; i < 64; ++i)
    {
        NS_TEST_ASSERT_MSG_EQ((uint32_t)buf[14 + i], (uint32_t)0x00, "Payload byte mismatch");
    }

    close(fds[0]);
}

// ==========================================================================
// Test 3: FdNetDevice correctly handles DIXPI (PI header) mode
// ==========================================================================

class FdNetDeviceDixpiReceiveTest : public TestCase
{
  public:
    FdNetDeviceDixpiReceiveTest()
        : TestCase("FdNetDevice strips the 4-byte PI header in DIXPI mode"),
          m_received(false),
          m_protocol(0),
          m_pktSize(0)
    {
    }

  private:
    void DoRun() override;

    bool DoReceive(Ptr<NetDevice>, Ptr<const Packet> packet, uint16_t protocol, const Address&)
    {
        m_received = true;
        m_protocol = protocol;
        m_pktSize = packet->GetSize();
        m_pktBuf.resize(m_pktSize);
        packet->CopyData(m_pktBuf.data(), m_pktSize);
        Simulator::Stop();
        return true;
    }

    bool m_received;
    uint16_t m_protocol;
    uint32_t m_pktSize;
    std::vector<uint8_t> m_pktBuf;
};

void
FdNetDeviceDixpiReceiveTest::DoRun()
{
    GlobalValue::Bind("SimulatorImplementationType", StringValue("ns3::RealtimeSimulatorImpl"));

    int fds[2];
    NS_TEST_ASSERT_MSG_EQ(pipe(fds), 0, "pipe() failed");

    Ptr<Node> node = CreateObject<Node>();
    Ptr<FdNetDevice> device = CreateObject<FdNetDevice>();

    Mac48Address devAddr("11:22:33:44:55:66");
    device->SetAddress(devAddr);
    device->SetEncapsulationMode(FdNetDevice::DIXPI);
    device->SetFileDescriptor(fds[0]);
    node->AddDevice(device);
    device->SetReceiveCallback(MakeCallback(&FdNetDeviceDixpiReceiveTest::DoReceive, this));
    device->Start(Seconds(0.0));

    // PI header: flags(2) + proto(2) = 0x0000 0x0008 (IPv4 in LE)
    uint8_t dst[6] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
    uint8_t src[6] = {0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF};
    uint8_t payload[20] = {};
    payload[0] = 0x45;
    auto etherFrame = BuildEthernetFrame(dst, src, payload, sizeof(payload));

    // Prepend PI header (flags=0, proto=0x0800 in little-endian as the kernel writes it)
    std::vector<uint8_t> piFrame;
    piFrame.push_back(0x00); // flags low
    piFrame.push_back(0x00); // flags high
    piFrame.push_back(0x08); // proto low byte (0x0800 IPv4 in LE → 0x08)
    piFrame.push_back(0x00); // proto high byte
    piFrame.insert(piFrame.end(), etherFrame.begin(), etherFrame.end());

    ssize_t piWritten = write(fds[1], piFrame.data(), static_cast<unsigned int>(piFrame.size()));
    NS_TEST_ASSERT_MSG_EQ((size_t)piWritten, piFrame.size(), "write to pipe failed (DIXPI)");

    Simulator::Stop(Seconds(5));
    Simulator::Run();
    Simulator::Destroy();

    close(fds[1]);
    NS_TEST_ASSERT_MSG_EQ(m_received, true, "Packet was not received in DIXPI mode");
    if (!m_received)
    {
        return;
    }

    // After stripping the 4-byte PI header and the 14-byte Ethernet header the
    // callback should see the network-layer PDU (46 bytes: 20 payload + 26 pad).
    NS_TEST_ASSERT_MSG_EQ(m_protocol, (uint16_t)0x0800, "Protocol mismatch in DIXPI mode");
    NS_TEST_ASSERT_MSG_EQ(m_pktSize, (uint32_t)46, "PDU size mismatch in DIXPI mode");
    NS_TEST_ASSERT_MSG_EQ((uint32_t)m_pktBuf[0],
                          (uint32_t)0x45,
                          "IPv4 byte mismatch in DIXPI mode");

    GlobalValue::Bind("SimulatorImplementationType", StringValue("ns3::DefaultSimulatorImpl"));
}

// ==========================================================================
// Test 4: FdNetDevice UTUN mode — receives raw IP with 4-byte AF header
// ==========================================================================

class FdNetDeviceUtunReceiveTest : public TestCase
{
  public:
    FdNetDeviceUtunReceiveTest()
        : TestCase("FdNetDevice strips the 4-byte AF header in UTUN mode"),
          m_received(false),
          m_protocol(0),
          m_pktSize(0)
    {
    }

  private:
    void DoRun() override;

    bool DoReceive(Ptr<NetDevice>, Ptr<const Packet> packet, uint16_t protocol, const Address&)
    {
        m_received = true;
        m_protocol = protocol;
        m_pktSize = packet->GetSize();
        m_pktBuf.resize(m_pktSize);
        packet->CopyData(m_pktBuf.data(), m_pktSize);
        Simulator::Stop();
        return true;
    }

    bool m_received;
    uint16_t m_protocol;
    uint32_t m_pktSize;
    std::vector<uint8_t> m_pktBuf;
};

void
FdNetDeviceUtunReceiveTest::DoRun()
{
    GlobalValue::Bind("SimulatorImplementationType", StringValue("ns3::RealtimeSimulatorImpl"));

    int fds[2];
    NS_TEST_ASSERT_MSG_EQ(pipe(fds), 0, "pipe() failed");

    Ptr<Node> node = CreateObject<Node>();
    Ptr<FdNetDevice> device = CreateObject<FdNetDevice>();

    device->SetAddress(Mac48Address("AA:BB:CC:DD:EE:FF"));
    device->SetEncapsulationMode(FdNetDevice::UTUN);
    device->SetFileDescriptor(fds[0]);
    node->AddDevice(device);
    device->SetReceiveCallback(MakeCallback(&FdNetDeviceUtunReceiveTest::DoReceive, this));
    device->Start(Seconds(0.0));

    // AF_INET (2) in network byte order, followed by a minimal IPv4 header
    uint8_t pkt[24] = {};
    pkt[0] = 0x00; // AF high bytes
    pkt[1] = 0x00;
    pkt[2] = 0x00;
    pkt[3] = 0x02; // AF_INET = 2
    pkt[4] = 0x45; // IPv4 version + IHL
    pkt[5] = 0x00; // DSCP
    pkt[6] = 0x00; // total length high
    pkt[7] = 0x14; // total length = 20 bytes
    pkt[8] = 0x00; // ID
    pkt[9] = 0x01;
    pkt[10] = 0x00; // flags + fragment offset
    pkt[11] = 0x00;
    pkt[12] = 0x40; // TTL = 64
    pkt[13] = 0x11; // protocol = UDP
    pkt[14] = 0x00; // checksum (0 for test)
    pkt[15] = 0x00;
    pkt[16] = 192; // src 192.168.1.1
    pkt[17] = 168;
    pkt[18] = 1;
    pkt[19] = 1;
    pkt[20] = 10; // dst 10.0.0.1
    pkt[21] = 0;
    pkt[22] = 0;
    pkt[23] = 1;

    ssize_t utunWritten = write(fds[1], pkt, sizeof(pkt));
    NS_TEST_ASSERT_MSG_EQ((size_t)utunWritten, sizeof(pkt), "write to pipe failed (UTUN)");

    Simulator::Stop(Seconds(5));
    Simulator::Run();
    Simulator::Destroy();

    close(fds[1]);

    NS_TEST_ASSERT_MSG_EQ(m_received, true, "Packet was not received in UTUN mode");
    if (!m_received)
    {
        return;
    }
    NS_TEST_ASSERT_MSG_EQ(m_protocol, (uint16_t)0x0800, "Protocol should be IPv4 (0x0800)");

    // After stripping the 4-byte AF header the callback sees the raw IP datagram.
    // The datagram we injected is 20 bytes; verify key fields survived intact.
    NS_TEST_ASSERT_MSG_EQ(m_pktSize, (uint32_t)20, "IP datagram size mismatch");
    NS_TEST_ASSERT_MSG_EQ((uint32_t)m_pktBuf[0], (uint32_t)0x45, "IPv4 version+IHL mismatch");
    NS_TEST_ASSERT_MSG_EQ((uint32_t)m_pktBuf[9], (uint32_t)0x11, "IP protocol (UDP) mismatch");
    NS_TEST_ASSERT_MSG_EQ((uint32_t)m_pktBuf[12], (uint32_t)192, "Src IP[0] mismatch");
    NS_TEST_ASSERT_MSG_EQ((uint32_t)m_pktBuf[16], (uint32_t)10, "Dst IP[0] mismatch");

    GlobalValue::Bind("SimulatorImplementationType", StringValue("ns3::DefaultSimulatorImpl"));
}

// ==========================================================================
// Platform-specific probe tests
//
// Each of these tests exercises the real kernel/driver API for TAP/TUN on
// its respective OS.  If the required capability is not available (missing
// driver, no root/CAP_NET_ADMIN, old kernel, etc.) the test logs a message
// and returns without failing — it never calls NS_TEST_ASSERT_MSG_EQ with
// a false condition.  This allows the test suite to run everywhere while
// still exercising the native path when the environment supports it.
// ==========================================================================

// --------------------------------------------------------------------------
// Linux: TUN device probe
// --------------------------------------------------------------------------
#if defined(__linux__)
#include <errno.h>
#include <fcntl.h>
#include <linux/if_tun.h>
#include <net/if.h>
#include <string.h>
#include <sys/ioctl.h>

/**
 * @ingroup fd-net-device
 * @brief Linux TUN device availability probe
 */
class FdNetDeviceLinuxTunProbeTest : public TestCase
{
  public:
    FdNetDeviceLinuxTunProbeTest()
        : TestCase("Linux TUN device availability probe")
    {
    }

  private:
    void DoRun() override;
};

void
FdNetDeviceLinuxTunProbeTest::DoRun()
{
    int tunFd = open("/dev/net/tun", O_RDWR);
    if (tunFd < 0)
    {
        NS_LOG_UNCOND("[Linux TUN probe] /dev/net/tun unavailable (" << strerror(errno)
                                                                     << "); skipping");
        return;
    }

    struct ifreq ifr = {};
    ifr.ifr_flags = IFF_TUN | IFF_NO_PI;
    strncpy(ifr.ifr_name, "ns3tstun0", IFNAMSIZ - 1);

    if (ioctl(tunFd, TUNSETIFF, &ifr) < 0)
    {
        close(tunFd);
        NS_LOG_UNCOND("[Linux TUN probe] TUNSETIFF requires CAP_NET_ADMIN; skipping");
        return;
    }

    // FdNetDevice DIX smoke-test: start, run briefly, no crash = pass.
    Ptr<Node> node = CreateObject<Node>();
    Ptr<FdNetDevice> device = CreateObject<FdNetDevice>();
    device->SetAddress(Mac48Address("AA:BB:CC:DD:EE:01"));
    device->SetEncapsulationMode(FdNetDevice::DIX);
    device->SetFileDescriptor(tunFd);
    node->AddDevice(device);
    device->Start(Seconds(0.0));

    Simulator::Stop(MilliSeconds(20));
    Simulator::Run();
    Simulator::Destroy();

    NS_LOG_UNCOND("[Linux TUN probe] TUN interface '" << ifr.ifr_name << "' created OK");
}
#endif // __linux__

// --------------------------------------------------------------------------
// macOS: utun socket probe
// --------------------------------------------------------------------------
#if defined(__APPLE__)
#include <errno.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/kern_control.h>
#include <sys/socket.h>
#include <sys/sys_domain.h>

/**
 * @ingroup fd-net-device
 * @brief macOS utun socket creation probe
 */
class FdNetDeviceMacOsUtunProbeTest : public TestCase
{
  public:
    FdNetDeviceMacOsUtunProbeTest()
        : TestCase("macOS utun socket creation probe")
    {
    }

  private:
    void DoRun() override;
};

void
FdNetDeviceMacOsUtunProbeTest::DoRun()
{
    int fd = socket(PF_SYSTEM, SOCK_DGRAM, SYSPROTO_CONTROL);
    if (fd < 0)
    {
        NS_LOG_UNCOND("[macOS utun probe] socket(PF_SYSTEM) failed (" << strerror(errno)
                                                                      << "); skipping");
        return;
    }

    struct ctl_info info = {};
    strlcpy(info.ctl_name, "com.apple.net.utun_control", sizeof(info.ctl_name));
    if (ioctl(fd, CTLIOCGINFO, &info) < 0)
    {
        close(fd);
        NS_LOG_UNCOND("[macOS utun probe] CTLIOCGINFO failed (" << strerror(errno)
                                                                << "); skipping");
        return;
    }

    struct sockaddr_ctl addr = {};
    addr.sc_len = sizeof(addr);
    addr.sc_family = AF_SYSTEM;
    addr.ss_sysaddr = AF_SYS_CONTROL;
    addr.sc_id = info.ctl_id;
    addr.sc_unit = 0; // 0 = auto-assign unit number

    if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0)
    {
        close(fd);
        NS_LOG_UNCOND("[macOS utun probe] connect() failed (" << strerror(errno) << "); skipping");
        return;
    }

    // utun fd in hand — UTUN-mode FdNetDevice smoke-test.
    Ptr<Node> node = CreateObject<Node>();
    Ptr<FdNetDevice> device = CreateObject<FdNetDevice>();
    device->SetAddress(Mac48Address("AA:BB:CC:DD:EE:02"));
    device->SetEncapsulationMode(FdNetDevice::UTUN);
    device->SetFileDescriptor(fd);
    node->AddDevice(device);
    device->Start(Seconds(0.0));

    Simulator::Stop(MilliSeconds(20));
    Simulator::Run();
    Simulator::Destroy();

    NS_LOG_UNCOND("[macOS utun probe] utun interface created OK");
}
#endif // __APPLE__

// --------------------------------------------------------------------------
// Windows: tap-windows6 registry probe
// --------------------------------------------------------------------------
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <winreg.h>
#ifdef GetObject
#undef GetObject
#endif

/**
 * @ingroup fd-net-device
 * @brief Windows tap-windows6 adapter registry probe
 */
class FdNetDeviceWindowsTapRegistryTest : public TestCase
{
  public:
    FdNetDeviceWindowsTapRegistryTest()
        : TestCase("Windows tap-windows6 adapter registry probe")
    {
    }

  private:
    void DoRun() override;
};

void
FdNetDeviceWindowsTapRegistryTest::DoRun()
{
    static const char* classKey =
        "SYSTEM\\CurrentControlSet\\Control\\Class\\{4D36E972-E325-11CE-BFC1-08002BE10318}";
    HKEY hClass;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, classKey, 0, KEY_READ, &hClass) != ERROR_SUCCESS)
    {
        NS_LOG_UNCOND("[Win tap probe] Cannot open NDIS class key (may need admin); skipping");
        return;
    }

    int tapCount = 0;
    char subKey[256];
    for (DWORD i = 0;; ++i)
    {
        DWORD nameLen = sizeof(subKey);
        if (RegEnumKeyExA(hClass, i, subKey, &nameLen, nullptr, nullptr, nullptr, nullptr) !=
            ERROR_SUCCESS)
        {
            break;
        }
        HKEY hAdapter;
        if (RegOpenKeyExA(hClass, subKey, 0, KEY_READ, &hAdapter) != ERROR_SUCCESS)
        {
            continue;
        }
        char compId[256] = {};
        DWORD len = sizeof(compId);
        DWORD type = REG_SZ;
        RegQueryValueExA(hAdapter, "ComponentId", nullptr, &type, (LPBYTE)compId, &len);
        RegCloseKey(hAdapter);
        // tap0901 = tap-windows6 (OpenVPN), tap0801 = older variant
        if (strncmp(compId, "tap0901", 7) == 0 || strncmp(compId, "tap0801", 7) == 0)
        {
            ++tapCount;
        }
    }
    RegCloseKey(hClass);

    NS_LOG_UNCOND("[Win tap probe] found " << tapCount << " tap-windows6 adapter(s)");
    // Test passes regardless: the probe exercised the registry scan path.
}
#endif // _WIN32

// ==========================================================================
// Test suite registration
// ==========================================================================

/**
 * @ingroup fd-net-device
 * @brief Tests for FdNetDevice
 */
class FdNetDeviceTestSuite : public TestSuite
{
  public:
    FdNetDeviceTestSuite()
        : TestSuite("fd-net-device", Type::UNIT)
    {
        AddTestCase(new FdNetDeviceReceiveTest, TestCase::Duration::QUICK);
        AddTestCase(new FdNetDeviceSendTest, TestCase::Duration::QUICK);
        AddTestCase(new FdNetDeviceDixpiReceiveTest, TestCase::Duration::QUICK);
        AddTestCase(new FdNetDeviceUtunReceiveTest, TestCase::Duration::QUICK);
#if defined(__linux__)
        AddTestCase(new FdNetDeviceLinuxTunProbeTest, TestCase::Duration::QUICK);
#endif
#if defined(__APPLE__)
        AddTestCase(new FdNetDeviceMacOsUtunProbeTest, TestCase::Duration::QUICK);
#endif
#if defined(_WIN32)
        AddTestCase(new FdNetDeviceWindowsTapRegistryTest, TestCase::Duration::QUICK);
#endif
    }
};

static FdNetDeviceTestSuite g_fdNetDeviceTestSuite; //!< Static variable for test initialization
