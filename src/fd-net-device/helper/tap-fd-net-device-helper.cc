/*
 * Copyright (c) 2012 INRIA, 2012 University of Washington
 *
 * SPDX-License-Identifier: GPL-2.0-only
 */

#include "tap-fd-net-device-helper.h"

#include "encode-decode.h"

#include "ns3/abort.h"
#include "ns3/config.h"
#include "ns3/fd-net-device.h"
#include "ns3/log.h"
#include "ns3/names.h"
#include "ns3/object-factory.h"
#include "ns3/packet.h"
#include "ns3/simulator.h"
#include "ns3/trace-helper.h"

#include <cstdlib>
#include <errno.h>
#include <iomanip>
#include <iostream>
#include <string>

// ---- Platform-specific includes ----
#if defined(__linux__)
#include <arpa/inet.h>
#include <net/ethernet.h>
#include <net/if.h>
#include <netinet/in.h>
#include <netpacket/packet.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#elif defined(__APPLE__)
#include <arpa/inet.h>
#include <net/if.h>
#include <netinet/in.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#elif defined(_WIN32)
// winsock2.h must come before windows.h to avoid winsock.h re-inclusion.
// WIN32_LEAN_AND_MEAN prevents windows.h from pulling in winsock.h and
// avoids name collisions with macros such as GetObject -> GetObjectA.
#define WIN32_LEAN_AND_MEAN
// clang-format off
// winsock2.h must precede windows.h; alphabetical order breaks the build.
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <winioctl.h>
#include <winreg.h>
#include <fcntl.h>
#include <io.h>
// clang-format on
// Undo the GetObject -> GetObjectA macro that windows.h still defines via
// wingdi.h even with WIN32_LEAN_AND_MEAN, so ns-3's templated GetObject<>
// methods are reachable.
#ifdef GetObject
#undef GetObject
#endif

// tap-windows6 IOCTL codes (same as OpenVPN's tapctl)
#define TAP_WIN_IOCTL_SET_MEDIA_STATUS                                                             \
    CTL_CODE(FILE_DEVICE_UNKNOWN, 6, METHOD_BUFFERED, FILE_ANY_ACCESS)
#define TAP_WIN_IOCTL_CONFIG_TUN CTL_CODE(FILE_DEVICE_UNKNOWN, 10, METHOD_BUFFERED, FILE_ANY_ACCESS)

// Registry path for NDIS adapter classes
static const char* TAP_CLASS_KEY =
    "SYSTEM\\CurrentControlSet\\Control\\Class\\{4D36E972-E325-11CE-BFC1-08002BE10318}";
#endif // platform

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("TapFdNetDeviceHelper");

#define TAP_MAGIC 95549

TapFdNetDeviceHelper::TapFdNetDeviceHelper()
{
    m_deviceName = "";
    m_modePi = false;
    m_modeTap = true; // default to TAP (L2) mode
    m_tapIp4 = Ipv4Address::GetZero();
    m_tapMask4 = Ipv4Mask::GetZero();
    m_tapIp6 = Ipv6Address::GetZero();
    m_tapPrefix6 = 64;
    m_tapMac = Mac48Address::Allocate();
}

void
TapFdNetDeviceHelper::SetModePi(bool modePi)
{
    m_modePi = modePi;
}

void
TapFdNetDeviceHelper::SetModeTap(bool modeTap)
{
    m_modeTap = modeTap;
}

void
TapFdNetDeviceHelper::SetTapIpv4Address(Ipv4Address address)
{
    m_tapIp4 = address;
}

void
TapFdNetDeviceHelper::SetTapIpv4Mask(Ipv4Mask mask)
{
    m_tapMask4 = mask;
}

void
TapFdNetDeviceHelper::SetTapIpv6Address(Ipv6Address address)
{
    m_tapIp6 = address;
}

void
TapFdNetDeviceHelper::SetTapIpv6Prefix(int prefix)
{
    m_tapPrefix6 = prefix;
}

void
TapFdNetDeviceHelper::SetTapMacAddress(Mac48Address mac)
{
    m_tapMac = mac;
}

Ptr<NetDevice>
TapFdNetDeviceHelper::InstallPriv(Ptr<Node> node) const
{
    Ptr<NetDevice> d = FdNetDeviceHelper::InstallPriv(node);
    Ptr<FdNetDevice> device = d->GetObject<FdNetDevice>();

#if defined(__APPLE__)
    if (!m_modeTap)
    {
        // macOS utun delivers raw IP with a 4-byte address-family prefix
        device->SetEncapsulationMode(FdNetDevice::UTUN);
    }
    else if (m_modePi)
    {
        device->SetEncapsulationMode(FdNetDevice::DIXPI);
    }
#else
    if (m_modePi)
    {
        device->SetEncapsulationMode(FdNetDevice::DIXPI);
    }
#endif

    SetFileDescriptor(device);
    return device;
}

void
TapFdNetDeviceHelper::SetFileDescriptor(Ptr<FdNetDevice> device) const
{
    NS_LOG_LOGIC("Creating TAP/TUN device");
    int fd = CreateFileDescriptor();
    device->SetFileDescriptor(fd);
}

// ==========================================================================
// Platform implementations of CreateFileDescriptor()
// ==========================================================================

#if defined(__linux__)
// ---- Linux: fork/exec tap-device-creator (suid root) --------------------

int
TapFdNetDeviceHelper::CreateFileDescriptor() const
{
    NS_LOG_FUNCTION(this);

    int sock = socket(PF_UNIX, SOCK_DGRAM, 0);
    NS_ABORT_MSG_IF(
        sock == -1,
        "TapFdNetDeviceHelper::CreateFileDescriptor(): Unix socket creation error, errno = "
            << strerror(errno));

    struct sockaddr_un un;
    memset(&un, 0, sizeof(un));
    un.sun_family = AF_UNIX;
    int status = bind(sock, (struct sockaddr*)&un, sizeof(sa_family_t));
    NS_ABORT_MSG_IF(status == -1,
                    "TapFdNetDeviceHelper::CreateFileDescriptor(): Could not bind(): errno = "
                        << strerror(errno));
    NS_LOG_INFO("Created Unix socket");

    socklen_t len = sizeof(un);
    status = getsockname(sock, (struct sockaddr*)&un, &len);
    NS_ABORT_MSG_IF(
        status == -1,
        "TapFdNetDeviceHelper::CreateFileDescriptor(): Could not getsockname(): errno = "
            << strerror(errno));

    std::string path = BufferToString((uint8_t*)&un, len);
    NS_LOG_INFO("Encoded Unix socket as \"" << path << "\"");

    pid_t pid = ::fork();
    if (pid == 0)
    {
        NS_LOG_DEBUG("Child process");

        std::ostringstream ossDeviceName;
        if (!m_deviceName.empty())
        {
            ossDeviceName << "-d" << m_deviceName;
        }

        std::ostringstream ossMac;
        ossMac << "-m" << m_tapMac;

        std::ostringstream ossIp4;
        if (m_tapIp4 != Ipv4Address::GetZero())
        {
            ossIp4 << "-i" << m_tapIp4;
        }

        std::ostringstream ossIp6;
        if (m_tapIp6 != Ipv6Address::GetZero())
        {
            ossIp6 << "-I" << m_tapIp6;
        }

        std::ostringstream ossNetmask4;
        if (m_tapMask4 != Ipv4Mask::GetZero())
        {
            ossNetmask4 << "-n" << m_tapMask4;
        }

        std::ostringstream ossPrefix6;
        ossPrefix6 << "-P" << m_tapPrefix6;

        std::ostringstream ossMode;
        if (m_modeTap)
        {
            ossMode << "-t"; // IFF_TAP
        }
        // Without -t the creator creates IFF_TUN

        std::ostringstream ossPI;
        if (m_modePi)
        {
            ossPI << "-h";
        }

        std::ostringstream ossPath;
        ossPath << "-p" << path;

        status = ::execlp(TAP_DEV_CREATOR,
                          TAP_DEV_CREATOR,
                          ossDeviceName.str().c_str(),
                          ossMac.str().c_str(),
                          ossIp4.str().c_str(),
                          ossIp6.str().c_str(),
                          ossNetmask4.str().c_str(),
                          ossPrefix6.str().c_str(),
                          ossMode.str().c_str(),
                          ossPI.str().c_str(),
                          ossPath.str().c_str(),
                          (char*)nullptr);

        NS_FATAL_ERROR("TapFdNetDeviceHelper::CreateFileDescriptor(): Back from execlp(), status = "
                       << status << ", errno = " << ::strerror(errno));
    }
    else
    {
        NS_LOG_DEBUG("Parent process");

        int st;
        pid_t waited = waitpid(pid, &st, 0);
        NS_ABORT_MSG_IF(waited == -1,
                        "TapFdNetDeviceHelper::CreateFileDescriptor(): waitpid() fails, errno = "
                            << strerror(errno));
        NS_ASSERT_MSG(pid == waited, "TapFdNetDeviceHelper::CreateFileDescriptor(): pid mismatch");

        if (WIFEXITED(st))
        {
            int exitStatus = WEXITSTATUS(st);
            NS_ABORT_MSG_IF(exitStatus != 0,
                            "TapFdNetDeviceHelper::CreateFileDescriptor(): socket creator exited "
                            "normally with status "
                                << exitStatus);
        }
        else
        {
            NS_FATAL_ERROR(
                "TapFdNetDeviceHelper::CreateFileDescriptor(): socket creator exited abnormally");
        }

        struct iovec iov;
        uint32_t magic;
        iov.iov_base = &magic;
        iov.iov_len = sizeof(magic);

        constexpr size_t msg_size = sizeof(int);
        char control[CMSG_SPACE(msg_size)];

        struct msghdr msg;
        msg.msg_name = nullptr;
        msg.msg_namelen = 0;
        msg.msg_iov = &iov;
        msg.msg_iovlen = 1;
        msg.msg_control = control;
        msg.msg_controllen = sizeof(control);
        msg.msg_flags = 0;

        ssize_t bytesRead = recvmsg(sock, &msg, 0);
        NS_ABORT_MSG_IF(
            bytesRead != sizeof(int),
            "TapFdNetDeviceHelper::CreateFileDescriptor(): Wrong byte count from socket creator");

        struct cmsghdr* cmsg;
        for (cmsg = CMSG_FIRSTHDR(&msg); cmsg != nullptr; cmsg = CMSG_NXTHDR(&msg, cmsg))
        {
            if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS)
            {
                if (magic == TAP_MAGIC)
                {
                    NS_LOG_INFO("Got SCM_RIGHTS with correct magic " << magic);
                    int* rawSocket = (int*)CMSG_DATA(cmsg);
                    NS_LOG_INFO("Got the socket from the socket creator = " << *rawSocket);
                    return *rawSocket;
                }
                else
                {
                    NS_LOG_INFO("Got SCM_RIGHTS, but with bad magic " << magic);
                }
            }
        }
        NS_FATAL_ERROR("Did not get the raw socket from the socket creator");
    }
    NS_FATAL_ERROR("Should be unreachable");
    return 0;
}

// ==========================================================================
#elif defined(__APPLE__)
// ---- macOS: fork/exec macos-tap-tun-device-creator (suid root) ----------

int
TapFdNetDeviceHelper::CreateFileDescriptor() const
{
    NS_LOG_FUNCTION(this);

    int sock = socket(PF_UNIX, SOCK_DGRAM, 0);
    NS_ABORT_MSG_IF(
        sock == -1,
        "TapFdNetDeviceHelper::CreateFileDescriptor(): Unix socket creation error, errno = "
            << strerror(errno));

    struct sockaddr_un un;
    memset(&un, 0, sizeof(un));
    un.sun_family = AF_UNIX;
    int status = bind(sock, (struct sockaddr*)&un, sizeof(sa_family_t));
    NS_ABORT_MSG_IF(status == -1,
                    "TapFdNetDeviceHelper::CreateFileDescriptor(): Could not bind(): errno = "
                        << strerror(errno));

    socklen_t len = sizeof(un);
    status = getsockname(sock, (struct sockaddr*)&un, &len);
    NS_ABORT_MSG_IF(
        status == -1,
        "TapFdNetDeviceHelper::CreateFileDescriptor(): Could not getsockname(): errno = "
            << strerror(errno));

    std::string path = BufferToString((uint8_t*)&un, len);
    NS_LOG_INFO("Encoded Unix socket as \"" << path << "\"");

    pid_t pid = ::fork();
    if (pid == 0)
    {
        NS_LOG_DEBUG("Child process (macOS creator)");

        std::ostringstream ossDeviceName;
        if (!m_deviceName.empty())
        {
            ossDeviceName << "-d" << m_deviceName;
        }

        std::ostringstream ossIp4;
        if (m_tapIp4 != Ipv4Address::GetZero())
        {
            ossIp4 << "-i" << m_tapIp4;
        }

        std::ostringstream ossNetmask4;
        if (m_tapMask4 != Ipv4Mask::GetZero())
        {
            ossNetmask4 << "-n" << m_tapMask4;
        }

        // -t selects TAP (/dev/tapN); without -t, creates utun (TUN)
        std::ostringstream ossMode;
        if (m_modeTap)
        {
            ossMode << "-t";
        }

        std::ostringstream ossPath;
        ossPath << "-p" << path;

        status = ::execlp(MACOS_TAP_TUN_CREATOR,
                          MACOS_TAP_TUN_CREATOR,
                          ossDeviceName.str().c_str(),
                          ossIp4.str().c_str(),
                          ossNetmask4.str().c_str(),
                          ossMode.str().c_str(),
                          ossPath.str().c_str(),
                          (char*)nullptr);

        NS_FATAL_ERROR("TapFdNetDeviceHelper::CreateFileDescriptor(): Back from execlp(), status = "
                       << status << ", errno = " << ::strerror(errno));
    }
    else
    {
        NS_LOG_DEBUG("Parent process");

        int st;
        pid_t waited = waitpid(pid, &st, 0);
        NS_ABORT_MSG_IF(waited == -1,
                        "TapFdNetDeviceHelper::CreateFileDescriptor(): waitpid() fails, errno = "
                            << strerror(errno));
        NS_ASSERT_MSG(pid == waited, "TapFdNetDeviceHelper::CreateFileDescriptor(): pid mismatch");

        if (WIFEXITED(st))
        {
            int exitStatus = WEXITSTATUS(st);
            NS_ABORT_MSG_IF(exitStatus != 0,
                            "TapFdNetDeviceHelper::CreateFileDescriptor(): macOS creator exited "
                            "with status "
                                << exitStatus);
        }
        else
        {
            NS_FATAL_ERROR("TapFdNetDeviceHelper::CreateFileDescriptor(): macOS creator exited "
                           "abnormally");
        }

        struct iovec iov;
        uint32_t magic;
        iov.iov_base = &magic;
        iov.iov_len = sizeof(magic);

        constexpr size_t msg_size = sizeof(int);
        char control[CMSG_SPACE(msg_size)];

        struct msghdr msg;
        msg.msg_name = nullptr;
        msg.msg_namelen = 0;
        msg.msg_iov = &iov;
        msg.msg_iovlen = 1;
        msg.msg_control = control;
        msg.msg_controllen = sizeof(control);
        msg.msg_flags = 0;

        ssize_t bytesRead = recvmsg(sock, &msg, 0);
        NS_ABORT_MSG_IF(
            bytesRead != sizeof(int),
            "TapFdNetDeviceHelper::CreateFileDescriptor(): Wrong byte count from macOS creator");

        struct cmsghdr* cmsg;
        for (cmsg = CMSG_FIRSTHDR(&msg); cmsg != nullptr; cmsg = CMSG_NXTHDR(&msg, cmsg))
        {
            if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS)
            {
                if (magic == TAP_MAGIC)
                {
                    NS_LOG_INFO("Got SCM_RIGHTS with correct magic " << magic);
                    int* rawSocket = (int*)CMSG_DATA(cmsg);
                    return *rawSocket;
                }
            }
        }
        NS_FATAL_ERROR("Did not get the fd from the macOS creator");
    }
    NS_FATAL_ERROR("Should be unreachable");
    return 0;
}

// ==========================================================================
#elif defined(_WIN32)
// ---- Windows: open tap-windows6 adapter directly (requires admin) --------

/**
 * Search HKLM registry for a tap-windows6 (OpenVPN) network adapter and
 * return its NetCfgInstanceId GUID string.  Returns an empty string if no
 * such adapter is found.
 */
static std::string
FindTapWindowsGuid(const std::string& preferredName)
{
    HKEY classKey;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, TAP_CLASS_KEY, 0, KEY_READ, &classKey) != ERROR_SUCCESS)
    {
        return {};
    }

    char subKeyName[64];
    DWORD index = 0;

    while (RegEnumKeyA(classKey, index++, subKeyName, sizeof(subKeyName)) == ERROR_SUCCESS)
    {
        HKEY adapterKey;
        if (RegOpenKeyExA(classKey, subKeyName, 0, KEY_READ, &adapterKey) != ERROR_SUCCESS)
        {
            continue;
        }

        // Check ComponentId to identify tap-windows6
        char componentId[64] = {};
        DWORD len = sizeof(componentId);
        DWORD type = 0;
        LONG res =
            RegQueryValueExA(adapterKey, "ComponentId", nullptr, &type, (LPBYTE)componentId, &len);

        bool isTapWin = (res == ERROR_SUCCESS && (strncmp(componentId, "tap0901", 7) == 0 ||
                                                  strncmp(componentId, "tap0902", 7) == 0));

        if (isTapWin)
        {
            // Optionally filter by adapter name
            if (!preferredName.empty())
            {
                char netCfgName[256] = {};
                DWORD nameLen = sizeof(netCfgName);
                RegQueryValueExA(adapterKey,
                                 "NetCfgInstanceId",
                                 nullptr,
                                 nullptr,
                                 (LPBYTE)netCfgName,
                                 &nameLen);
                // Check the adapter's human-readable name in
                // HKLM\SYSTEM\...\Control\Network\{4D36...}\{GUID}\Connection
                // For simplicity we skip name filtering here and take the first match.
            }

            char guid[256] = {};
            DWORD guidLen = sizeof(guid);
            res = RegQueryValueExA(adapterKey,
                                   "NetCfgInstanceId",
                                   nullptr,
                                   nullptr,
                                   (LPBYTE)guid,
                                   &guidLen);
            RegCloseKey(adapterKey);
            RegCloseKey(classKey);

            if (res == ERROR_SUCCESS)
            {
                return std::string(guid);
            }
            return {};
        }

        RegCloseKey(adapterKey);
    }

    RegCloseKey(classKey);
    return {};
}

int
TapFdNetDeviceHelper::CreateFileDescriptor() const
{
    NS_LOG_FUNCTION(this);

    std::string guid = FindTapWindowsGuid(m_deviceName);
    NS_ABORT_MSG_IF(guid.empty(),
                    "TapFdNetDeviceHelper::CreateFileDescriptor(): No tap-windows6 adapter found. "
                    "Install the OpenVPN TAP driver (tap-windows6) and try again.");

    std::string tapPath = std::string("\\\\.\\Global\\") + guid + ".tap";
    NS_LOG_INFO("Opening TAP device: " << tapPath);

    HANDLE hTap = CreateFileA(tapPath.c_str(),
                              GENERIC_READ | GENERIC_WRITE,
                              0,
                              nullptr,
                              OPEN_EXISTING,
                              FILE_ATTRIBUTE_SYSTEM,
                              nullptr);
    NS_ABORT_MSG_IF(hTap == INVALID_HANDLE_VALUE,
                    "TapFdNetDeviceHelper::CreateFileDescriptor(): Could not open TAP device \""
                        << tapPath << "\", error=" << GetLastError()
                        << ". Is the simulation running as Administrator?");

    // Bring the virtual link up
    ULONG mediaStatus = 1;
    DWORD returned = 0;
    BOOL ok = DeviceIoControl(hTap,
                              TAP_WIN_IOCTL_SET_MEDIA_STATUS,
                              &mediaStatus,
                              sizeof(mediaStatus),
                              &mediaStatus,
                              sizeof(mediaStatus),
                              &returned,
                              nullptr);
    NS_ABORT_MSG_IF(!ok,
                    "TapFdNetDeviceHelper::CreateFileDescriptor(): "
                    "TAP_WIN_IOCTL_SET_MEDIA_STATUS failed, error="
                        << GetLastError());

    // For TUN (L3) mode, configure the point-to-point IP routing
    if (!m_modeTap && m_tapIp4 != Ipv4Address::GetZero() && m_tapMask4 != Ipv4Mask::GetZero())
    {
        // ip[0] = local addr, ip[1] = remote (gateway) addr = same as local for P2P,
        // ip[2] = netmask
        ULONG ip[3];
        m_tapIp4.Serialize(reinterpret_cast<uint8_t*>(&ip[0]));
        m_tapIp4.Serialize(reinterpret_cast<uint8_t*>(&ip[1])); // P2P remote = local
        // Ipv4Mask has no Serialize(); construct a temporary Ipv4Address from
        // the raw 32-bit mask value (same internal byte order) and use that.
        Ipv4Address(m_tapMask4.Get()).Serialize(reinterpret_cast<uint8_t*>(&ip[2]));

        DeviceIoControl(hTap,
                        TAP_WIN_IOCTL_CONFIG_TUN,
                        ip,
                        sizeof(ip),
                        ip,
                        sizeof(ip),
                        &returned,
                        nullptr);
    }

    // Convert the Windows HANDLE to a POSIX-compatible integer fd.
    // Use _O_BINARY | _O_RDWR so read()/write() treat it as a binary stream.
    int fd = _open_osfhandle(reinterpret_cast<intptr_t>(hTap), _O_RDWR | _O_BINARY);
    NS_ABORT_MSG_IF(fd == -1,
                    "TapFdNetDeviceHelper::CreateFileDescriptor(): _open_osfhandle() failed");

    NS_LOG_INFO("Opened TAP device, fd=" << fd);
    return fd;
}

// ==========================================================================
#else
// ---- Unsupported platform ------------------------------------------------
int
TapFdNetDeviceHelper::CreateFileDescriptor() const
{
    NS_FATAL_ERROR("TapFdNetDeviceHelper: TAP/TUN device creation is not supported on this "
                   "platform. Supported platforms: Linux, macOS, Windows.");
    return -1;
}
#endif // platform

} // namespace ns3
