/*
 * Copyright (c) 2026 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Privileged helper process for creating utun (TUN) interfaces on macOS.
 *
 * Uses the kernel's SYSPROTO_CONTROL socket API — no third-party driver
 * required.  Needs root or the
 * com.apple.developer.networking.networkextension entitlement.
 *
 * The fd is sent back to the parent ns-3 process via a Unix-domain socket
 * using SCM_RIGHTS ancillary data, matching the pattern used by the Linux
 * tap-device-creator.
 *
 * Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
 */

#include "creator-utils.h"

#include "ns3/mac48-address.h"

#include <arpa/inet.h>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <net/if.h>
#include <netinet/in.h>
#include <sys/ioctl.h>
#include <sys/kern_control.h>
#include <sys/socket.h>
#include <sys/sys_domain.h>
#include <sys/un.h>
#include <unistd.h>

#define TAP_MAGIC 95549
#define UTUN_CONTROL_NAME "com.apple.net.utun_control"
#define UTUN_OPT_IFNAME 2

using namespace ns3;

/**
 * Configure an IPv4 address and netmask on the given interface.
 * utun interfaces are point-to-point, so the destination address is set to
 * the same value as the local address.
 */
static void
SetIpv4(const char* deviceName, const char* ip, const char* netmask)
{
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    ABORT_IF(sock < 0, "Could not open DGRAM socket for IPv4 config", true);

    struct ifreq ifr;

    // Set local address
    memset(&ifr, 0, sizeof(ifr));
    strncpy(ifr.ifr_name, deviceName, IFNAMSIZ - 1);
    auto* sin = reinterpret_cast<struct sockaddr_in*>(&ifr.ifr_addr);
    sin->sin_family = AF_INET;
    inet_pton(AF_INET, ip, &sin->sin_addr);
    ABORT_IF(ioctl(sock, SIOCSIFADDR, &ifr) == -1, "Could not set IPv4 address", true);
    LOG("Set device IPv4 address to " << ip);

    // Set destination address (the other end of the point-to-point link)
    memset(&ifr, 0, sizeof(ifr));
    strncpy(ifr.ifr_name, deviceName, IFNAMSIZ - 1);
    sin = reinterpret_cast<struct sockaddr_in*>(&ifr.ifr_dstaddr);
    sin->sin_family = AF_INET;
    inet_pton(AF_INET, ip, &sin->sin_addr);
    // Ignore errors — some macOS versions don't require this
    ioctl(sock, SIOCSIFDSTADDR, &ifr);

    // Set netmask
    memset(&ifr, 0, sizeof(ifr));
    strncpy(ifr.ifr_name, deviceName, IFNAMSIZ - 1);
    sin = reinterpret_cast<struct sockaddr_in*>(&ifr.ifr_addr);
    sin->sin_family = AF_INET;
    inet_pton(AF_INET, netmask, &sin->sin_addr);
    ABORT_IF(ioctl(sock, SIOCSIFNETMASK, &ifr) == -1, "Could not set IPv4 netmask", true);
    LOG("Set device IPv4 netmask to " << netmask);

    close(sock);
}

/** Bring the interface up with IFF_UP | IFF_RUNNING. */
static void
SetUp(const char* deviceName)
{
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    ABORT_IF(sock < 0, "Could not open DGRAM socket to bring interface up", true);

    struct ifreq ifr;
    memset(&ifr, 0, sizeof(ifr));
    strncpy(ifr.ifr_name, deviceName, IFNAMSIZ - 1);
    ABORT_IF(ioctl(sock, SIOCGIFFLAGS, &ifr) == -1, "Could not get interface flags", true);
    ifr.ifr_flags |= IFF_UP | IFF_RUNNING;
    ABORT_IF(ioctl(sock, SIOCSIFFLAGS, &ifr) == -1, "Could not bring interface up", true);
    LOG("Interface " << deviceName << " is up");

    close(sock);
}

/**
 * Create a utun (user-space TUN) interface on macOS using the kernel control
 * socket API.  Does not require a third-party driver.
 *
 * @param unit  Desired utun unit number (e.g. 0 for utun0). Use -1 for
 *              automatic assignment.
 * @param[out] assignedName  The interface name assigned by the kernel.
 * @return fd on success, aborts on failure.
 */
static int
CreateUtun(int unit, char* assignedName)
{
    int fd = socket(PF_SYSTEM, SOCK_DGRAM, SYSPROTO_CONTROL);
    ABORT_IF(fd < 0, "Could not create SYSPROTO_CONTROL socket for utun", true);

    struct ctl_info ci;
    memset(&ci, 0, sizeof(ci));
    strlcpy(ci.ctl_name, UTUN_CONTROL_NAME, sizeof(ci.ctl_name));
    ABORT_IF(ioctl(fd, CTLIOCGINFO, &ci) < 0, "CTLIOCGINFO for utun failed", true);

    struct sockaddr_ctl sc;
    memset(&sc, 0, sizeof(sc));
    sc.sc_id = ci.ctl_id;
    sc.sc_len = sizeof(sc);
    sc.sc_family = AF_SYSTEM;
    sc.ss_sysaddr = AF_SYS_CONTROL;
    // sc_unit 0 = auto-assign; otherwise unit+1 selects utunN
    sc.sc_unit = (unit < 0) ? 0 : static_cast<uint32_t>(unit + 1);

    ABORT_IF(connect(fd, reinterpret_cast<struct sockaddr*>(&sc), sizeof(sc)) < 0,
             "connect() to utun control failed",
             true);

    // Retrieve the kernel-assigned interface name (e.g. "utun0")
    socklen_t nameLen = IFNAMSIZ;
    ABORT_IF(getsockopt(fd, SYSPROTO_CONTROL, UTUN_OPT_IFNAME, assignedName, &nameLen) < 0,
             "getsockopt(UTUN_OPT_IFNAME) failed",
             true);

    LOG("Created utun interface: " << assignedName);
    return fd;
}

int
main(int argc, char* argv[])
{
    int c;
    char* dev = nullptr;
    char* ip4 = nullptr;
    char* netmask = nullptr;
    char* path = nullptr;
    int utunUnit = -1; // -1 = auto-assign

    while ((c = getopt(argc, argv, "vd:i:n:p:u:")) != -1)
    {
        switch (c)
        {
        case 'd':
            dev = optarg;
            break;
        case 'i':
            ip4 = optarg;
            break;
        case 'n':
            netmask = optarg;
            break;
        case 'p':
            path = optarg;
            break;
        case 'u':
            utunUnit = atoi(optarg);
            break;
        case 'v':
            gVerbose = true;
            break;
        default:
            break;
        }
    }

    ABORT_IF(path == nullptr, "path is a required argument", 0);

    (void)dev; // utun names are kernel-assigned; caller hint is ignored

    char assignedName[IFNAMSIZ] = {};
    int fd = CreateUtun(utunUnit, assignedName);
    ABORT_IF(fd < 0, "Could not create utun device", true);

    LOG("Opened interface: " << assignedName);

    if (ip4 && netmask)
    {
        SetIpv4(assignedName, ip4, netmask);
    }
    else if (ip4 || netmask)
    {
        ABORT("Both -i <IPv4> and -n <netmask> must be provided together", 0);
    }

    SetUp(assignedName);

    SendSocket(path, fd, TAP_MAGIC);
    return 0;
}
