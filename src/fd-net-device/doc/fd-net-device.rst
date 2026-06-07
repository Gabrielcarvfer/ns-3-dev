File Descriptor NetDevice
-------------------------
.. include:: replace.txt
.. highlight:: cpp

.. heading hierarchy:
   ------------- Chapter
   ************* Section (#.#)
   ============= Subsection (#.#.#)
   ############# Paragraph (no number)

The ``src/fd-net-device`` module provides the ``FdNetDevice`` class,
which is able to read and write traffic using a file descriptor
provided by the user.  This file descriptor can be associated to a TAP
device, to a raw socket, to a user space process generating/consuming
traffic, etc.
The user has full freedom to define how external traffic is generated
and |ns3| traffic is consumed.

Different mechanisms to associate a simulation to external traffic can
be provided through helper classes.  Two specific helpers are provided:

* EmuFdNetDeviceHelper (to associate the |ns3| device with a physical device
  in the host machine)
* TapFdNetDeviceHelper (to associate the ns-3 device with the file descriptor
  from a tap device in the host machine)

Model Description
*****************

The source code for this module lives in the directory ``src/fd-net-device``.

The FdNetDevice is a special type of |ns3| NetDevice that reads traffic
to and from a file descriptor.  That is, unlike pure simulation NetDevice
objects that write frames to and from a simulated channel, this FdNetDevice
directs frames out of the simulation to a file descriptor.  The file
descriptor may be associated to a Linux TUN/TAP device, to a socket, or
to a user-space process.

It is up to the user of this device to provide a file descriptor.  The
type of file descriptor being provided determines what is being
modelled.  For instance, if the file descriptor provides a raw socket
to a WiFi card on the host machine, the device being modelled is a
WiFi device.

From the conceptual "top" of the device looking down, it looks to the
simulated node like a device supporting a 48-bit IEEE MAC address that
can be bridged, supports broadcast, and uses IPv4 ARP or IPv6 Neighbor
Discovery, although these attributes can be tuned on a per-use-case basis.

Design
======

The FdNetDevice implementation makes use of a reader object,
extended from the ``FdReader`` class in the |ns3| ``src/core`` module,
which manages a separate thread from the main |ns3| execution thread, in order
to read traffic from the file descriptor.

Upon invocation of the ``StartDevice`` method, the reader object is initialized
and starts the reading thread.
Before device start, a file descriptor must be previously associated to the
FdNetDevice with the ``SetFileDescriptor`` invocation.

The creation and configuration of the file descriptor can be left to a
number of helpers, described in more detail below. When this is done, the
invocation of ``SetFileDescriptor`` is responsibility of
the helper and must not be directly invoked by the user.

Upon reading an incoming frame from the file descriptor, the reader
will pass the frame to the ``ReceiveCallback`` method, whose
task it is to schedule the reception of the frame by the device as a
|ns3| simulation event. Since the new frame is passed from the reader
thread to the main |ns3| simulation thread, thread-safety issues
are avoided by using the ``ScheduleWithContext`` call instead of the
regular ``Schedule`` call.

In order to avoid overwhelming the scheduler when the incoming data rate
is too high, a counter is kept with the number of frames that are currently
scheduled to be received by the device. If this counter reaches the value
given by the ``RxQueueSize`` attribute in the device, then the new frame will
be dropped silently.

The actual reception of the new frame by the device occurs when the
scheduled ``FordwarUp`` method is invoked by the simulator.
This method acts as if a new frame had arrived from a channel attached
to the device. The device then decapsulates the frame, removing any layer 2
headers, and forwards it to upper network stack layers of the node.
The ``ForwardUp`` method will remove the frame headers,
according to the frame encapsulation type defined by the ``EncapsulationMode``
attribute, and invoke the receive callback passing an IP packet.

An extra header, the PI header, can be present when the file descriptor is
associated to a TAP device that was created without setting the IFF_NO_PI flag.
This extra header is removed if ``EncapsulationMode`` is set to DIXPI value.

In the opposite direction, packets generated inside the simulation that are
sent out through the device, will be passed to the ``Send`` method, which
will in turn invoke the ``SendFrom`` method. The latter method will add the
necessary layer 2 headers, and simply write the newly created frame to the
file descriptor.


Scope and Limitations
=====================

Users of this device are cautioned that there is no flow control
across the file descriptor boundary, when using in emulation mode.
That is, in a Linux system, if the speed of writing network packets
exceeds the ability of the underlying physical device to buffer the
packets, backpressure up to the writing application will be applied
to avoid local packet loss.  No such flow control is provided across
the file descriptor interface, so users must be aware of this limitation.

As explained before, the RxQueueSize attribute limits the number of packets
that can be pending to be received by the device.
Frames read from the file descriptor while the number of pending packets is
in its maximum will be silently dropped.

The mtu of the device defaults to the Ethernet II MTU value. However, helpers
are supposed to set the mtu to the right value to reflect the characteristics
of the network interface associated to the file descriptor.
If no helper is used, then the responsibility of setting the correct mtu value
for the device falls back to the user.
The size of the read buffer on the file descriptor reader is set to the
mtu value in the ``StartDevice`` method.

The FdNetDevice class currently supports three encapsulation modes,
DIX for Ethernet II frames, LLC for 802.2 LLC/SNAP frames,
and DIXPI for Ethernet II frames with an additional TAP PI header.
This means that traffic traversing the file descriptor is expected to be
Ethernet II compatible.  IEEE 802.1q (VLAN) tagging is not supported.
Attaching an FdNetDevice to a wireless interface is possible as long as the
driver provides Ethernet II frames to the socket API.
Note that to associate a FdNetDevice to a wireless card in ad-hoc mode,
the MAC address of the device must be set to the real card MAC address, else
any incoming traffic a fake MAC address will be discarded by the driver.

As mentioned before, three helpers are provided with the fd-net-device module.
Each individual helper (file descriptor type) may have platform
limitations.  For instance, threading, real-time simulation mode, and the
ability to create TUN/TAP devices are prerequisites to using the
provided helpers.  Support for these modes can be found in the output
of the ``ns3 configure`` step, e.g.:

.. sourcecode:: text

   Threading Primitives          : enabled
   Real Time Simulator           : enabled
   Emulated Net Device           : enabled
   Tap Bridge                    : enabled


It is important to mention that while testing the ``FdNetDevice`` we have found
an upper bound limit for TCP throughput when using 1Gb Ethernet links of 60Mbps.
This limit is most likely due to the processing power of the computers involved
in the tests.


Usage
*****

The usage pattern for this type of device is similar to other net devices
with helpers that install to node pointers or node containers.
When using the base ``FdNetDeviceHelper`` the user is responsible for
creating and setting the file descriptor by himself.

::

   FdNetDeviceHelper fd;
   NetDeviceContainer devices = fd.Install(nodes);

   // file descriptor generation
   ...

   device->SetFileDescriptor(fd);


Most commonly a FdNetDevice will be used to interact with the host system.
In these cases it is almost certain that the user will want to run in real-time
emulation mode, and to enable checksum computations.
The typical program statements are as follows:

::

   GlobalValue::Bind("SimulatorImplementationType", StringValue("ns3::RealtimeSimulatorImpl"));
   GlobalValue::Bind("ChecksumEnabled", BooleanValue(true));

The easiest way to set up an experiment that interacts with a Linux host
system is to user the ``Emu`` and ``Tap`` helpers.
Perhaps the most unusual part of these helper implementations
relates to the requirement for executing some of the code with super-user
permissions. Rather than force the user to execute the entire simulation as
root, we provide a small "creator" program that runs as root and does any
required high-permission sockets work. The easiest way to set the right
privileges for the "creator" programs, is by enabling the ``--enable-sudo``
flag when performing ``ns3 configure``.

We do a similar thing for both the ``Emu`` and the ``Tap`` devices.  The
high-level view is that the ``CreateFileDescriptor`` method creates a local interprocess
(Unix) socket, forks, and executes the small creation program. The small
program, which runs as suid root, creates a raw socket and sends back the raw
socket file descriptor over the Unix socket that is passed to it as a parameter.
The raw socket is passed as a control message (sometimes called ancillary data)
of type SCM_RIGHTS.


Helpers
=======

EmuFdNetDeviceHelper
####################

The EmuFdNetDeviceHelper creates a raw socket to an underlying physical
device, and provides the socket descriptor to the FdNetDevice.  This
allows the |ns3| simulation to read frames from and write frames to
a network device on the host.

The emulation helper permits to transparently integrate a simulated
|ns3| node into a network composed of real nodes.

.. sourcecode:: text

 +----------------------+     +-----------------------+
 |         host 1       |     |         host 2        |
 +----------------------+     +-----------------------+
 |    ns-3 simulation   |     |                       |
 +----------------------+     |         Linux         |
 |       ns-3 Node      |     |     Network Stack     |
 |  +----------------+  |     |   +----------------+  |
 |  |    ns-3 TCP    |  |     |   |       TCP      |  |
 |  +----------------+  |     |   +----------------+  |
 |  |    ns-3 IP     |  |     |   |       IP       |  |
 |  +----------------+  |     |   +----------------+  |
 |  |   FdNetDevice  |  |     |   |                |  |
 |  |    10.1.1.1    |  |     |   |                |  |
 |  +----------------+  |     |   +    ETHERNET    +  |
 |  |   raw socket   |  |     |   |                |  |
 |--+----------------+--|     |   +----------------+  |
 |       | eth0 |       |     |        | eth0 |       |
 +-------+------+-------+     +--------+------+-------+

         10.1.1.11                     10.1.1.12

             |                            |
             +----------------------------+


This helper replaces the functionality of the ``EmuNetDevice`` found in
|ns3| prior to ns-3.17, by bringing this type of device into the common
framework of the FdNetDevice.  The ``EmuNetDevice`` was deprecated
in favor of this new helper.

The device is configured to perform
MAC spoofing to separate simulation network traffic from other
network traffic that may be flowing to and from the host.

One can use this helper in a testbed situation where the host on
which the simulation is running has a specific interface of interest which
drives the testbed hardware. You would also need to set this specific interface
into promiscuous mode and provide an appropriate device name to the |ns3|
simulation.  Additionally, hardware offloading of segmentation and checksums
should be disabled.

The helper only works if the underlying interface is up and in
promiscuous mode. Packets will be sent out over the device, but we use MAC
spoofing. The MAC addresses will be generated (by default) using the
Organizationally Unique Identifier (OUI) 00:00:00 as a base. This vendor code
is not assigned to any organization and so should not conflict with any real
hardware.

It is always up to the user to determine that using these MAC addresses is okay
on your network and won't conflict with anything else (including another
simulation using such devices) on your network. If you are using the emulated
FdNetDevice configuration in separate simulations,
you must consider global MAC address
assignment issues and ensure that MAC addresses are unique across all
simulations. The emulated net device respects the MAC address provided in the
``Address`` attribute so you can do this manually. For larger simulations, you
may want to set the OUI in the MAC address allocation function.

Before invoking the ``Install`` method, the correct device name must be configured
on the helper using the ``SetDeviceName`` method. The device name is required to
identify which physical device should be used to open the raw socket.

::

  EmuFdNetDeviceHelper emu;
  emu.SetDeviceName(deviceName);
  NetDeviceContainer devices = emu.Install(node);
  Ptr<NetDevice> device = devices.Get(0);
  device->SetAttribute("Address", Mac48AddressValue(Mac48Address::Allocate()));


TapFdNetDeviceHelper
####################

A Tap device is a special type of Linux device for which one end of the
device appears to the kernel as a virtual net_device, and the other
end is provided as a file descriptor to user-space.  This file descriptor
can be passed to the FdNetDevice.  Packets forwarded to the TAP device
by the kernel will show up in the FdNetDevice in |ns3|.

Users should note that this usage of TAP devices is different than that
provided by the TapBridge NetDevice found in ``src/tap-bridge``.
The model in this helper is as follows:

.. sourcecode:: text

 +-------------------------------------+
 |                host                 |
 +-------------------------------------+
 |    ns-3 simulation   |              |
 +----------------------+              |
 |      ns-3 Node       |              |
 |  +----------------+  |              |
 |  |    ns-3 TCP    |  |              |
 |  +----------------+  |              |
 |  |    ns-3 IP     |  |              |
 |  +----------------+  |              |
 |  |   FdNetDevice  |  |              |
 |--+----------------+--+    +------+  |
 |       | TAP  |            | eth0 |  |
 |       +------+            +------+  |
 |     192.168.0.1               |     |
 +-------------------------------|-----+
                                 |
                                 |
                                 ------------ (Internet) -----


In the above, the configuration requires that the host be able to forward
traffic generated by the simulation to the Internet.

The model in TapBridge (in another module) is as follows:

.. sourcecode:: text

    +--------+
    |  Linux |
    |  host  |                    +----------+
    | ------ |                    |   ghost  |
    |  apps  |                    |   node   |
    | ------ |                    | -------- |
    |  stack |                    |    IP    |     +----------+
    | ------ |                    |   stack  |     |   node   |
    |  TAP   |                    |==========|     | -------- |
    | device | <----- IPC ------> |   tap    |     |    IP    |
    +--------+                    |  bridge  |     |   stack  |
                                  | -------- |     | -------- |
                                  |   ns-3   |     |   ns-3   |
                                  |   net    |     |   net    |
                                  |  device  |     |  device  |
                                  +----------+     +----------+
                                       ||               ||
                                  +---------------------------+
                                  |        ns-3 channel       |
                                  +---------------------------+


In the above, packets instead traverse |ns3| NetDevices and Channels.

The usage pattern for this example is that the user sets the
MAC address and either (or both) the IPv4 and IPv6 addresses and masks
on the device, and the PI header if needed.  For example:

::

  TapFdNetDeviceHelper helper;
  helper.SetDeviceName(deviceName);
  helper.SetModePi(modePi);
  helper.SetTapIpv4Address(tapIp);
  helper.SetTapIpv4Mask(tapMask);
  ...
  helper.Install(node);


Attributes
==========

The ``FdNetDevice`` provides a number of attributes:

* ``Address``:  The MAC address of the device
* ``Start``:  The simulation start time to spin up the device thread
* ``Stop``:  The simulation start time to stop the device thread
* ``EncapsulationMode``:  Link-layer encapsulation format
* ``RxQueueSize``:  The buffer size of the read queue on the file descriptor
    thread (default of 1000 packets)

``Start`` and ``Stop`` do not normally need to be specified unless the
user wants to limit the time during which this device is active.
``Address`` needs to be set to some kind of unique MAC address if the
simulation will be interacting with other real devices somehow using
real MAC addresses.  Typical code:

::

   device->SetAttribute("Address", Mac48AddressValue(Mac48Address::Allocate()));

Output
======

Ascii and PCAP tracing is provided similar to the other |ns3| NetDevice
types, through the helpers, such as (e.g.):

::
    EmuFdNetDeviceHelper emu;
    NetDeviceContainer devices = emu.Install(node);
    ...
    emu.EnablePcap("emu-ping", device, true);


The standard set of Mac-level NetDevice trace sources is provided.

* ``MacTx``:  Trace source triggered when |ns3| provides the device with a
  new frame to send
* ``MacTxDrop``:  Trace source indicating a packet has been dropped by the
  device before transmission
* ``MacPromiscRx``:  Whenever any valid Mac frame is received
* ``MacRx``:  Whenever a valid Mac frame is received for this device
* ``Sniffer``:  Non-promiscuous packet sniffer
* ``PromiscSniffer``:  Promiscuous packet sniffer (for tcpdump-like traces)

Examples
========

Several examples are provided:

* ``dummy-network.cc``:  This simple example creates two nodes and
  interconnects them with a Unix pipe by passing the file descriptors
  from the socketpair into the FdNetDevice objects of the respective nodes.
* ``realtime-dummy-network.cc``:  Same as dummy-network.cc but uses the real time
  simulator implementnation instead of the default one.
* ``fd2fd-onoff.cc``: This example is aimed at measuring the throughput of the
  FdNetDevice in a pure simulation. For this purpose two FdNetDevices, attached to
  different nodes but in a same simulation, are connected using a socket pair.
  TCP traffic is sent at a saturating data rate.
* ``fd-emu-onoff.cc``: This example is aimed at measuring the throughput of the
  FdNetDevice  when using the EmuFdNetDeviceHelper to attach the simulated
  device to a real device in the host machine. This is achieved by saturating
  the channel with TCP traffic.
* ``fd-emu-ping.cc``: This example uses the EmuFdNetDeviceHelper to send ICMP
  traffic over a real channel.
* ``fd-emu-udp-echo.cc``: This example uses the EmuFdNetDeviceHelper to send UDP
  traffic over a real channel.
* ``fd-tap-ping.cc``: This example uses the TapFdNetDeviceHelper to send ICMP
  traffic over a real channel.

Platform-specific TAP/TUN Implementations
==========================================

This section explains in detail how each supported platform obtains a file
descriptor that represents a virtual network interface.  The explanation is
aimed at readers who are new to the relevant kernel or OS APIs.

Linux
#####

On Linux the kernel exposes TAP (Layer 2, Ethernet frames) and TUN (Layer 3,
raw IP datagrams) interfaces through the character device ``/dev/net/tun``.

Because creating these interfaces requires the ``CAP_NET_ADMIN`` capability
(or root privileges), ns-3 delegates the work to a small privileged helper
program, ``tap-device-creator``, that can be installed setuid-root.  The
helper communicates the file descriptor back to the ns-3 process using a
Unix-domain socket and the ``SCM_RIGHTS`` ancillary-data mechanism (which
lets one process hand an open file descriptor to another over a socket).

The sequence in ``tap-device-creator.cc`` is:

#. ``open("/dev/net/tun", O_RDWR)`` — open the kernel TUN/TAP multiplexer.
#. ``ioctl(fd, TUNSETIFF, &ifr)`` — configure the interface name and flags.
   ``IFF_TAP`` selects Layer 2 (Ethernet); ``IFF_TUN`` selects Layer 3 (IP).
   ``IFF_NO_PI`` suppresses the 4-byte Packet Information (PI) header that
   the kernel otherwise prepends to every read.  Without ``IFF_NO_PI`` each
   frame is prefixed with ``flags(2) + proto(2)``; the ``DIXPI`` encapsulation
   mode tells FdNetDevice to strip that header.
#. ``ioctl(sock, SIOCSIFADDR, …)`` / ``SIOCSIFNETMASK`` — set the IP address
   and netmask on the new interface via a datagram socket on ``AF_INET``.
#. ``ioctl(sock, SIOCSIFFLAGS, … | IFF_UP | IFF_RUNNING)`` — bring the
   interface up so the kernel starts routing packets to it.
#. ``sendmsg(unix_sock, …, SCM_RIGHTS)`` — send the TAP/TUN fd back to the
   ns-3 parent over the Unix-domain socket.

The parent (``TapFdNetDeviceHelper::CreateFileDescriptor()``) forks, exec's
the creator with the encoded socket path as an argument, waits for it to
exit, then calls ``recvmsg()`` with a ``CMSG_SPACE(sizeof(int))`` control
buffer.  On receipt the control message type is ``SCM_RIGHTS`` and
``CMSG_DATA`` contains the raw fd integer.

macOS
#####

macOS supports only TUN (Layer 3) via the built-in ``utun`` interface.  TAP
(Layer 2) required *tuntaposx*, which was abandoned in 2015 and cannot load
on macOS Catalina (10.15) or any later release.

utun interface (Layer 3)
^^^^^^^^^^^^^^^^^^^^^^^^^

The macOS path uses the built-in ``utun``
driver, which does **not** require a third-party kernel extension.  The API
uses the kernel control socket framework:

#. ``socket(PF_SYSTEM, SOCK_DGRAM, SYSPROTO_CONTROL)`` — create a kernel
   control socket.  ``PF_SYSTEM`` is macOS's namespace for kernel-internal
   communication; ``SYSPROTO_CONTROL`` selects the control plane sub-protocol.
#. ``ioctl(fd, CTLIOCGINFO, &ci)`` — look up the numeric ID of the
   ``com.apple.net.utun_control`` kernel control.  The kernel assigns IDs
   dynamically at boot, so this ioctl translates the well-known name string
   into the current numeric ``ctl_id``.
#. Fill in a ``sockaddr_ctl`` structure:

   .. code-block:: c

      struct sockaddr_ctl sc = {};
      sc.sc_family  = AF_SYSTEM;
      sc.ss_sysaddr = AF_SYS_CONTROL;
      sc.sc_id      = ci.ctl_id;
      sc.sc_unit    = unit + 1;  // 0 = auto-assign

#. ``connect(fd, (struct sockaddr *)&sc, sizeof(sc))`` — "connect" the
   control socket to the kernel's utun control.  Despite the name, this is
   not a TCP-style connection; it tells the kernel to create the utun
   interface and bind this fd to it.
#. ``getsockopt(fd, SYSPROTO_CONTROL, UTUN_OPT_IFNAME, …)`` — retrieve the
   kernel-assigned interface name (e.g. ``utun3``).

   .. note::
      The utun driver prepends a 4-byte address-family prefix (``AF_INET`` = 2
      in network byte order) to every datagram read from or written to the fd.
      FdNetDevice's ``UTUN`` encapsulation mode automatically strips this
      prefix on receive and adds it on send.

The fd returned by the helper is passed back to the ns-3 parent via the
same ``SCM_RIGHTS``/Unix-domain socket mechanism used on Linux.

Windows
#######

Windows has no in-kernel TUN/TAP API.  Bridging the simulator to a real
network interface requires the *tap-windows6* (OpenVPN TAP) virtual adapter
driver, which can be installed by OpenVPN or Tunnelblick.

Because the driver is a user-accessible device object, no privilege escalation
helper is needed; the ns-3 simulation process opens the adapter directly
(though administrator privileges are still required by the driver).

**Step 1 — Adapter discovery via the Registry**

The tap-windows6 driver registers each adapter under the NDIS network adapter
class key in HKLM:

.. code-block:: text

   HKLM\SYSTEM\CurrentControlSet\Control\Class\{4D36E972-E325-11CE-BFC1-08002BE10318}

``FindTapWindowsGuid()`` calls ``RegOpenKeyExA`` to open the class key, then
iterates over sub-keys with ``RegEnumKeyA``.  For each sub-key it reads the
``ComponentId`` value: ``tap0901`` or ``tap0902`` identifies a tap-windows6
adapter.  Once found, the ``NetCfgInstanceId`` value contains the GUID string
of the adapter (e.g. ``{55E64B16-AA3F-4A86-A120-7B2B8740432E}``).

**Step 2 — Opening the device**

The Windows device path for a tap-windows6 adapter is:

.. code-block:: text

   \\.\Global\{GUID}.tap

``CreateFileA()`` opens this path with ``GENERIC_READ | GENERIC_WRITE``.  The
``FILE_ATTRIBUTE_SYSTEM`` flag is required by the driver.

**Step 3 — Bringing the link up**

By default the adapter appears as "disconnected".  The tap-windows6 driver
exposes custom ``DeviceIoControl`` commands to control the virtual link:

.. code-block:: c

   #define TAP_WIN_IOCTL_SET_MEDIA_STATUS \
       CTL_CODE(FILE_DEVICE_UNKNOWN, 6, METHOD_BUFFERED, FILE_ANY_ACCESS)

   ULONG mediaStatus = 1;  // 1 = connected
   DeviceIoControl(hTap, TAP_WIN_IOCTL_SET_MEDIA_STATUS,
                   &mediaStatus, sizeof(mediaStatus),
                   &mediaStatus, sizeof(mediaStatus),
                   &returned, nullptr);

**Step 4 — TUN (L3) mode IP configuration**

For TUN mode, the driver also needs the local IP address, remote IP address,
and netmask so it can set up the point-to-point routing entry:

.. code-block:: c

   #define TAP_WIN_IOCTL_CONFIG_TUN \
       CTL_CODE(FILE_DEVICE_UNKNOWN, 10, METHOD_BUFFERED, FILE_ANY_ACCESS)

   ULONG ip[3];  // [0]=local, [1]=remote (same as local for P2P), [2]=netmask
   DeviceIoControl(hTap, TAP_WIN_IOCTL_CONFIG_TUN,
                   ip, sizeof(ip), ip, sizeof(ip), &returned, nullptr);

**Step 5 — Converting HANDLE to a POSIX fd**

The Windows C runtime provides ``_open_osfhandle()`` to wrap a Win32 ``HANDLE``
in a POSIX-style integer file descriptor.  The ``_O_RDWR | _O_BINARY`` flags
tell the CRT to treat the fd as a binary read/write stream:

.. code-block:: c

   int fd = _open_osfhandle((intptr_t)hTap, _O_RDWR | _O_BINARY);

This fd can then be passed to ``FdNetDevice::SetFileDescriptor()`` exactly as
on Linux or macOS.

Windows FdReader Threading Model
#################################

POSIX platforms use ``select()`` to wait for activity on multiple file
descriptors simultaneously, including a self-pipe ``evpipe`` used to signal
the read thread to stop.  Neither mechanism works for Windows anonymous pipes
or tap-windows6 device handles:

* Winsock's ``WSAEventSelect`` / ``select()`` only accept ``SOCKET`` handles,
  not arbitrary ``HANDLE`` values or POSIX-style fds.
* Anonymous pipes created by ``_pipe()`` do not support
  ``WaitForMultipleObjects``.

The Windows ``FdReader`` implementation (``win32-fd-reader.cc``) therefore
uses a polling approach based on ``PeekNamedPipe()``:

#. ``_get_osfhandle(m_fd)`` converts the POSIX fd to a Win32 ``HANDLE``.
#. The read loop calls ``PeekNamedPipe(hFd, nullptr, 0, nullptr, &avail,
   nullptr)`` once per millisecond (``Sleep(1)``) to check how many bytes are
   available without consuming them.
#. When ``avail > 0`` the loop calls ``DoRead()``, then invokes the receive
   callback.
#. When ``PeekNamedPipe`` itself fails (return value 0), the fd is not a
   pipe — it is a tap-windows6 device handle, for which the driver provides
   blocking read semantics directly.  In that case the loop calls ``DoRead()``
   directly without polling.

**Start/Run synchronization**

The ns-3 discrete-event simulator processes simulation time in near-zero
wall-clock time.  If the read thread is scheduled by the OS only *after* the
simulator has already advanced past the point where it would process the
incoming packet, the packet is silently lost.

To prevent this, ``FdReader::Start()`` creates a one-shot Windows Event object
(``CreateEvent(nullptr, FALSE, FALSE, nullptr)``) and blocks on
``WaitForSingleObject(m_eventsignal, INFINITE)`` after launching the read
thread.  The read thread signals the event (``SetEvent(m_eventsignal)``) after
completing its first loop iteration.

For the pipe case with pre-queued data (used in tests), the thread calls
``DoRead()``, invokes the receive callback which calls
``Simulator::ScheduleWithContext``, and *then* signals the event — so by the
time ``Start()`` returns to the simulator, the packet-receive event is already
in the scheduler queue at simulation time t=0.  When the simulator resumes
and processes t=0 events, it finds and processes the packet.

Testing
*******

The test suite (``src/fd-net-device/test/fd-net-device-test.cc``) contains:

* **FdNetDeviceReceiveTest** — injects a complete Ethernet II frame into the
  write end of an anonymous pipe and verifies that the FdNetDevice's receive
  callback is invoked with the correct EtherType (0x0800), payload size, and
  payload bytes.
* **FdNetDeviceSendTest** — schedules a ``device->Send()`` call and reads the
  resulting raw Ethernet frame from the read end of the pipe; verifies all
  header fields (destination MAC, source MAC, EtherType) and payload bytes.
* **FdNetDeviceDixpiReceiveTest** — same as the receive test, but with a
  4-byte PI header prepended to the frame; verifies that FdNetDevice in
  ``DIXPI`` mode strips the PI header correctly.
* **FdNetDeviceUtunReceiveTest** — injects a 4-byte ``AF_INET`` prefix plus a
  minimal 20-byte IPv4 header and verifies that FdNetDevice in ``UTUN`` mode
  strips the prefix and delivers the raw IP datagram.
* **FdNetDeviceLinuxTunProbeTest** (Linux only) — attempts to open
  ``/dev/net/tun`` and configure a ``IFF_TUN | IFF_NO_PI`` interface.  The
  test passes silently if ``/dev/net/tun`` is unavailable or
  ``CAP_NET_ADMIN`` is not granted.
* **FdNetDeviceMacOsUtunProbeTest** (macOS only) — attempts to create a utun
  socket using the ``SYSPROTO_CONTROL`` API.  The test passes silently if the
  API is unavailable.
* **FdNetDeviceWindowsTapRegistryTest** (Windows only) — reads the NDIS
  adapter class registry key and counts tap-windows6 adapters.  The test
  passes regardless of whether any adapter is installed; it exercises the
  registry scan code path used by ``TapFdNetDeviceHelper``.

All tests except the platform-specific probe tests run on every supported
platform (Linux, macOS, Windows) without any special hardware, drivers, or
privileges, using anonymous pipes to simulate the file-descriptor I/O.
