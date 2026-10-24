.. include:: replace.txt
.. highlight:: cpp

.. heading hierarchy:
   ------------- Chapter
   ************* Section (#.#)
   ============= Subsection (#.#.#)
   ############# Paragraph (no number)


IPv4
----

This chapter describes the |ns3| IPv4 address assignment and basic components tracking.

IPv4 addresses assignment
*************************

In order to use IPv4 on a network, the first thing to do is assigning IPv4 addresses.

Any IPv4-enabled |ns3| node will have at least one NetDevice: the :cpp:class:`ns3::LoopbackNetDevice`.
The loopback device address is ``127.0.0.1``.
All the other NetDevices will have one (or more) IPv4 addresses.

Note that, as today, |ns3| does not have a NAT module, and it does not follows the rules about
filtering private addresses (:rfc:`1918`): 10.0.0.0/8, 172.16.0.0/12, and 192.168.0.0/16.
These addresses are routed as any other address. This behaviour could change in the future.

IPv4 global addresses can be:

* manually assigned
* assigned though DHCP

|ns3| can use both methods, and it's quite important to understand the implications of both.

Manually assigned IPv4 addresses
================================

This is probably the easiest and most used method. As an example:

::

    Ptr<Node> n0 = CreateObject<Node>();
    Ptr<Node> n1 = CreateObject<Node>();
    NodeContainer net(n0, n1);
    CsmaHelper csma;
    NetDeviceContainer ndc = csma.Install(net);

    NS_LOG_INFO("Assign IPv4 Addresses.");
    Ipv4AddressHelper ipv4;
    ipv4.SetBase(Ipv4Address("192.168.1.0"), NetMask("/24"));
    Ipv4InterfaceContainer ic = ipv4.Assign(ndc);

This method will add two global IPv4 addresses to the nodes.

Note that the addresses are assigned in sequence. As a consequence, the first Node / NetDevice
will have "192.168.1.1", the second "192.168.1.2" and so on.

It is possible to repeat the above to assign more than one address to a node.
However, due to the :cpp:class:`Ipv4AddressHelper` singleton nature, one should first assign all the
addresses of a network, then change the network base (``SetBase``), then do a new assignment.

Alternatively, it is possible to assign a specific address to a node:

::

    Ptr<Node> n0 = CreateObject<Node>();
    NodeContainer net(n0);
    CsmaHelper csma;
    NetDeviceContainer ndc = csma.Install(net);

    NS_LOG_INFO("Specifically Assign an IPv4 Address.");
    Ipv4AddressHelper ipv4;
    Ptr<NetDevice> device = ndc.Get(0);
    Ptr<Node> node = device->GetNode();
    Ptr<Ipv4> ipv4proto = node->GetObject<Ipv4>();
    int32_t ifIndex = 0;
    ifIndex = ipv4proto->GetInterfaceForDevice(device);
    Ipv4InterfaceAddress ipv4Addr = Ipv4InterfaceAddress(Ipv4Address("192.168.1.42"), NetMask("/24"));
    ipv4proto->AddAddress(ifIndex, ipv4Addr);


DHCP assigned IPv4 addresses
============================

DHCP is available in the internet-apps module. In order to use DHCP you have to have a
:cpp:class:`DhcpServer` application in a node (the DHC server node) and a :cpp:class:`DhcpClient` application in
each of the nodes. Note that it not necessary that all the nodes in a subnet use DHCP. Some
nodes can have static addresses.

All the DHCP setup is performed though the :cpp:class:`DhcpHelper`  class. A complete example is in
``src/internet-apps/examples/dhcp-example.cc``.

Further info about the DHCP functionalities can be found in the ``internet-apps`` model documentation.


Tracing in the IPv4 Stack
*************************

The internet stack provides a number of trace sources in its various
protocol implementations.  These trace sources can be hooked using your own
custom trace code, or you can use our helper functions in some cases to
arrange for tracing to be enabled.

Tracing in ARP
==============

ARP provides two trace hooks, one in the cache, and one in the layer three
protocol.  The trace accessor in the cache is given the name "Drop."  When
a packet is transmitted over an interface that requires ARP, it is first
queued for transmission in the ARP cache until the required MAC address is
resolved.  There are a number of retries that may be done trying to get the
address, and if the maximum retry count is exceeded the packet in question
is dropped by ARP.  The single trace hook in the ARP cache is called,

- If an outbound packet is placed in the ARP cache pending address resolution
  and no resolution can be made within the maximum retry count, the outbound
  packet is dropped and this trace is fired;

A second trace hook lives in the ARP L3 protocol (also named "Drop") and may
be called for a  number of reasons.

- If an ARP reply is received for an entry that is not waiting for a reply,
  the ARP reply packet is dropped and this trace is fired;
- If an ARP reply is received for a non-existent entry, the ARP reply packet
  is dropped and this trace is fired;
- If an ARP cache entry is in the DEAD state (has timed out) and an ARP reply
  packet is received, the reply packet is dropped and this trace is fired.
- Each ARP cache entry has a queue of pending packets.  If the size of the
  queue is exceeded, the outbound packet is dropped and this trace is fired.

Tracing in IPv4
===============

The IPv4 layer three protocol provides three trace hooks.  These are the
"Tx" (ns3::Ipv4L3Protocol::m_txTrace), "Rx" (ns3::Ipv4L3Protocol::m_rxTrace)
and "Drop" (ns3::Ipv4L3Protocol::m_dropTrace) trace sources.

The "Tx" trace is fired in a number of situations, all of which indicate that
a given packet is about to be sent down to a given ns3::Ipv4Interface.

- In the case of a packet destined for the broadcast address, the
  Ipv4InterfaceList is iterated and for every interface that is up and can
  fragment the packet or has a large enough MTU to transmit the packet,
  the trace is hit.  See ns3::Ipv4L3Protocol::Send.

- In the case of a packet that needs routing, the "Tx" trace may be fired
  just before a packet is sent to the interface appropriate to the default
  gateway.  See ns3::Ipv4L3Protocol::SendRealOut.

- Also in the case of a packet that needs routing, the "Tx" trace may be
  fired just before a packet is sent to the outgoing interface appropriate
  to the discovered route.  See ns3::Ipv4L3Protocol::SendRealOut.

The "Rx" trace is fired when a packet is passed from the device up to the
ns3::Ipv4L3Protocol::Receive function.

- In the receive function, the Ipv4InterfaceList is iterated, and if the
  Ipv4Interface corresponding to the receiving device is found to be in the
  UP state, the trace is fired.

The "Drop" trace is fired in any case where the packet is dropped (in both
the transmit and receive paths).

- In the ns3::Ipv4Interface::Receive function, the packet is dropped and the
  drop trace is hit if the interface corresponding to the receiving device
  is in the DOWN state.

- Also in the ns3::Ipv4Interface::Receive function, the packet is dropped and
  the drop trace is hit if the checksum is found to be bad.

- In ns3::Ipv4L3Protocol::Send, an outgoing packet bound for the broadcast
  address is dropped and the "Drop" trace is fired if the "don't fragment"
  bit is set and fragmentation is available and required.

- Also in ns3::Ipv4L3Protocol::Send, an outgoing packet destined for the
  broadcast address is dropped and the "Drop" trace is hit if fragmentation
  is not available and is required (MTU < packet size).

- In the case of a broadcast address, an outgoing packet is cloned for each
  outgoing interface.  If any of the interfaces is in the DOWN state, the
  "Drop" trace event fires with a reference to the copied packet.

- In the case of a packet requiring a route, an outgoing packet is dropped
  and the "Drop" trace event fires if no route to the remote host is found.

- In ns3::Ipv4L3Protocol::SendRealOut, an outgoing packet being routed
  is dropped and the "Drop" trace is fired if the "don't fragment" bit is
  set and fragmentation is available and required.

- Also in ns3::Ipv4L3Protocol::SendRealOut, an outgoing packet being routed
  is dropped and the "Drop" trace is hit if fragmentation is not available
  and is required (MTU < packet size).

- An outgoing packet being routed is dropped and the "Drop" trace event fires
  if the required Ipv4Interface is in the DOWN state.

- If a packet is being forwarded, and the TTL is exceeded (see
  ns3::Ipv4L3Protocol::DoForward), the packet is dropped and the "Drop" trace
  event is fired.

Explicit Congestion Notification (ECN) bits
*******************************************

- In IPv4, ECN bits are the last 2 bits in TOS field and occupy 14th and 15th
  bits in the header.

- The IPv4 header class defines an EcnType enum with all four ECN codepoints
  (ECN_NotECT, ECN_ECT1, ECN_ECT0, ECN_CE) mentioned
  in RFC 3168, and also a setter and getter method to handle ECN values in
  the TOS field.

IPv4 options and loose source routing
*************************************

The ``Ipv4Header`` class models one option, the Loose Source and Record Route
(LSRR) of :rfc:`791`. The other options of a received header are kept as raw
bytes, in their place, so that a header serializes back to the bytes it was
read from: a forwarded datagram carries them unchanged (:rfc:`1812`, Section
5.3.13), and an ICMP error quotes them whole. When a datagram is fragmented,
only the options whose copied flag is set, the LSRR among them, appear in the
fragments but the first.

A received header which does not hold together is discarded by the IP layer,
with the ``DROP_BAD_CHECKSUM`` reason of the "Drop" trace, whether or not the
checksums are enabled: a datagram shorter than the 20 bytes of the fixed
fields or than its header length, an IHL field below 5, a total length below
the header length or beyond the bytes received, an option with an illegal
length, an LSRR whose pointer is below 4 or not on an address boundary, a
second LSRR, or a Strict Source and Record Route, which is not supported
(:rfc:`1812`, Section 5.2.4.1). The ``Ipv4Header::IsMalformed ()`` method
tells whether a header failed these checks. A pointer past the end of the
option means that the route is exhausted.

An application asks a datagram to follow a route by attaching an
``Ipv4SourceRouteTag`` holding the addresses of the route to it; the
destination is appended to a route which stops short of it. The first hop
travels in the destination field of the header and the others in the option,
which holds at most 9 addresses, so a route holds at most
``Ipv4SourceRouteTag::MAX_HOPS`` (10) hops, the destination included: the IP
layer drops a datagram whose route does not fit, with the ``DROP_ROUTE_ERROR``
reason. The route of a datagram sent to a broadcast or a multicast address is
ignored. A TCP socket follows a route of the application through
``TcpSocketBase::SetIpv4SourceRoute ()``, which refuses a route which cannot
fit or which holds an address which is not a unicast one, and its segment
size leaves room for the option, so that its full sized segments are not
fragmented (:rfc:`9293`, Section 3.7.1).

A node which receives a datagram whose route is not exhausted acts as a hop of
the route only if its ``SourceRouteForwarding`` attribute is set; the
attribute is false by default, and independent of ``IpForward``, like the
``accept_source_route`` setting of Linux (:rfc:`1122`, Section 3.3.5;
:rfc:`7126`, Section 4.3.5). A hop passes over the next addresses of the
route which are its own, then sends the datagram on towards the next one,
recording in its slot the address it forwards the datagram from. A node
which does not follow the route, or which has no route towards the next hop
or finds it is not a unicast address, discards the datagram and returns an
ICMP Destination Unreachable, Source Route Failed, unless the datagram is
itself an ICMP error message, a fragment but the first, or was addressed to
or sent from a broadcast or a multicast address. A datagram which reached a
hop as a broadcast or a multicast is never relayed. A hop decrements the TTL
like a router, and returns an ICMP Time Exceeded when it runs out.

At the final destination, the addresses the option recorded, reversed and
followed by the sender, are the way back, which the IP layer attaches to the
datagram as an ``Ipv4SourceRouteTag``. TCP sends the segments of a passively
opened connection along it, as well as the RSTs answering segments which
reach no connection, unless the application set a route of its own (:rfc:`9293`,
Section 3.9.2.1); a later acceptable segment recording a different route
updates it. The ICMP Echo Reply follows it too (:rfc:`1122`, Section
3.2.2.6). The UDP and raw sockets remove it from the datagrams they deliver:
the application which answers a datagram is the one to reverse its route,
which the raw sockets find in the header (:rfc:`1122`, Section 4.1.3.2).

Ipv4QueueDiscItem
*****************

The traffic control sublayer in |ns3| handles objects of class
``QueueDiscItem`` which are used to hold an ns3::Packet and an ns3::Header.
This is done to facilitate the marking of packets for Explicit
Congestion Notification.  The ``Mark ()`` method is implemented in
Ipv4QueueDiscItem. It returns true if marking the packet is successful, i.e.,
it successfully sets the CE bit in the IPv4 header. The ``Mark ()`` method
will return false, however, if the IPv4 header indicates the ``ECN_NotECT``
codepoint.

RFC 6621 duplicate packet detection
***********************************

To support mesh network protocols over broadcast-capable networks (e.g. Wi-Fi),
it is useful to have support for duplicate packet detection and filtering,
since nodes in a network may receive multiple copies of flooded multicast
packets arriving on different paths.  The ``Ipv4L3Protocol`` model in |ns3|
has a model for hash-based duplicate packet detection (DPD) based on
Section 6.2.2 of (:rfc:`6621`).  The model, disabled by default, must be
enabled by setting ``EnableRFC6621`` to true.  A second attribute,
``DuplicateExpire``, sets the expiration delay for erasing the cache entry
of a packet in the duplicate cache; the delay value defaults to 1ms.

NeighborCache
*************

NeighborCacheHelper provides a way to generate ARP cache automatically. It
generates needed ARP cache before simulation start to avoid the delay and message overhead of
address resolution in simulations that are focused on other performance aspects.
The state of entries which are generated by NeighborCacheHelper is ``STATIC_AUTOGENERATED``,
which is similar to ``PERMANENT``, but they are not manually added or removed by user, they
will be managed by NeighborCacheHelper when user need pre-generate cache.
When user is generating neighbor caches globally, neighbor caches will update dynamically when
IPv4 addresses are removed or added; when user is generating neighbor caches partially,
NeighborCacheHelper will take care of address removal, for adding address user may rerun a
reduced-scope PopulateNeighbor() again to pick up the new IP address or manually
add an entry to keep the neighbor cache up-to-date, the reason is that: when PopulateNeighborCache()
has previously been run with a scope less than global, the code does not know whether it was previously
run with a scope of Channel, NetDeviceContainer, or Ip interface container.
The source code for NeighborCache is located in ``src/internet/helper/neighbor-cache-helper``
A complete example is in ``src/internet/examples/neighbor-cache-example.cc``.

Usage
=====

The typical usages are::

* Populate neighbor ARP caches for all devices:

.. code-block:: c++

  NeighborCacheHelper neighborCache;
  neighborCache.PopulateNeighborCache();

* Populate neighbor ARP caches for a given channel:

.. code-block:: c++

  NeighborCacheHelper neighborCache;
  neighborCache.PopulateNeighborCache(channel);     // channel is the Ptr<Channel> want to generate ARP caches

* Populate neighbor ARP caches for devices in a given NetDeviceContainer:

.. code-block:: c++

  NeighborCacheHelper neighborCache;
  neighborCache.PopulateNeighborCache(netDevices);   // netDevices is the NetDeviceContainer want to generate ARP caches

* Populate neighbor ARP caches for a given Ipv4InterfaceContainer:

.. code-block:: c++

  NeighborCacheHelper neighborCache;
  neighborCache.PopulateNeighborCache(interfaces);    // interfaces is the Ipv4InterfaceContainer want to generate ARP caches

Neighbor cache entries are not removed by calls to ``ArpCache::Flush()``,
which may be called during the process of bringing IPv4 interfaces up
or down.  Instead, to remove them, users must call
``ArpCache::RemoveAutoGenerated()``.
