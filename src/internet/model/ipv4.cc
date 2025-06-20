/*
 * Copyright (c) 2007 INRIA
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */

#include "ipv4.h"

#include "ns3/assert.h"
#include "ns3/boolean.h"
#include "ns3/log.h"
#include "ns3/node.h"
#include "ns3/warnings.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("Ipv4");

NS_OBJECT_ENSURE_REGISTERED(Ipv4);

TypeId
Ipv4::GetTypeId()
{
    NS_WARNING_PUSH_DEPRECATED;
    static TypeId tid =
        TypeId("ns3::Ipv4")
            .SetParent<Object>()
            .SetGroupName("Internet")
            .AddAttribute(
                "IpForward",
                "Globally enable or disable IP forwarding for all current and future Ipv4 devices.",
                BooleanValue(true),
                MakeBooleanAccessor(&Ipv4::SetIpForward, &Ipv4::GetIpForward),
                MakeBooleanChecker())
            .AddAttribute(
                "StrongEndSystemModel",
                "Reject packets for an address not configured on the interface they're "
                "coming from (RFC1122, section 3.3.4.2).",
                BooleanValue(false),
                MakeBooleanAccessor(&Ipv4::SetStrongEndSystemModel, &Ipv4::GetStrongEndSystemModel),
                MakeBooleanChecker())
#if 0
    .AddAttribute ("MtuDiscover", "If enabled, every outgoing ip packet will have the DF flag set.",
                   BooleanValue (false),
                   MakeBooleanAccessor (&UdpSocket::SetMtuDiscover,
                                        &UdpSocket::GetMtuDiscover),
                   MakeBooleanChecker ())
#endif
        ;
    NS_WARNING_POP;
    return tid;
}

Ipv4::Ipv4()
{
    NS_LOG_FUNCTION(this);
}

Ipv4::~Ipv4()
{
    NS_LOG_FUNCTION(this);
}

} // namespace ns3
