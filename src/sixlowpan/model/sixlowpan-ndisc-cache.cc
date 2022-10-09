/*
 * Copyright (c) 2015 Università di Firenze, Italy
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Alessio Bonadio <alessio.bonadio@gmail.com>
 *         Tommaso Pecorella <tommaso.pecorella@unifi.it>
 *         Adnan Rashid <adnanrashidpk@gmail.com>
 */

#include "sixlowpan-ndisc-cache.h"

#include "sixlowpan-nd-context.h"
#include "sixlowpan-nd-prefix.h"
#include "sixlowpan-nd-protocol.h"

#include "ns3/ipv6-address.h"
#include "ns3/ipv6-l3-protocol.h"
#include "ns3/ipv6-routing-protocol.h"
#include "ns3/log.h"
#include "ns3/mac16-address.h"
#include "ns3/mac64-address.h"
#include "ns3/names.h"
#include "ns3/node.h"
#include "ns3/uinteger.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("SixLowPanNdiscCache");

NS_OBJECT_ENSURE_REGISTERED(SixLowPanNdiscCache);

TypeId
SixLowPanNdiscCache::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::SixLowPanNdiscCache").SetParent<NdiscCache>().SetGroupName("SixLowPan");
    return tid;
}

SixLowPanNdiscCache::SixLowPanNdiscCache()
{
    NS_LOG_FUNCTION(this);
}

SixLowPanNdiscCache::~SixLowPanNdiscCache()
{
    NS_LOG_FUNCTION(this);
    Flush();
}

void
SixLowPanNdiscCache::DoDispose()
{
    NS_LOG_FUNCTION(this);

    Flush();
    NdiscCache::DoDispose();
}

NdiscCache::Entry*
SixLowPanNdiscCache::Lookup(Ipv6Address dst)
{
    NS_LOG_FUNCTION(this << dst);
    SixLowPanNdiscCache::SixLowPanEntry* entry;

    if (m_ndCache.find(dst) != m_ndCache.end())
    {
        entry = dynamic_cast<SixLowPanNdiscCache::SixLowPanEntry*>(m_ndCache[dst]);
    }
    else
    {
        entry = nullptr;
    }
    return entry;
}

NdiscCache::Entry*
SixLowPanNdiscCache::Add(Ipv6Address to)
{
    NS_LOG_FUNCTION(this << to);
    NS_ASSERT(m_ndCache.find(to) == m_ndCache.end());

    SixLowPanNdiscCache::SixLowPanEntry* entry = new SixLowPanNdiscCache::SixLowPanEntry(this);
    entry->SetIpv6Address(to);
    m_ndCache[to] = entry;
    return entry;
}

void
SixLowPanNdiscCache::PrintNdiscCache(Ptr<OutputStreamWrapper> stream)
{
    NS_LOG_FUNCTION(this << stream);
    std::ostream* os = stream->GetStream();

    for (auto i = m_ndCache.begin(); i != m_ndCache.end(); i++)
    {
        *os << i->first << " dev ";
        std::string found = Names::FindName(GetDevice());
        if (Names::FindName(GetDevice()) != "")
        {
            *os << found;
        }
        else
        {
            *os << static_cast<int>(GetDevice()->GetIfIndex());
        }

        SixLowPanNdiscCache::SixLowPanEntry* entry =
            dynamic_cast<SixLowPanNdiscCache::SixLowPanEntry*>(i->second);
        *os << " lladdr " << entry->GetMacAddress();

        if (entry->IsReachable())
        {
            *os << " REACHABLE ";
        }
        else if (entry->IsDelay())
        {
            *os << " DELAY ";
        }
        else if (entry->IsIncomplete())
        {
            *os << " INCOMPLETE ";
        }
        else if (entry->IsProbe())
        {
            *os << " PROBE ";
        }
        else
        {
            *os << " STALE ";
        }

        if (entry->IsRegistered())
        {
            *os << "REGISTERED\n";
        }
        else if (entry->IsTentative())
        {
            *os << "TENTATIVE\n";
        }
        else
        {
            *os << "GARBAGE-COLLECTIBLE\n";
        }
    }
}

SixLowPanNdiscCache::SixLowPanEntry::SixLowPanEntry(NdiscCache* nd)
    : NdiscCache::Entry::Entry(nd),
      m_type(GARBAGE),
      m_registeredTimer(Timer::CANCEL_ON_DESTROY),
      m_tentativeTimer(Timer::CANCEL_ON_DESTROY)
{
    NS_LOG_FUNCTION(this);
}

void
SixLowPanNdiscCache::SixLowPanEntry::Print(std::ostream& os) const
{
    NdiscCache::Entry::Print(os);
    switch (m_type)
    {
    case REGISTERED:
        os << " - REGISTERED";
        break;
    case GARBAGE:
        os << " - GARBAGE-COLLECTIBLE";
        break;
    case TENTATIVE:
        os << " - TENTATIVE";
        break;
    }
}

void
SixLowPanNdiscCache::SixLowPanEntry::MarkRegistered(uint16_t time)
{
    NS_LOG_FUNCTION(this);
    m_type = REGISTERED;

    //  Ptr<Node> node = m_ndCache->GetDevice ()->GetNode ();
    //  std::cout << "++++ " << node->GetId () << " + " << Now ().As (Time::S) << " MarkRegistered -
    //  " << *this << std::endl;

    if (m_tentativeTimer.IsRunning())
    {
        m_tentativeTimer.Cancel();
    }

    if (m_registeredTimer.IsRunning())
    {
        m_registeredTimer.Cancel();
    }
    m_registeredTimer.SetFunction(&SixLowPanNdiscCache::SixLowPanEntry::FunctionTimeout, this);
    m_registeredTimer.SetDelay(Minutes(time));
    m_registeredTimer.Schedule();
}

void
SixLowPanNdiscCache::SixLowPanEntry::MarkTentative()
{
    NS_LOG_FUNCTION(this);
    m_type = TENTATIVE;

    if (m_tentativeTimer.IsRunning())
    {
        m_tentativeTimer.Cancel();
    }
    m_tentativeTimer.SetFunction(&SixLowPanNdiscCache::SixLowPanEntry::FunctionTimeout, this);
    m_tentativeTimer.SetDelay(Seconds(SixLowPanNdProtocol::TENTATIVE_NCE_LIFETIME));
    m_tentativeTimer.Schedule();
}

void
SixLowPanNdiscCache::SixLowPanEntry::MarkGarbage()
{
    NS_LOG_FUNCTION(this);
    m_type = GARBAGE;
}

bool
SixLowPanNdiscCache::SixLowPanEntry::IsRegistered() const
{
    NS_LOG_FUNCTION(this);
    return (m_type == REGISTERED);
}

bool
SixLowPanNdiscCache::SixLowPanEntry::IsTentative() const
{
    NS_LOG_FUNCTION(this);
    return (m_type == TENTATIVE);
}

bool
SixLowPanNdiscCache::SixLowPanEntry::IsGarbage() const
{
    NS_LOG_FUNCTION(this);
    return (m_type == GARBAGE);
}

void
SixLowPanNdiscCache::SixLowPanEntry::FunctionTimeout()
{
    NS_LOG_FUNCTION(this);

    Ptr<Node> node = m_ndCache->GetDevice()->GetNode();
    //  std::cout << "**** " << node->GetId () << " * " << Now ().As (Time::S) << " timeout -
    //  removing entry " << *this << std::endl;

    Ptr<Ipv6L3Protocol> ipv6l3Protocol = node->GetObject<Ipv6L3Protocol>();
    ipv6l3Protocol->GetRoutingProtocol()->NotifyRemoveRoute(
        GetIpv6Address(),
        Ipv6Prefix(128),
        Ipv6Address::GetAny(),
        ipv6l3Protocol->GetInterfaceForDevice(m_ndCache->GetDevice()));
    m_ndCache->Remove(this);

    return;
}

std::vector<uint8_t>
SixLowPanNdiscCache::SixLowPanEntry::GetRovr() const
{
    return m_rovr;
}

void
SixLowPanNdiscCache::SixLowPanEntry::SetRovr(const std::vector<uint8_t>& rovr)
{
    m_rovr = std::move(rovr);
}

std::ostream&
operator<<(std::ostream& os, const SixLowPanNdiscCache::SixLowPanEntry& entry)
{
    entry.Print(os);
    return os;
}

} /* namespace ns3 */
