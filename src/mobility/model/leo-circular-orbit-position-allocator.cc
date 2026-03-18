// Copyright (c) Tim Schubert
//
// SPDX-License-Identifier: GPL-2.0-only
//
// Author: Tim Schubert <ns-3-leo@timschubert.net>
// Porting: Thiago Miyazaki <miyathiago@gmail.com> <t.miyazaki@unesp.br>

#include "leo-circular-orbit-position-allocator.h"

#include "math.h"

#include "ns3/integer.h"
#include "ns3/uinteger.h"

namespace ns3
{

NS_OBJECT_ENSURE_REGISTERED(LeoCircularOrbitAllocator);

LeoCircularOrbitAllocator::LeoCircularOrbitAllocator()
    : m_lastOrbit(0),
      m_lastSatellite(0)
{
}

LeoCircularOrbitAllocator::~LeoCircularOrbitAllocator()
{
}

TypeId
LeoCircularOrbitAllocator::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::LeoCircularOrbitPositionAllocator")
            .SetParent<PositionAllocator>()
            .SetGroupName("Leo")
            .AddConstructor<LeoCircularOrbitAllocator>()
            .AddAttribute("NumOrbits",
                          "The number of orbits",
                          IntegerValue(1),
                          MakeIntegerAccessor(&LeoCircularOrbitAllocator::m_numOrbits),
                          MakeIntegerChecker<uint16_t>())
            .AddAttribute("NumSatellites",
                          "The number of satellites per orbit",
                          IntegerValue(1),
                          MakeIntegerAccessor(&LeoCircularOrbitAllocator::m_numSatellites),
                          MakeIntegerChecker<uint16_t>())
            .AddAttribute("PhasingFactor",
                          "Walker Delta phasing factor F; staggers satellites in "
                          "adjacent planes by F * 360 / T degrees, where "
                          "T = NumOrbits * NumSatellites",
                          UintegerValue(0),
                          MakeUintegerAccessor(&LeoCircularOrbitAllocator::m_phasingFactor),
                          MakeUintegerChecker<uint16_t>());
    return tid;
}

int64_t
LeoCircularOrbitAllocator::AssignStreams(int64_t stream)
{
    return -1;
}

Vector
LeoCircularOrbitAllocator::GetNext() const
{
    double phasingOffset = m_phasingFactor * m_lastOrbit * 360.0 / (m_numOrbits * m_numSatellites);
    Vector next = Vector(360.0 * (m_lastOrbit / (double)m_numOrbits),
                         360.0 * (m_lastSatellite / (double)m_numSatellites) + phasingOffset,
                         m_lastSatellite);

    m_lastSatellite = (m_lastSatellite + 1) % m_numSatellites;
    if (!m_lastSatellite)
    {
        m_lastOrbit = (m_lastOrbit + 1) % m_numOrbits;
    }

    return next;
}

}; // namespace ns3
