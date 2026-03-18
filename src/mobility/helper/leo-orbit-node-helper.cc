// Copyright (c) Tim Schubert
//
// SPDX-License-Identifier: GPL-2.0-only
//
// Author: Tim Schubert <ns-3-leo@timschubert.net>
// Porting: Thiago Miyazaki <miyathiago@gmail.com> <t.miyazaki@unesp.br>

#include "leo-orbit-node-helper.h"

#include "mobility-helper.h"

#include "ns3/csv-reader.h"
#include "ns3/double.h"
#include "ns3/integer.h"
#include "ns3/log.h"
#include "ns3/uinteger.h"

#include <fstream>

using namespace std;

namespace ns3
{
NS_LOG_COMPONENT_DEFINE("LeoOrbitNodeHelper");

LeoOrbitNodeHelper::LeoOrbitNodeHelper(const Time& resolution)
{
    m_nodeFactory.SetTypeId("ns3::Node");
    m_resolution = resolution;
}

LeoOrbitNodeHelper::~LeoOrbitNodeHelper()
{
}

void
LeoOrbitNodeHelper::SetAttribute(string name, const AttributeValue& value)
{
    m_nodeFactory.Set(name, value);
}

NodeContainer
LeoOrbitNodeHelper::CreateNodesAndInstallMobility(const LeoOrbit& orbit)
{
    NS_LOG_FUNCTION(this << orbit);

    NodeContainer satelliteContainer;
    for (uint16_t i = 0; i < orbit.planes; i++)
    {
        for (int32_t j = 0; j < orbit.sats; j++)
        {
            satelliteContainer.Add(m_nodeFactory.Create<Node>());
        }
    }

    MobilityHelper mobility;
    mobility.SetPositionAllocator("ns3::LeoCircularOrbitPositionAllocator",
                                  "NumOrbits",
                                  IntegerValue(orbit.planes),
                                  "NumSatellites",
                                  IntegerValue(orbit.sats),
                                  "PhasingFactor",
                                  UintegerValue(orbit.phasing));
    mobility.SetMobilityModel("ns3::LeoCircularOrbitMobilityModel",
                              "Altitude",
                              DoubleValue(orbit.alt),
                              "Inclination",
                              DoubleValue(orbit.inc),
                              "Resolution",
                              TimeValue(m_resolution));
    mobility.Install(satelliteContainer);

    return satelliteContainer;
}

NodeContainer
LeoOrbitNodeHelper::CreateNodesAndInstallMobility(const std::string& orbitFile)
{
    NS_LOG_FUNCTION(this << orbitFile);

    // Read orbit file contents
    std::ifstream orbitsf;
    orbitsf.open(orbitFile);
    std::vector<LeoOrbit> orbits;
    std::string line;
    CsvReader csv(orbitFile);
    while (csv.FetchNextRow())
    {
        LeoOrbit orbit{};

        bool ok = csv.GetValue(0, orbit.alt);
        ok &= csv.GetValue(1, orbit.inc);
        ok &= csv.GetValue(2, orbit.planes);
        ok &= csv.GetValue(3, orbit.sats);
        if (ok)
        {
            // Optional 5th column: Walker Delta phasing factor
            csv.GetValue(4, orbit.phasing);
            orbits.push_back(orbit);
        }
    }
    orbitsf.close();

    NS_ABORT_MSG_IF(orbits.empty(), "Orbit files is empty or badly formatted.");

    return CreateNodesAndInstallMobility(orbits);
}

NodeContainer
LeoOrbitNodeHelper::CreateNodesAndInstallMobility(const vector<LeoOrbit>& orbits)
{
    NS_LOG_FUNCTION(this << orbits);

    NodeContainer nodes;
    for (auto& orbit : orbits)
    {
        nodes.Add(CreateNodesAndInstallMobility(orbit));
        NS_LOG_DEBUG("Added orbit plane");
    }

    NS_LOG_DEBUG("Added " << nodes.GetN() << " nodes");

    return nodes;
}

}; // namespace ns3
