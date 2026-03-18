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

    NS_ABORT_MSG_IF(orbit.planes == 0, "Number of orbital planes must be > 0");
    NS_ABORT_MSG_IF(orbit.sats == 0, "Number of satellites per plane must be > 0");
    NS_ABORT_MSG_IF(orbit.alt <= 0, "Orbital altitude must be > 0 km");
    NS_ABORT_MSG_IF(orbit.inc == 0, "Orbital inclination must not be 0 degrees");
    NS_ABORT_MSG_IF(orbit.raanSpanDeg <= 0 || orbit.raanSpanDeg > 360,
                    "RAAN span must be in (0, 360] degrees");
    NS_ABORT_MSG_IF(orbit.phasing >= orbit.planes, "Phasing factor must be in [0, planes - 1]");

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
                                  UintegerValue(orbit.planes),
                                  "NumSatellites",
                                  UintegerValue(orbit.sats),
                                  "PhasingFactor",
                                  UintegerValue(orbit.phasing),
                                  "RaanSpanDeg",
                                  DoubleValue(orbit.raanSpanDeg));
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
        // Require at least 4 columns; allow up to 6
        if (csv.ColumnCount() < 4)
        {
            NS_LOG_WARN("Skipping row " << csv.RowNumber() << " of " << orbitFile
                                        << ": expected at least 4 columns, got "
                                        << csv.ColumnCount());
            continue;
        }

        LeoOrbit orbit{};

        bool ok = csv.GetValue(0, orbit.alt);
        ok &= csv.GetValue(1, orbit.inc);
        ok &= csv.GetValue(2, orbit.planes);
        ok &= csv.GetValue(3, orbit.sats);
        if (!ok)
        {
            NS_LOG_WARN("Skipping row " << csv.RowNumber() << " of " << orbitFile
                                        << ": non-numeric value in required column");
            continue;
        }

        // Optional 5th column: Walker Delta phasing factor
        csv.GetValue(4, orbit.phasing);
        // Optional 6th column: RAAN span in degrees (360 = Delta, 180 = Star)
        csv.GetValue(5, orbit.raanSpanDeg);
        orbits.push_back(orbit);
    }
    orbitsf.close();

    NS_ABORT_MSG_IF(orbits.empty(),
                    "No valid orbit rows found in " << orbitFile << "; check file format");

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
