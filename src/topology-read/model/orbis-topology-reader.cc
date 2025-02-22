/*
 * Copyright (c) 2010 Universita' di Firenze, Italy
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Tommaso Pecorella (tommaso.pecorella@unifi.it)
 * Author: Valerio Sartini (valesar@gmail.com)
 */

#include "orbis-topology-reader.h"

#include "ns3/log.h"
#include "ns3/names.h"
#include "ns3/node-container.h"

#include <cstdlib>
#include <fstream>
#include <sstream>

/**
 * @file
 * @ingroup topology
 * ns3::OrbisTopologyReader implementation.
 */

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("OrbisTopologyReader");

NS_OBJECT_ENSURE_REGISTERED(OrbisTopologyReader);

TypeId
OrbisTopologyReader::GetTypeId()
{
    static TypeId tid = TypeId("ns3::OrbisTopologyReader")
                            .SetParent<TopologyReader>()
                            .SetGroupName("TopologyReader")
                            .AddConstructor<OrbisTopologyReader>();
    return tid;
}

OrbisTopologyReader::OrbisTopologyReader()
{
    NS_LOG_FUNCTION(this);
}

OrbisTopologyReader::~OrbisTopologyReader()
{
    NS_LOG_FUNCTION(this);
}

NodeContainer
OrbisTopologyReader::Read()
{
    std::ifstream topgen;
    topgen.open(GetFileName());
    std::map<std::string, Ptr<Node>> nodeMap;
    NodeContainer nodes;

    if (!topgen.is_open())
    {
        return nodes;
    }

    std::string from;
    std::string to;
    std::istringstream lineBuffer;
    std::string line;

    int linksNumber = 0;
    int nodesNumber = 0;

    while (!topgen.eof())
    {
        line.clear();
        lineBuffer.clear();
        from.clear();
        to.clear();

        getline(topgen, line);
        lineBuffer.str(line);
        lineBuffer >> from;
        lineBuffer >> to;

        if ((!from.empty()) && (!to.empty()))
        {
            NS_LOG_INFO(linksNumber << " From: " << from << " to: " << to);
            if (!nodeMap[from])
            {
                Ptr<Node> tmpNode = CreateObject<Node>();
                std::string nodename = "OrbisTopology/NodeName/" + from;
                Names::Add(nodename, tmpNode);
                nodeMap[from] = tmpNode;
                nodes.Add(tmpNode);
                nodesNumber++;
            }

            if (!nodeMap[to])
            {
                Ptr<Node> tmpNode = CreateObject<Node>();
                std::string nodename = "OrbisTopology/NodeName/" + to;
                Names::Add(nodename, tmpNode);
                nodeMap[to] = tmpNode;
                nodes.Add(tmpNode);
                nodesNumber++;
            }

            Link link(nodeMap[from], from, nodeMap[to], to);
            AddLink(link);

            linksNumber++;
        }
    }
    NS_LOG_INFO("Orbis topology created with " << nodesNumber << " nodes and " << linksNumber
                                               << " links");
    topgen.close();

    return nodes;
}

} /* namespace ns3 */
