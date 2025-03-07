/*
 * Copyright (c) 2010 Hemanth Narra
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Hemanth Narra <hemanth@ittc.ku.com>, written after OlsrHelper by Mathieu Lacage
 * <mathieu.lacage@sophia.inria.fr>
 *
 * James P.G. Sterbenz <jpgs@ittc.ku.edu>, director
 * ResiliNets Research Group  https://resilinets.org/
 * Information and Telecommunication Technology Center (ITTC)
 * and Department of Electrical Engineering and Computer Science
 * The University of Kansas Lawrence, KS USA.
 *
 * Work supported in part by NSF FIND (Future Internet Design) Program
 * under grant CNS-0626918 (Postmodern Internet Architecture),
 * NSF grant CNS-1050226 (Multilayer Network Resilience Analysis and Experimentation on GENI),
 * US Department of Defense (DoD), and ITTC at The University of Kansas.
 */

#ifndef DSDV_HELPER_H
#define DSDV_HELPER_H

#include "ns3/ipv4-routing-helper.h"
#include "ns3/node-container.h"
#include "ns3/node.h"
#include "ns3/object-factory.h"

namespace ns3
{
/**
 * @ingroup dsdv
 * @brief Helper class that adds DSDV routing to nodes.
 */
class DsdvHelper : public Ipv4RoutingHelper
{
  public:
    DsdvHelper();
    ~DsdvHelper() override;
    /**
     * @returns pointer to clone of this DsdvHelper
     *
     * This method is mainly for internal use by the other helpers;
     * clients are expected to free the dynamic memory allocated by this method
     */
    DsdvHelper* Copy() const override;

    /**
     * @param node the node on which the routing protocol will run
     * @returns a newly-created routing protocol
     *
     * This method will be called by ns3::InternetStackHelper::Install
     *
     */
    Ptr<Ipv4RoutingProtocol> Create(Ptr<Node> node) const override;
    /**
     * @param name the name of the attribute to set
     * @param value the value of the attribute to set.
     *
     * This method controls the attributes of ns3::dsdv::RoutingProtocol
     */
    void Set(std::string name, const AttributeValue& value);

  private:
    ObjectFactory m_agentFactory; //!< Object factory
};

} // namespace ns3

#endif /* DSDV_HELPER_H */
