/*
 * Copyright (c) 2025 CTTC
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Gabriel Ferreira <gabrielcarvfer@gmail.com>
 */

#include "wraparound-model.h"

#include "ns3/log.h"
#include "ns3/mobility-building-info.h"
#include "ns3/node.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("WraparoundModel");
NS_OBJECT_ENSURE_REGISTERED(WraparoundModel);

TypeId
WraparoundModel::GetTypeId()
{
    static TypeId tid = TypeId("ns3::WraparoundModel")
                            .SetParent<Object>()
                            .SetGroupName("Spectrum")
                            .AddConstructor<WraparoundModel>();
    return tid;
}

Ptr<MobilityModel>
WraparoundModel::GetVirtualMobilityModel(Ptr<const MobilityModel> tx,
                                         Ptr<const MobilityModel> rx) const
{
    NS_LOG_DEBUG("Transmitter using virtual mobility model. Real position "
                 << tx->GetPosition() << ", receiver position " << rx->GetPosition()
                 << ", wrapped position "
                 << GetVirtualPosition(tx->GetPosition(), rx->GetPosition()) << ".");
    // Creating a virtual mobility model for every signal and receiver dominated the cost of large
    // deployments; the propagation and channel models key their state by node, not by mobility
    // model, so the virtual mobility model of a pair can be reused and moved.
    auto& virtualMm = m_virtualMobilityModels[{tx, rx}];
    if (!virtualMm || virtualMm->GetVelocity() != tx->GetVelocity())
    {
        virtualMm = tx->Copy();

        // Unidirectionally aggregate NodeId to it, so it can be fetched later by
        // propagation models
        auto node = tx->GetObject<Node>();
        if (node)
        {
            virtualMm->UnidirectionalAggregateObject(node);
        }

        // Some mobility models access building info related to mobility model
        auto mbi = tx->GetObject<MobilityBuildingInfo>();
        if (mbi)
        {
            virtualMm->UnidirectionalAggregateObject(mbi);
        }
    }

    // Set the transmitter to its virtual position respective to receiver
    const auto virtualPosition = GetVirtualPosition(tx->GetPosition(), rx->GetPosition());
    if (virtualMm->GetPosition() != virtualPosition)
    {
        virtualMm->SetPosition(virtualPosition);
    }
    return virtualMm;
}

void
WraparoundModel::DoDispose()
{
    m_virtualMobilityModels.clear();
    Object::DoDispose();
}

Vector3D
WraparoundModel::GetVirtualPosition(const Vector3D tx, const Vector3D rx) const
{
    return tx; // Placeholder, you are supposed to implement whatever wraparound model you want
}
