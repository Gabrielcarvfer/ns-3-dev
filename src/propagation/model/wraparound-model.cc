/*
 * Copyright (c) 2025 CTTC
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 */

#include "wraparound-model.h"

#include "ns3/log.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("WraparoundModel");

NS_OBJECT_ENSURE_REGISTERED(WraparoundModel);

TypeId
WraparoundModel::GetTypeId()
{
    static TypeId tid = TypeId("ns3::WraparoundModel").SetParent<Object>().SetGroupName("Mobility");
    return tid;
}

WraparoundModel::WraparoundModel()
{
}

} // namespace ns3
