/*
 *   Copyright (c) 2020 University of Padova, Dep. of Information Engineering, SIGNET lab.
 *
 *   SPDX-License-Identifier: GPL-2.0-only
 */

#include "phased-array-model.h"

#include "isotropic-antenna-model.h"

#include "ns3/boolean.h"
#include "ns3/double.h"
#include "ns3/log.h"
#include "ns3/pointer.h"
#include "ns3/uinteger.h"

namespace ns3
{

uint32_t PhasedArrayModel::m_idCounter = 0;
SymmetricAdjacencyMatrix<bool> PhasedArrayModel::m_outOfDateAntennaPairChannel;

NS_LOG_COMPONENT_DEFINE("PhasedArrayModel");

NS_OBJECT_ENSURE_REGISTERED(PhasedArrayModel);

PhasedArrayModel::PhasedArrayModel()
    : m_isBfVectorValid{false}
{
    m_id = m_idCounter++;
    m_outOfDateAntennaPairChannel.AddRow();
    m_outOfDateAntennaPairChannel.SetValueAdjacent(m_id, true);
}

PhasedArrayModel::~PhasedArrayModel()
{
}

TypeId
PhasedArrayModel::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::PhasedArrayModel")
            .SetParent<Object>()
            .SetGroupName("Antenna")
            .AddAttribute("AntennaElement",
                          "A pointer to the antenna element used by the phased array",
                          PointerValue(CreateObject<IsotropicAntennaModel>()),
                          MakePointerAccessor(&PhasedArrayModel::m_antennaElement),
                          MakePointerChecker<AntennaModel>());
    return tid;
}

void
PhasedArrayModel::SetBeamformingVector(const ComplexVector& beamformingVector)
{
    NS_LOG_FUNCTION(this << beamformingVector);
    NS_ASSERT_MSG(beamformingVector.GetSize() == GetNumElems(),
                  beamformingVector.GetSize() << " != " << GetNumElems());
    m_beamformingVector = beamformingVector;
    m_isBfVectorValid = true;
    m_bfHashValid = false;
}

uint64_t
PhasedArrayModel::GetBeamformingVectorHash() const
{
    NS_ASSERT_MSG(m_isBfVectorValid,
                  "The beamforming vector should be Set before its hash is requested");
    if (!m_bfHashValid)
    {
        // FNV-1a over the raw complex weights. Cached because beam-aware
        // spectrum caches request it once per (tx, rx) evaluation —
        // millions of times per simulated second — while beams change
        // rarely in comparison.
        uint64_t h = 1469598103934665603ull;
        const size_t n = m_beamformingVector.GetSize();
        for (size_t i = 0; i < n; ++i)
        {
            const auto c = m_beamformingVector[i];
            const double parts[2] = {c.real(), c.imag()};
            const auto* bytes = reinterpret_cast<const unsigned char*>(parts);
            for (size_t b = 0; b < sizeof(parts); ++b)
            {
                h ^= bytes[b];
                h *= 1099511628211ull;
            }
        }
        m_bfHash = h;
        m_bfHashValid = true;
    }
    return m_bfHash;
}

PhasedArrayModel::ComplexVector
PhasedArrayModel::GetBeamformingVector() const
{
    NS_LOG_FUNCTION(this);
    NS_ASSERT_MSG(m_isBfVectorValid,
                  "The beamforming vector should be Set before it's Get, and should refer to the "
                  "current array configuration");
    return m_beamformingVector;
}

const PhasedArrayModel::ComplexVector&
PhasedArrayModel::GetBeamformingVectorRef() const
{
    NS_LOG_FUNCTION(this);
    NS_ASSERT_MSG(m_isBfVectorValid,
                  "The beamforming vector should be Set before it's Get, and should refer to the "
                  "current array configuration");
    return m_beamformingVector;
}

PhasedArrayModel::ComplexVector
PhasedArrayModel::GetBeamformingVector(Angles a) const
{
    NS_LOG_FUNCTION(this << a);

    ComplexVector beamformingVector = GetSteeringVector(a);
    // The normalization takes into account the total number of ports as only a
    // portion (K,L) of beam weights associated with a specific port are non-zero.
    // See 3GPP Section 5.2.2 36.897. This normalization corresponds to
    // a sub-array partition model (which is different from the full-connection
    // model). Note that the total number of ports used to perform normalization
    // is the ratio between the total number of antenna elements and the
    // number of antenna elements per port.
    double normRes = norm(beamformingVector) / sqrt(GetNumPorts());

    for (size_t i = 0; i < GetNumElems(); i++)
    {
        beamformingVector[i] = std::conj(beamformingVector[i]) / normRes;
    }

    return beamformingVector;
}

PhasedArrayModel::ComplexVector
PhasedArrayModel::GetSteeringVector(Angles a) const
{
    ComplexVector steeringVector(GetNumElems());
    for (size_t i = 0; i < GetNumElems(); i++)
    {
        Vector loc = GetElementLocation(i);
        double phase = -2 * M_PI *
                       (sin(a.GetInclination()) * cos(a.GetAzimuth()) * loc.x +
                        sin(a.GetInclination()) * sin(a.GetAzimuth()) * loc.y +
                        cos(a.GetInclination()) * loc.z);
        steeringVector[i] = std::polar<double>(1.0, phase);
    }
    return steeringVector;
}

void
PhasedArrayModel::SetAntennaElement(Ptr<AntennaModel> antennaElement)
{
    NS_LOG_FUNCTION(this);
    m_antennaElement = antennaElement;
}

Ptr<const AntennaModel>
PhasedArrayModel::GetAntennaElement() const
{
    NS_LOG_FUNCTION(this);
    return m_antennaElement;
}

uint32_t
PhasedArrayModel::GetId() const
{
    return m_id;
}

bool
PhasedArrayModel::IsChannelOutOfDate(Ptr<const PhasedArrayModel> antennaB) const
{
    // Check that channel needs update
    bool needsUpdate = m_outOfDateAntennaPairChannel.GetValue(m_id, antennaB->m_id);

    // Assume the channel will be updated (needsUpdate == true), reset these
    m_outOfDateAntennaPairChannel.SetValue(m_id, antennaB->m_id, false);
    return needsUpdate;
}

void
PhasedArrayModel::InvalidateChannels() const
{
    m_outOfDateAntennaPairChannel.SetValueAdjacent(m_id, true);
}

} /* namespace ns3 */
