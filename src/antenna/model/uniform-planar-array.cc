/*
 *   Copyright (c) 2020 University of Padova, Dep. of Information Engineering, SIGNET lab.
 *
 *   SPDX-License-Identifier: GPL-2.0-only
 */

#include "uniform-planar-array.h"

#include "ns3/boolean.h"
#include "ns3/double.h"
#include "ns3/log.h"
#include "ns3/uinteger.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("UniformPlanarArray");

NS_OBJECT_ENSURE_REGISTERED(UniformPlanarArray);

UniformPlanarArray::UniformPlanarArray()
    : PhasedArrayModel()
{
}

UniformPlanarArray::~UniformPlanarArray()
{
}

TypeId
UniformPlanarArray::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::UniformPlanarArray")
            .SetParent<PhasedArrayModel>()
            .AddConstructor<UniformPlanarArray>()
            .SetGroupName("Antenna")
            .AddAttribute(
                "AntennaHorizontalSpacing",
                "Horizontal spacing between antenna elements, in multiples of wave length",
                DoubleValue(0.5),
                MakeDoubleAccessor(&UniformPlanarArray::SetAntennaHorizontalSpacing,
                                   &UniformPlanarArray::GetAntennaHorizontalSpacing),
                MakeDoubleChecker<double>(0.0))
            .AddAttribute("AntennaVerticalSpacing",
                          "Vertical spacing between antenna elements, in multiples of wave length",
                          DoubleValue(0.5),
                          MakeDoubleAccessor(&UniformPlanarArray::SetAntennaVerticalSpacing,
                                             &UniformPlanarArray::GetAntennaVerticalSpacing),
                          MakeDoubleChecker<double>(0.0))
            .AddAttribute("NumColumns",
                          "Horizontal size of the array",
                          UintegerValue(4),
                          MakeUintegerAccessor(&UniformPlanarArray::SetNumColumns,
                                               &UniformPlanarArray::GetNumColumns),
                          MakeUintegerChecker<uint32_t>(1))
            .AddAttribute("NumRows",
                          "Vertical size of the array",
                          UintegerValue(4),
                          MakeUintegerAccessor(&UniformPlanarArray::SetNumRows,
                                               &UniformPlanarArray::GetNumRows),
                          MakeUintegerChecker<uint32_t>(1))
            .AddAttribute(
                "BearingAngle",
                "The bearing angle in radians",
                DoubleValue(0.0),
                MakeDoubleAccessor(&UniformPlanarArray::SetAlpha, &UniformPlanarArray::GetAlpha),
                MakeDoubleChecker<double>(-M_PI, M_PI))
            .AddAttribute(
                "DowntiltAngle",
                "The downtilt angle in radians",
                DoubleValue(0.0),
                MakeDoubleAccessor(&UniformPlanarArray::SetBeta, &UniformPlanarArray::GetBeta),
                MakeDoubleChecker<double>(-M_PI, M_PI))
            .AddAttribute("PolSlantAngle",
                          "The polarization slant angle in radians",
                          DoubleValue(0.0),
                          MakeDoubleAccessor(&UniformPlanarArray::SetPolSlant,
                                             &UniformPlanarArray::GetPolSlant),
                          MakeDoubleChecker<double>(-M_PI, M_PI))
            .AddAttribute("NumVerticalPorts",
                          "Vertical number of ports",
                          UintegerValue(1),
                          MakeUintegerAccessor(&UniformPlanarArray::GetNumVerticalPorts,
                                               &UniformPlanarArray::SetNumVerticalPorts),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("NumHorizontalPorts",
                          "Horizontal number of ports",
                          UintegerValue(1),
                          MakeUintegerAccessor(&UniformPlanarArray::GetNumHorizontalPorts,
                                               &UniformPlanarArray::SetNumHorizontalPorts),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("IsDualPolarized",
                          "If true, dual polarized antenna",
                          BooleanValue(false),
                          MakeBooleanAccessor(&UniformPlanarArray::SetDualPol,
                                              &UniformPlanarArray::IsDualPol),
                          MakeBooleanChecker());
    return tid;
}

void
UniformPlanarArray::SetNumColumns(uint32_t n)
{
    NS_LOG_FUNCTION(this << n);
    if (n != m_numColumns)
    {
        m_isBfVectorValid = false;
        InvalidateChannels();
    }
    m_numColumns = n;
}

uint32_t
UniformPlanarArray::GetNumColumns() const
{
    return m_numColumns;
}

void
UniformPlanarArray::SetNumRows(uint32_t n)
{
    NS_LOG_FUNCTION(this << n);
    if (n != m_numRows)
    {
        m_isBfVectorValid = false;
        InvalidateChannels();
    }
    m_numRows = n;
}

uint32_t
UniformPlanarArray::GetNumRows() const
{
    return m_numRows;
}

void
UniformPlanarArray::SetAlpha(double alpha)
{
    if (alpha != m_alpha)
    {
        m_alpha = alpha;
        m_cosAlpha = cos(m_alpha);
        m_sinAlpha = sin(m_alpha);
        InvalidateChannels();
    }
}

void
UniformPlanarArray::SetBeta(double beta)
{
    if (beta != m_beta)
    {
        m_beta = beta;
        m_cosBeta = cos(m_beta);
        m_sinBeta = sin(m_beta);
        InvalidateChannels();
    }
}

void
UniformPlanarArray::SetPolSlant(double polSlant)
{
    if (polSlant != m_polSlant)
    {
        m_polSlant = polSlant;
        m_cosPolSlant[0] = cos(m_polSlant);
        m_sinPolSlant[0] = sin(m_polSlant);
        InvalidateChannels();
    }
}

void
UniformPlanarArray::SetAntennaHorizontalSpacing(double s)
{
    NS_LOG_FUNCTION(this << s);
    NS_ABORT_MSG_IF(s <= 0, "Trying to set an invalid spacing: " << s);

    if (s != m_disH)
    {
        m_isBfVectorValid = false;
        InvalidateChannels();
    }
    m_disH = s;
}

double
UniformPlanarArray::GetAntennaHorizontalSpacing() const
{
    return m_disH;
}

void
UniformPlanarArray::SetAntennaVerticalSpacing(double s)
{
    NS_LOG_FUNCTION(this << s);
    NS_ABORT_MSG_IF(s <= 0, "Trying to set an invalid spacing: " << s);

    if (s != m_disV)
    {
        m_isBfVectorValid = false;
        InvalidateChannels();
    }
    m_disV = s;
}

double
UniformPlanarArray::GetAntennaVerticalSpacing() const
{
    return m_disV;
}

std::pair<double, double>
UniformPlanarArray::GetElementFieldPattern(Angles a, uint8_t polIndex) const
{
    NS_LOG_FUNCTION(this << a);
    NS_ASSERT_MSG(polIndex < GetNumPols(), "Polarization index can be 0 or 1.");

    // convert the theta and phi angles from GCS to LCS using eq. 7.1-7 and 7.1-8 in 3GPP TR 38.901
    // NOTE we assume a fixed slant angle of 0 degrees
    double inclination = a.GetInclination();
    double azimuth = a.GetAzimuth();
    double cosIncl = cos(inclination);
    double sinIncl = sin(inclination);
    double cosAzim = cos(azimuth - m_alpha);
    double sinAzim = sin(azimuth - m_alpha);
    double thetaPrime = std::acos(m_cosBeta * cosIncl + m_sinBeta * cosAzim * sinIncl);
    double phiPrime =
        std::arg(std::complex<double>(m_cosBeta * sinIncl * cosAzim - m_sinBeta * cosIncl,
                                      sinAzim * sinIncl));
    Angles aPrime(phiPrime, thetaPrime);
    NS_LOG_DEBUG(a << " -> " << aPrime);

    // compute the antenna element field patterns using eq. 7.3-4 and 7.3-5 in 3GPP TR 38.901,
    // using the configured polarization slant angle (m_polSlant)
    // NOTE: the slant angle (assumed to be 0) differs from the polarization slant angle
    // (m_polSlant, given by the attribute), in 3GPP TR 38.901
    double aPrimeDb = m_antennaElement->GetGainDb(aPrime);
    double fieldThetaPrime =
        pow(10, aPrimeDb / 20) * m_cosPolSlant[polIndex]; // convert to linear magnitude
    double fieldPhiPrime =
        pow(10, aPrimeDb / 20) * m_sinPolSlant[polIndex]; // convert to linear magnitude

    // compute psi using eq. 7.1-15 in 3GPP TR 38.901, assuming that the slant
    // angle (gamma) is 0
    double psi = std::arg(std::complex<double>(m_cosBeta * sinIncl - m_sinBeta * cosIncl * cosAzim,
                                               m_sinBeta * sinAzim));
    NS_LOG_DEBUG("psi " << psi);

    // convert the antenna element field pattern to GCS using eq. 7.1-11
    // in 3GPP TR 38.901
    double fieldTheta = cos(psi) * fieldThetaPrime - sin(psi) * fieldPhiPrime;
    double fieldPhi = sin(psi) * fieldThetaPrime + cos(psi) * fieldPhiPrime;
    NS_LOG_DEBUG(RadiansToDegrees(a.GetAzimuth())
                 << " " << RadiansToDegrees(a.GetInclination()) << " "
                 << fieldTheta * fieldTheta + fieldPhi * fieldPhi);

    return std::make_pair(fieldPhi, fieldTheta);
}

Vector
UniformPlanarArray::GetElementLocation(uint64_t index) const
{
    NS_LOG_FUNCTION(this << index);
    uint64_t tmpIndex = index;
    // for dual polarization, the top half corresponds to one polarization and
    // lower half corresponds to the other polarization
    if (m_isDualPolarized && tmpIndex >= m_numRows * m_numColumns)
    {
        tmpIndex -= m_numRows * m_numColumns;
    }
    // compute the element coordinates in the LCS
    // assume the left bottom corner is (0,0,0), and the rectangular antenna array is on the y-z
    // plane.
    double xPrime = 0;
    double yPrime = m_disH * (tmpIndex % m_numColumns);
    double zPrime = m_disV * floor(tmpIndex / m_numColumns);

    // convert the coordinates to the GCS using the rotation matrix 7.1-4 in 3GPP
    // TR 38.901
    Vector loc;
    loc.x = m_cosAlpha * m_cosBeta * xPrime - m_sinAlpha * yPrime + m_cosAlpha * m_sinBeta * zPrime;
    loc.y = m_sinAlpha * m_cosBeta * xPrime + m_cosAlpha * yPrime + m_sinAlpha * m_sinBeta * zPrime;
    loc.z = -m_sinBeta * xPrime + m_cosBeta * zPrime;
    return loc;
}

uint8_t
UniformPlanarArray::GetNumPols() const
{
    return m_isDualPolarized ? 2 : 1;
}

size_t
UniformPlanarArray::GetNumElems() const
{
    // From 38.901  [M, N, P, Mg, Ng] = [m_numRows, m_numColumns, 2, 1, 1]
    return GetNumPols() * m_numRows * m_numColumns;
    // with dual polarization, the number of antenna elements double up
}

void
UniformPlanarArray::SetNumVerticalPorts(uint16_t nPorts)
{
    NS_LOG_FUNCTION(this);
    NS_ASSERT_MSG(nPorts > 0, "Ports should be greater than 0");
    NS_ASSERT_MSG(((m_numRows % nPorts) == 0),
                  "The number of vertical ports must divide number of rows");
    if (nPorts != m_numVPorts)
    {
        m_numVPorts = nPorts;
        InvalidateChannels();
    }
}

void
UniformPlanarArray::SetNumHorizontalPorts(uint16_t nPorts)
{
    NS_ASSERT_MSG(nPorts > 0, "Ports should be greater than 0");
    NS_ASSERT_MSG(((m_numColumns % nPorts) == 0),
                  "The number of horizontal ports must divide number of columns");
    if (nPorts != m_numHPorts)
    {
        m_numHPorts = nPorts;
        InvalidateChannels();
    }
}

uint16_t
UniformPlanarArray::GetNumVerticalPorts() const
{
    return m_numVPorts;
}

uint16_t
UniformPlanarArray::GetNumHorizontalPorts() const
{
    return m_numHPorts;
}

uint16_t
UniformPlanarArray::GetNumPorts() const
{
    return GetNumPols() * m_numVPorts * m_numHPorts;
}

size_t
UniformPlanarArray::GetVElemsPerPort() const
{
    return m_numRows / m_numVPorts;
}

size_t
UniformPlanarArray::GetHElemsPerPort() const
{
    return m_numColumns / m_numHPorts;
}

size_t
UniformPlanarArray::GetNumElemsPerPort() const
{
    // Multiply the number of rows and number of columns belonging to one antenna port.
    // This also holds for dual polarization, where each polarization belongs to a separate port.
    return GetVElemsPerPort() * GetHElemsPerPort();
}

uint16_t
UniformPlanarArray::ArrayIndexFromPortIndex(uint16_t portIndex, uint16_t subElementIndex) const
{
    NS_ASSERT_MSG(portIndex < GetNumPorts(), "Port should be less than total Ports");
    NS_ASSERT(subElementIndex < (GetHElemsPerPort() * GetVElemsPerPort()));

    // In case the array is dual-polarized, change to the index that belongs to the first
    // polarization
    auto firstPolPortIdx = portIndex;
    auto polarizationOffset = 0;
    auto arraySize = GetNumHorizontalPorts() * GetNumVerticalPorts();
    if (firstPolPortIdx >= arraySize)
    {
        firstPolPortIdx = portIndex - arraySize;
        polarizationOffset = GetNumColumns() * GetNumRows();
    }
    // column-major indexing
    auto hPortIdx = firstPolPortIdx / GetNumVerticalPorts();
    auto vPortIdx = firstPolPortIdx % GetNumVerticalPorts();
    auto hElemIdx = (hPortIdx * GetHElemsPerPort()) + (subElementIndex % GetHElemsPerPort());
    auto vElemIdx = (vPortIdx * GetVElemsPerPort()) + (subElementIndex / GetHElemsPerPort());
    return (vElemIdx * GetNumColumns() + hElemIdx + polarizationOffset);
}

bool
UniformPlanarArray::IsDualPol() const
{
    return m_isDualPolarized;
}

void
UniformPlanarArray::SetDualPol(bool isDualPol)
{
    if (isDualPol != m_isDualPolarized)
    {
        m_isDualPolarized = isDualPol;
        if (isDualPol)
        {
            m_cosPolSlant[1] = cos(m_polSlant - M_PI / 2);
            m_sinPolSlant[1] = sin(m_polSlant - M_PI / 2);
        }
        InvalidateChannels();
    }
}

double
UniformPlanarArray::GetAlpha() const
{
    return m_alpha;
}

double
UniformPlanarArray::GetBeta() const
{
    return m_beta;
}

double
UniformPlanarArray::GetPolSlant() const
{
    return m_polSlant;
}

uint8_t
UniformPlanarArray::GetElemPol(size_t elemIndex) const
{
    NS_ASSERT(elemIndex < GetNumElems());
    return (elemIndex < GetNumRows() * GetNumColumns()) ? 0 : 1;
}

} /* namespace ns3 */
