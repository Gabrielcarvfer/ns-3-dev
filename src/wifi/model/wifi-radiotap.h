/*
 * Copyright (c) 2008 INRIA
 * Copyright (c) 2009 MIRKO BANCHI
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 *          Mirko Banchi <mk.banchi@gmail.com>
 */

#ifndef WIFI_RADIOTAP_H
#define WIFI_RADIOTAP_H

#include "wifi-phy-common.h"
#include "wifi-tx-vector.h"
#include "wifi-types.h"

#include "ns3/packet.h"
#include "ns3/radiotap-header.h"

#include <optional>

namespace ns3
{

/**
 * @ingroup wifi
 * Get the MPDU of a frame traced by the MonitorSnifferRx or MonitorSnifferTx
 * trace sources of WifiPhy, as a monitor interface delivers it: without the
 * A-MPDU delimiter and padding, nor the FCS (whose octets are not a checksum).
 *
 * @param packet the traced packet
 * @param txVector the TXVECTOR of the PPDU
 * @param[out] eof whether the A-MPDU delimiter had its EOF bit set
 * @return the MPDU
 */
Ptr<Packet> GetMonitorMpdu(Ptr<const Packet> packet, const WifiTxVector& txVector, bool& eof);

/**
 * @ingroup wifi
 * Get the Radiotap header of a frame traced by the MonitorSnifferRx or
 * MonitorSnifferTx trace sources of WifiPhy.
 *
 * @param channelFreqMhz the channel frequency
 * @param p20Index the index of the primary20 channel
 * @param txVector the TXVECTOR
 * @param aMpdu the A-MPDU information
 * @param eof whether the A-MPDU delimiter had its EOF bit set (see GetMonitorMpdu)
 * @param staId the STA-ID
 * @param signalNoise the rx signal and noise information (for receiver only)
 * @return the radiotap header
 */
RadiotapHeader GetRadiotapHeader(uint16_t channelFreqMhz,
                                 uint8_t p20Index,
                                 const WifiTxVector& txVector,
                                 MpduInfo aMpdu,
                                 bool eof,
                                 uint16_t staId,
                                 std::optional<SignalNoiseDbm> signalNoise = std::nullopt);

} // namespace ns3

#endif /* WIFI_RADIOTAP_H */
