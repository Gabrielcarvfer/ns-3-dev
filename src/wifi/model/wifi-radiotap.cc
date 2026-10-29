/*
 * Copyright (c) 2008 INRIA
 * Copyright (c) 2009 MIRKO BANCHI
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 *          Mirko Banchi <mk.banchi@gmail.com>
 */

#include "wifi-radiotap.h"

#include "ampdu-subframe-header.h"
#include "wifi-mac-trailer.h"
#include "wifi-ru.h"

#include "ns3/eht-ppdu.h"
#include "ns3/he-ru.h"
#include "ns3/simulator.h"

#include <bit>
#include <map>

namespace ns3
{

namespace
{
/**
 * Helper function to place the value to set in the correct bit(s) of a radiotap subfield.
 *
 * @param mask the mask of the corresponding subfield
 * @param val the value the subfield should be set to
 * @return the value placed at the correct position based on the mask
 */
uint32_t
GetRadiotapField(uint32_t mask, uint32_t val)
{
    const auto shift = std::countr_zero(mask);
    return (val << shift) & mask;
}

/// 320 MHz channelization map
const std::map<MHz_u, RadiotapHeader::UsigCommonBw> channelization320MHzMap{
    {MHz_u{6105}, RadiotapHeader::USIG_COMMON_BW_320MHZ_1},
    {MHz_u{6425}, RadiotapHeader::USIG_COMMON_BW_320MHZ_1},
    {MHz_u{6745}, RadiotapHeader::USIG_COMMON_BW_320MHZ_1},
    {MHz_u{6265}, RadiotapHeader::USIG_COMMON_BW_320MHZ_2},
    {MHz_u{6585}, RadiotapHeader::USIG_COMMON_BW_320MHZ_2},
    {MHz_u{6905}, RadiotapHeader::USIG_COMMON_BW_320MHZ_2},
};
} // namespace

Ptr<Packet>
GetMonitorMpdu(Ptr<const Packet> packet, const WifiTxVector& txVector, bool& eof)
{
    Ptr<Packet> mpdu = packet->Copy();
    eof = false;
    if (txVector.IsAggregation())
    {
        // remove the MPDU delimiter and the padding, as the MAC driver does
        AmpduSubframeHeader hdr;
        mpdu->RemoveHeader(hdr);
        eof = hdr.GetEof() && hdr.GetLength() > 0;
        mpdu = mpdu->CreateFragment(0, static_cast<uint32_t>(hdr.GetLength()));
    }
    // The FCS added by the simulator is not a real checksum (its four octets are
    // zero), hence it is removed, so that packet analyzers do not check it (see
    // issue #300)
    WifiMacTrailer fcs;
    mpdu->RemoveTrailer(fcs);
    return mpdu;
}

RadiotapHeader
GetRadiotapHeader(uint16_t channelFreqMhz,
                  uint8_t p20Index,
                  const WifiTxVector& txVector,
                  MpduInfo aMpdu,
                  bool eof,
                  uint16_t staId,
                  std::optional<SignalNoiseDbm> signalNoise)
{
    RadiotapHeader header;

    const auto preamble = txVector.GetPreambleType();
    const auto modClass = txVector.GetModulationClass();
    const auto channelWidth = txVector.GetChannelWidth();
    const auto gi = txVector.GetGuardInterval();

    header.SetWifiHeader(IsEht(preamble) ? 2 : 1);

    header.SetTsft(Simulator::Now().GetMicroSeconds());

    uint8_t frameFlags = RadiotapHeader::FRAME_FLAG_NONE;
    if (preamble == WIFI_PREAMBLE_SHORT)
    {
        frameFlags |= RadiotapHeader::FRAME_FLAG_SHORT_PREAMBLE;
    }
    if (gi.GetNanoSeconds() == 400)
    {
        frameFlags |= RadiotapHeader::FRAME_FLAG_SHORT_GUARD;
    }
    header.SetFrameFlags(frameFlags);

    uint8_t mcs = 0;
    uint8_t nss = 1;
    uint64_t rate = 0;
    if (modClass < WIFI_MOD_CLASS_HT)
    {
        rate = txVector.GetMode(staId).GetDataRate(channelWidth, gi, 1) * nss / 500000;
        header.SetRate(static_cast<uint8_t>(rate));
    }
    else
    {
        mcs = txVector.GetMode(staId).GetMcsValue();
        nss = txVector.GetNss(staId);
    }

    RadiotapHeader::ChannelFields channelFields{.frequency = channelFreqMhz};
    switch (rate)
    {
    case 2:  // 1Mbps
    case 4:  // 2Mbps
    case 10: // 5Mbps
    case 22: // 11Mbps
        channelFields.flags |= RadiotapHeader::CHANNEL_FLAG_CCK;
        break;
    default:
        channelFields.flags |= RadiotapHeader::CHANNEL_FLAG_OFDM;
        break;
    }
    if (channelFreqMhz < 2500)
    {
        channelFields.flags |= RadiotapHeader::CHANNEL_FLAG_SPECTRUM_2GHZ;
    }
    else
    {
        channelFields.flags |= RadiotapHeader::CHANNEL_FLAG_SPECTRUM_5GHZ;
    }
    header.SetChannelFields(channelFields);

    if (signalNoise)
    {
        header.SetAntennaSignalPower(signalNoise->signal);
        header.SetAntennaNoisePower(signalNoise->noise);
    }

    if (modClass == WIFI_MOD_CLASS_HT)
    {
        RadiotapHeader::McsFields mcsFields{.mcs = mcs};

        mcsFields.known |= RadiotapHeader::MCS_KNOWN_INDEX | RadiotapHeader::MCS_KNOWN_BANDWIDTH |
                           RadiotapHeader::MCS_KNOWN_GUARD_INTERVAL |
                           RadiotapHeader::MCS_KNOWN_HT_FORMAT | RadiotapHeader::MCS_KNOWN_NESS |
                           RadiotapHeader::MCS_KNOWN_FEC_TYPE | RadiotapHeader::MCS_KNOWN_STBC;

        if (channelWidth == MHz_u{40})
        {
            mcsFields.flags |= RadiotapHeader::MCS_FLAGS_BANDWIDTH_40;
        }

        if (gi.GetNanoSeconds() == 400)
        {
            mcsFields.flags |= RadiotapHeader::MCS_FLAGS_GUARD_INTERVAL;
        }

        const auto ness = txVector.GetNess();
        if (ness & 0x01) // bit 1
        {
            mcsFields.flags |= RadiotapHeader::MCS_FLAGS_NESS_BIT_0;
        }
        if (ness & 0x02) // bit 2
        {
            mcsFields.flags |= RadiotapHeader::MCS_KNOWN_NESS_BIT_1;
        }

        if (txVector.IsStbc())
        {
            mcsFields.flags |= RadiotapHeader::MCS_FLAGS_STBC_STREAMS;
        }

        header.SetMcsFields(mcsFields);
    }

    if (txVector.IsAggregation())
    {
        RadiotapHeader::AmpduStatusFields ampduStatusFields{.referenceNumber = aMpdu.mpduRefNumber};
        ampduStatusFields.flags |= RadiotapHeader::A_MPDU_STATUS_LAST_KNOWN;
        if (aMpdu.type == LAST_MPDU_IN_AGGREGATE || eof)
        {
            ampduStatusFields.flags |= RadiotapHeader::A_MPDU_STATUS_LAST;
        }
        header.SetAmpduStatus(ampduStatusFields);
    }

    if (modClass == WIFI_MOD_CLASS_VHT)
    {
        RadiotapHeader::VhtFields vhtFields{};

        vhtFields.known |= RadiotapHeader::VHT_KNOWN_STBC;
        if (txVector.IsStbc())
        {
            vhtFields.flags |= RadiotapHeader::VHT_FLAGS_STBC;
        }

        vhtFields.known |= RadiotapHeader::VHT_KNOWN_GUARD_INTERVAL;
        if (gi.GetNanoSeconds() == 400)
        {
            vhtFields.flags |= RadiotapHeader::VHT_FLAGS_GUARD_INTERVAL;
        }

        vhtFields.known |=
            RadiotapHeader::VHT_KNOWN_BEAMFORMED | RadiotapHeader::VHT_KNOWN_BANDWIDTH;
        // TODO: bandwidths can be provided with sideband info
        if (channelWidth == MHz_u{40})
        {
            vhtFields.bandwidth = 1;
        }
        else if (channelWidth == MHz_u{80})
        {
            vhtFields.bandwidth = 4;
        }
        else if (channelWidth == MHz_u{160})
        {
            vhtFields.bandwidth = 11;
        }

        // only SU PPDUs are currently supported
        vhtFields.mcsNss.at(0) |= (nss & 0x0f) | ((mcs << 4) & 0xf0);

        header.SetVhtFields(vhtFields);
    }

    if (modClass == WIFI_MOD_CLASS_HE)
    {
        RadiotapHeader::HeFields heFields{};
        heFields.data1 = RadiotapHeader::HE_DATA1_BSS_COLOR_KNOWN |
                         RadiotapHeader::HE_DATA1_DATA_MCS_KNOWN |
                         RadiotapHeader::HE_DATA1_BW_RU_ALLOC_KNOWN;
        if (preamble == WIFI_PREAMBLE_HE_ER_SU)
        {
            heFields.data1 |= RadiotapHeader::HE_DATA1_FORMAT_EXT_SU;
        }
        else if (preamble == WIFI_PREAMBLE_HE_MU)
        {
            heFields.data1 |=
                RadiotapHeader::HE_DATA1_FORMAT_MU | RadiotapHeader::HE_DATA1_SPTL_REUSE2_KNOWN;
        }
        else if (preamble == WIFI_PREAMBLE_HE_TB)
        {
            heFields.data1 |= RadiotapHeader::HE_DATA1_FORMAT_TRIG;
        }

        heFields.data2 = RadiotapHeader::HE_DATA2_GI_KNOWN;
        if (preamble == WIFI_PREAMBLE_HE_MU || preamble == WIFI_PREAMBLE_HE_TB)
        {
            NS_ASSERT(WifiRu::IsHe(txVector.GetHeMuUserInfo(staId).ru));
            auto heRu = std::get<HeRu::RuSpec>(txVector.GetHeMuUserInfo(staId).ru);
            heFields.data2 |=
                RadiotapHeader::HE_DATA2_RU_OFFSET_KNOWN |
                // HeRu indices start at 1 whereas RadioTap starts at 0
                GetRadiotapField(RadiotapHeader::HE_DATA2_RU_OFFSET, heRu.GetIndex() - 1) |
                GetRadiotapField(RadiotapHeader::HE_DATA2_PRISEC_80_SEC, !heRu.GetPrimary80MHz());
        }

        heFields.data3 =
            GetRadiotapField(RadiotapHeader::HE_DATA3_BSS_COLOR, txVector.GetBssColor()) |
            GetRadiotapField(RadiotapHeader::HE_DATA3_DATA_MCS, mcs);

        heFields.data4 = (preamble == WIFI_PREAMBLE_HE_MU)
                             ? GetRadiotapField(RadiotapHeader::HE_DATA4_MU_STA_ID, staId)
                             : 0;

        heFields.data5 = 0;
        if (preamble == WIFI_PREAMBLE_HE_MU || preamble == WIFI_PREAMBLE_HE_TB)
        {
            const auto ruType = WifiRu::GetRuType(txVector.GetHeMuUserInfo(staId).ru);
            switch (ruType)
            {
            case RuType::RU_26_TONE:
                heFields.data5 |= RadiotapHeader::HE_DATA5_DATA_BW_RU_ALLOC_26T;
                break;
            case RuType::RU_52_TONE:
                heFields.data5 |= RadiotapHeader::HE_DATA5_DATA_BW_RU_ALLOC_52T;
                break;
            case RuType::RU_106_TONE:
                heFields.data5 |= RadiotapHeader::HE_DATA5_DATA_BW_RU_ALLOC_106T;
                break;
            case RuType::RU_242_TONE:
                heFields.data5 |= RadiotapHeader::HE_DATA5_DATA_BW_RU_ALLOC_242T;
                break;
            case RuType::RU_484_TONE:
                heFields.data5 |= RadiotapHeader::HE_DATA5_DATA_BW_RU_ALLOC_484T;
                break;
            case RuType::RU_996_TONE:
                heFields.data5 |= RadiotapHeader::HE_DATA5_DATA_BW_RU_ALLOC_996T;
                break;
            case RuType::RU_2x996_TONE:
                heFields.data5 |= RadiotapHeader::HE_DATA5_DATA_BW_RU_ALLOC_2x996T;
                break;
            default:
                NS_ABORT_MSG("Unexpected RU type");
            }
        }
        else if (channelWidth == MHz_u{40})
        {
            heFields.data5 |= RadiotapHeader::HE_DATA5_DATA_BW_RU_ALLOC_40MHZ;
        }
        else if (channelWidth == MHz_u{80})
        {
            heFields.data5 |= RadiotapHeader::HE_DATA5_DATA_BW_RU_ALLOC_80MHZ;
        }
        else if (channelWidth == MHz_u{160})
        {
            heFields.data5 |= RadiotapHeader::HE_DATA5_DATA_BW_RU_ALLOC_160MHZ;
        }
        if (gi.GetNanoSeconds() == 1600)
        {
            heFields.data5 |= RadiotapHeader::HE_DATA5_GI_1_6;
        }
        else if (gi.GetNanoSeconds() == 3200)
        {
            heFields.data5 |= RadiotapHeader::HE_DATA5_GI_3_2;
        }

        header.SetHeFields(heFields);
    }

    if (preamble == WIFI_PREAMBLE_HE_MU)
    {
        RadiotapHeader::HeMuFields heMuFields{};
        // TODO: fill in fields
        header.SetHeMuFields(heMuFields);
        RadiotapHeader::HeMuOtherUserFields heMuOtherUserFields{};
        // TODO: fill in fields
        header.SetHeMuOtherUserFields(heMuOtherUserFields);
    }

    if (IsEht(preamble))
    {
        RadiotapHeader::UsigFields usigFields{};
        usigFields.common = RadiotapHeader::USIG_COMMON_PHY_VER_KNOWN |
                            RadiotapHeader::USIG_COMMON_BW_KNOWN |
                            RadiotapHeader::USIG_COMMON_BSS_COLOR_KNOWN;
        switch (static_cast<uint16_t>(channelWidth))
        {
        case 20:
            usigFields.common |= GetRadiotapField(RadiotapHeader::USIG_COMMON_BW,
                                                  RadiotapHeader::USIG_COMMON_BW_20MHZ);
            break;
        case 40:
            usigFields.common |= GetRadiotapField(RadiotapHeader::USIG_COMMON_BW,
                                                  RadiotapHeader::USIG_COMMON_BW_40MHZ);
            break;
        case 80:
            usigFields.common |= GetRadiotapField(RadiotapHeader::USIG_COMMON_BW,
                                                  RadiotapHeader::USIG_COMMON_BW_80MHZ);
            break;
        case 160:
            usigFields.common |= GetRadiotapField(RadiotapHeader::USIG_COMMON_BW,
                                                  RadiotapHeader::USIG_COMMON_BW_160MHZ);
            break;
        case 320:
            usigFields.common |= GetRadiotapField(RadiotapHeader::USIG_COMMON_BW,
                                                  channelization320MHzMap.at(channelFreqMhz));
            break;
        default:
            NS_ABORT_MSG("Unexpected channel width");
            break;
        }
        usigFields.common |=
            GetRadiotapField(RadiotapHeader::USIG_COMMON_BSS_COLOR, txVector.GetBssColor());
        if (preamble == WIFI_PREAMBLE_EHT_MU)
        {
            usigFields.mask = RadiotapHeader::USIG2_MU_B0_B1_PPDU_TYPE |
                              RadiotapHeader::USIG2_MU_B9_B10_SIG_MCS |
                              RadiotapHeader::USIG2_MU_B3_B7_PUNCTURED_INFO;
            usigFields.value = GetRadiotapField(RadiotapHeader::USIG2_MU_B0_B1_PPDU_TYPE,
                                                txVector.GetEhtPpduType()) |
                               GetRadiotapField(RadiotapHeader::USIG2_MU_B9_B10_SIG_MCS,
                                                txVector.GetSigBMode().GetMcsValue());
            std::optional<bool> isLow80MHz;
            if (txVector.IsDlMu() && channelWidth > MHz_u{80})
            {
                isLow80MHz = true;
                // TODO: fix once EHT RUs are used
            }
            const auto puncturedChannelInfo =
                EhtPpdu::GetPuncturedInfo(txVector.GetInactiveSubchannels(),
                                          txVector.GetEhtPpduType(),
                                          isLow80MHz);
            usigFields.value |= GetRadiotapField(RadiotapHeader::USIG2_MU_B3_B7_PUNCTURED_INFO,
                                                 puncturedChannelInfo);
        }
        else
        {
            usigFields.mask = RadiotapHeader::USIG2_TB_B0_B1_PPDU_TYPE;
            usigFields.value = GetRadiotapField(RadiotapHeader::USIG2_TB_B0_B1_PPDU_TYPE,
                                                txVector.GetEhtPpduType());
        }
        header.SetUsigFields(usigFields);
    }

    if (preamble == WIFI_PREAMBLE_EHT_MU)
    {
        RadiotapHeader::EhtFields ehtFields{};
        ehtFields.known = RadiotapHeader::EHT_KNOWN_GI | RadiotapHeader::EHT_KNOWN_RU_MRU_SIZE_OM |
                          RadiotapHeader::EHT_KNOWN_RU_MRU_INDEX_OM;
        switch (gi.GetNanoSeconds())
        {
        case 800:
            ehtFields.data.at(0) =
                GetRadiotapField(RadiotapHeader::EHT_DATA0_GI, RadiotapHeader::EHT_DATA0_GI_800_NS);
            break;
        case 1600:
            ehtFields.data.at(0) = GetRadiotapField(RadiotapHeader::EHT_DATA0_GI,
                                                    RadiotapHeader::EHT_DATA0_GI_1600_NS);
            break;
        case 3200:
            ehtFields.data.at(0) = GetRadiotapField(RadiotapHeader::EHT_DATA0_GI,
                                                    RadiotapHeader::EHT_DATA0_GI_3200_NS);
            break;
        default:
            NS_ABORT_MSG("Unexpected guard interval");
            break;
        }
        ehtFields.data.at(1) = RadiotapHeader::EHT_DATA1_RU_ALLOC_CC_1_1_1_KNOWN;
        const auto ruType = (txVector.GetEhtPpduType() == 1)
                                ? WifiRu::GetRuType(channelWidth)
                                : WifiRu::GetRuType(txVector.GetRu(staId));
        switch (ruType)
        {
        case RuType::RU_26_TONE:
            ehtFields.data.at(1) |= GetRadiotapField(RadiotapHeader::EHT_DATA1_RU_MRU_SIZE,
                                                     RadiotapHeader::EHT_DATA1_RU_MRU_SIZE_26);
            break;
        case RuType::RU_52_TONE:
            ehtFields.data.at(1) |= GetRadiotapField(RadiotapHeader::EHT_DATA1_RU_MRU_SIZE,
                                                     RadiotapHeader::EHT_DATA1_RU_MRU_SIZE_52);
            break;
        case RuType::RU_106_TONE:
            ehtFields.data.at(1) |= GetRadiotapField(RadiotapHeader::EHT_DATA1_RU_MRU_SIZE,
                                                     RadiotapHeader::EHT_DATA1_RU_MRU_SIZE_106);
            break;
        case RuType::RU_242_TONE:
            ehtFields.data.at(1) |= GetRadiotapField(RadiotapHeader::EHT_DATA1_RU_MRU_SIZE,
                                                     RadiotapHeader::EHT_DATA1_RU_MRU_SIZE_242);
            break;
        case RuType::RU_484_TONE:
            ehtFields.data.at(1) |= GetRadiotapField(RadiotapHeader::EHT_DATA1_RU_MRU_SIZE,
                                                     RadiotapHeader::EHT_DATA1_RU_MRU_SIZE_484);
            break;
        case RuType::RU_996_TONE:
            ehtFields.data.at(1) |= GetRadiotapField(RadiotapHeader::EHT_DATA1_RU_MRU_SIZE,
                                                     RadiotapHeader::EHT_DATA1_RU_MRU_SIZE_996);
            break;
        case RuType::RU_2x996_TONE:
            ehtFields.data.at(1) |= GetRadiotapField(RadiotapHeader::EHT_DATA1_RU_MRU_SIZE,
                                                     RadiotapHeader::EHT_DATA1_RU_MRU_SIZE_2x996);
            break;
        case RuType::RU_4x996_TONE:
            ehtFields.data.at(1) |= GetRadiotapField(RadiotapHeader::EHT_DATA1_RU_MRU_SIZE,
                                                     RadiotapHeader::EHT_DATA1_RU_MRU_SIZE_4x996);
            break;
        default:
            NS_ABORT_MSG("Unexpected RU type");
            break;
        }
        const auto ruIndex =
            (txVector.GetEhtPpduType() == 1) ? 1 : WifiRu::GetIndex(txVector.GetRu(staId));
        const auto& ruAllocation = txVector.GetRuAllocation(p20Index);
        ehtFields.data.at(1) |=
            GetRadiotapField(RadiotapHeader::EHT_DATA1_RU_MRU_INDEX, ruIndex) |
            GetRadiotapField(RadiotapHeader::EHT_DATA1_RU_ALLOC_CC_1_1_1, ruAllocation.at(0));
        if (channelWidth >= MHz_u{40})
        {
            ehtFields.data.at(2) =
                RadiotapHeader::EHT_DATA2_RU_ALLOC_CC_2_1_1_KNOWN |
                GetRadiotapField(RadiotapHeader::EHT_DATA2_RU_ALLOC_CC_2_1_1, ruAllocation.at(1));
        }
        if (channelWidth >= MHz_u{80})
        {
            ehtFields.data.at(2) |=
                RadiotapHeader::EHT_DATA2_RU_ALLOC_CC_1_1_2_KNOWN |
                RadiotapHeader::EHT_DATA2_RU_ALLOC_CC_2_1_2_KNOWN |
                GetRadiotapField(RadiotapHeader::EHT_DATA2_RU_ALLOC_CC_1_1_2, ruAllocation.at(2)) |
                GetRadiotapField(RadiotapHeader::EHT_DATA2_RU_ALLOC_CC_2_1_2, ruAllocation.at(3));
        }
        if (channelWidth >= MHz_u{160})
        {
            ehtFields.data.at(3) =
                RadiotapHeader::EHT_DATA3_RU_ALLOC_CC_1_2_1_KNOWN |
                RadiotapHeader::EHT_DATA3_RU_ALLOC_CC_2_2_1_KNOWN |
                RadiotapHeader::EHT_DATA3_RU_ALLOC_CC_1_2_2_KNOWN |
                GetRadiotapField(RadiotapHeader::EHT_DATA3_RU_ALLOC_CC_1_2_1, ruAllocation.at(4)) |
                GetRadiotapField(RadiotapHeader::EHT_DATA3_RU_ALLOC_CC_2_2_1, ruAllocation.at(5)) |
                GetRadiotapField(RadiotapHeader::EHT_DATA3_RU_ALLOC_CC_1_2_2, ruAllocation.at(6));
            ehtFields.data.at(4) =
                RadiotapHeader::EHT_DATA4_RU_ALLOC_CC_2_2_2_KNOWN |
                GetRadiotapField(RadiotapHeader::EHT_DATA4_RU_ALLOC_CC_2_2_2, ruAllocation.at(7));
            ehtFields.known |= RadiotapHeader::EHT_KNOWN_PRIMARY_80;
            const auto isLowP80 = p20Index < (channelWidth / MHz_u{40});
            ehtFields.data.at(1) |=
                GetRadiotapField(RadiotapHeader::EHT_DATA1_PRIMARY_80,
                                 (isLowP80 ? RadiotapHeader::EHT_DATA1_PRIMARY_80_LOWEST
                                           : RadiotapHeader::EHT_DATA1_PRIMARY_80_HIGHEST));
        }
        if (channelWidth >= MHz_u{320})
        {
            ehtFields.data.at(4) |=
                RadiotapHeader::EHT_DATA4_RU_ALLOC_CC_1_2_3_KNOWN |
                RadiotapHeader::EHT_DATA4_RU_ALLOC_CC_2_2_3_KNOWN |
                GetRadiotapField(RadiotapHeader::EHT_DATA4_RU_ALLOC_CC_1_2_3, ruAllocation.at(8)) |
                GetRadiotapField(RadiotapHeader::EHT_DATA4_RU_ALLOC_CC_2_2_3, ruAllocation.at(9));
            ehtFields.data.at(5) =
                RadiotapHeader::EHT_DATA5_RU_ALLOC_CC_1_2_4_KNOWN |
                RadiotapHeader::EHT_DATA5_RU_ALLOC_CC_2_2_4_KNOWN |
                RadiotapHeader::EHT_DATA5_RU_ALLOC_CC_1_2_5_KNOWN |
                GetRadiotapField(RadiotapHeader::EHT_DATA5_RU_ALLOC_CC_1_2_4, ruAllocation.at(10)) |
                GetRadiotapField(RadiotapHeader::EHT_DATA5_RU_ALLOC_CC_2_2_4, ruAllocation.at(11)) |
                GetRadiotapField(RadiotapHeader::EHT_DATA5_RU_ALLOC_CC_1_2_5, ruAllocation.at(12));
            ehtFields.data.at(6) =
                RadiotapHeader::EHT_DATA6_RU_ALLOC_CC_2_2_5_KNOWN |
                RadiotapHeader::EHT_DATA6_RU_ALLOC_CC_1_2_6_KNOWN |
                RadiotapHeader::EHT_DATA6_RU_ALLOC_CC_2_2_6_KNOWN |
                GetRadiotapField(RadiotapHeader::EHT_DATA6_RU_ALLOC_CC_2_2_5, ruAllocation.at(13)) |
                GetRadiotapField(RadiotapHeader::EHT_DATA6_RU_ALLOC_CC_1_2_6, ruAllocation.at(14)) |
                GetRadiotapField(RadiotapHeader::EHT_DATA6_RU_ALLOC_CC_2_2_6, ruAllocation.at(15));
        }
        uint32_t userInfo = RadiotapHeader::EHT_USER_INFO_STA_ID_KNOWN |
                            RadiotapHeader::EHT_USER_INFO_MCS_KNOWN |
                            RadiotapHeader::EHT_USER_INFO_NSS_KNOWN_O |
                            RadiotapHeader::EHT_USER_INFO_DATA_FOR_USER |
                            GetRadiotapField(RadiotapHeader::EHT_USER_INFO_STA_ID, staId) |
                            GetRadiotapField(RadiotapHeader::EHT_USER_INFO_MCS, mcs) |
                            GetRadiotapField(RadiotapHeader::EHT_USER_INFO_NSS_O, nss);
        ehtFields.userInfo.push_back(userInfo);
        header.SetEhtFields(ehtFields);
    }

    return header;
}

} // namespace ns3
