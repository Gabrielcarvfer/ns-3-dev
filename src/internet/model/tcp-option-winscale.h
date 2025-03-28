/*
 * Copyright (c) 2011 Adrian Sai-wah Tam
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Adrian Sai-wah Tam <adrian.sw.tam@gmail.com>
 * Documentation, test cases: Natale Patriciello <natale.patriciello@gmail.com>
 */

#ifndef TCP_OPTION_WINSCALE_H
#define TCP_OPTION_WINSCALE_H

#include "tcp-option.h"

namespace ns3
{

/**
 * @ingroup tcp
 *
 * @brief Defines the TCP option of kind 3 (window scale option) as in \RFC{1323}
 *
 * For more efficient use of high bandwidth networks, a larger TCP window size
 * may be used. The TCP window size field controls the flow of data and its
 * value is limited to between 2 and 65,535 bytes.
 *
 * Since the size field cannot be expanded, a scaling factor is used.
 * The TCP window scale option, as defined in \RFC{1323}, is an option used
 * to increase the maximum window size from 65,535 bytes to 1 gigabyte.
 * Scaling up to larger window sizes is a part of what is necessary for TCP Tuning.
 *
 * The window scale option is used only during the TCP 3-way handshake.
 * The window scale value represents the number of bits to left-shift the
 * 16-bit window size field. The window scale value can be set from 0
 * (no shift) to 14 for each direction independently. Both sides must
 * send the option in their SYN segments to enable window scaling in
 * either direction.
 */
class TcpOptionWinScale : public TcpOption
{
  public:
    /**
     * @brief Get the type ID.
     * @return the object TypeId
     */
    static TypeId GetTypeId();

    TcpOptionWinScale();
    ~TcpOptionWinScale() override;

    void Print(std::ostream& os) const override;
    void Serialize(Buffer::Iterator start) const override;
    uint32_t Deserialize(Buffer::Iterator start) override;

    uint8_t GetKind() const override;
    uint32_t GetSerializedSize() const override;

    /**
     * @brief Get the scale value (uint8_t)
     * @return The scale value
     */
    uint8_t GetScale() const;

    /**
     * @brief Set the scale option
     *
     * The scale option SHOULD be <= 14 (as \RFC{1323}).
     *
     * @param scale Scale factor
     */
    void SetScale(uint8_t scale);

  protected:
    uint8_t m_scale; //!< Window scaling in number of bit shift
};

} // namespace ns3

#endif /* TCP_OPTION_WINSCALE */
