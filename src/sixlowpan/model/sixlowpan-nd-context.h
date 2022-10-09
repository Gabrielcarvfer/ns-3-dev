/*
 * Copyright (c) 2015 Università di Firenze, Italy
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Alessio Bonadio <alessio.bonadio@gmail.com>
 *         Tommaso Pecorella <tommaso.pecorella@unifi.it>
 *         Adnan Rashid <adnanrashidpk@gmail.com>
 */

#ifndef SIXLOW_ND_CONTEXT_H
#define SIXLOW_ND_CONTEXT_H

#include "ns3/ipv6-address.h"
#include "ns3/nstime.h"
#include "ns3/output-stream-wrapper.h"
#include "ns3/simple-ref-count.h"

#include <stdint.h>

namespace ns3
{

/**
 * \ingroup sixlowpan
 * \brief 6LoWPAN context container for 6LoWPAN ND.
 */
class SixLowPanNdContext : public SimpleRefCount<SixLowPanNdContext>
{
  public:
    /**
     * \brief Constructor.
     */
    SixLowPanNdContext();

    /**
     * \brief Constructor.
     * \param flagC compression flag
     * \param cid context identifier ( 0 <= x <= 15)
     * \param time valid lifetime of context
     * \param context 6LoWPAN context advertised
     */
    SixLowPanNdContext(bool flagC, uint8_t cid, Time time, Ipv6Prefix context);

    /**
     * \brief Destructor.
     */
    ~SixLowPanNdContext();

    /**
     * \brief Get the context length.
     * \return context length value
     */
    uint8_t GetContextLen() const;

    /**
     * \brief Is compression flag ?
     * \return true if context is valid for use in compression, false otherwise
     */
    bool IsFlagC() const;

    /**
     * \brief Set the compression flag.
     * \param c the compression flag
     */
    void SetFlagC(bool c);

    /**
     * \brief Get the context identifier.
     * \return context identifier value
     */
    uint8_t GetCid() const;

    /**
     * \brief Set the context identifier.
     * \param cid the context identifier value
     */
    void SetCid(uint8_t cid);

    /**
     * \brief Get the valid lifetime.
     * \return valid lifetime value
     */
    Time GetValidTime() const;

    /**
     * \brief Set the valid lifetime.
     * \param time the valid lifetime value
     */
    void SetValidTime(Time time);

    /**
     * @brief Set the last update time.
     * @param time the last update time
     */
    void SetLastUpdateTime(Time time);

    /**
     * \brief Get the last update time.
     * \return the last update time
     */
    Time GetLastUpdateTime();

    /**
     * \brief Get the 6LoWPAN context prefix.
     * \return context prefix value
     */
    Ipv6Prefix GetContextPrefix() const;

    /**
     * \brief Set the 6LoWPAN context prefix.
     * \param context the context prefix value
     */
    void SetContextPrefix(Ipv6Prefix context);

    /**
     * \brief Print the 6LoWPAN context.
     * \param stream the ostream the 6LoWPAN context is printed to
     */
    void PrintContext(Ptr<OutputStreamWrapper> stream);

  private:
    /**
     * \brief The compression flag, indicates that this context is valid for use in compression.
     */
    bool m_c;

    /**
     * \brief The context identifier value.
     */
    uint8_t m_cid;

    /**
     * \brief The valid lifetime value.
     */
    Time m_validTime;

    /**
     * \brief The context last update time.
     */
    Time m_lastUpdateTime;

    /**
     * \brief The context prefix value.
     */
    Ipv6Prefix m_context;
};

} /* namespace ns3 */

#endif /* SIXLOW_ND_CONTEXT_H */
