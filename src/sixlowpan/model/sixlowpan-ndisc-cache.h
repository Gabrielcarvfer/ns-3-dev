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
#ifndef SIXLOW_NDISC_CACHE_H
#define SIXLOW_NDISC_CACHE_H

#include "ns3/ipv6-address.h"
#include "ns3/mac64-address.h"
#include "ns3/ndisc-cache.h"
#include "ns3/output-stream-wrapper.h"
#include "ns3/ptr.h"
#include "ns3/timer.h"

namespace ns3
{

/**
 * \ingroup sixlowpan
 * \class SixLowPanNdiscCache
 * \brief Neighbor Discovery cache for 6LoWPAN ND. Keeps also RAs, prefixes and contexts.
 */
class SixLowPanNdiscCache : public virtual NdiscCache
{
  public:
    class SixLowPanEntry;

    /**
     * \brief Get the type ID
     * \return type ID
     */
    static TypeId GetTypeId();

    /**
     * \brief Constructor.
     */
    SixLowPanNdiscCache();

    /**
     * \brief Destructor.
     */
    ~SixLowPanNdiscCache() override;

    /**
     * \brief Lookup in the cache.
     * \param dst destination address
     * \return the entry if found, 0 otherwise
     */
    NdiscCache::Entry* Lookup(Ipv6Address dst) override;

    /**
     * \brief Add an entry.
     * \param to address to add
     * \return an new Entry
     */
    NdiscCache::Entry* Add(Ipv6Address to) override;

    /**
     * \brief Print the SixLowPanNdisc cache entries
     *
     * \param stream the ostream the SixLowPanNdisc cache entries is printed to
     */
    void PrintNdiscCache(Ptr<OutputStreamWrapper> stream) override;

    /**
     * \class SixLowPanEntry
     * \brief A record that holds information about an SixLowPanNdiscCache entry.
     */
    class SixLowPanEntry : public NdiscCache::Entry
    {
      public:
        /**
         * \brief Constructor.
         * \param nd The NdiscCache this entry belongs to.
         */
        SixLowPanEntry(NdiscCache* nd);

        void Print(std::ostream& os) const override;

        /**
         * \brief Changes the state to this entry to REGISTERED.
         * It starts the registered timer.
         * \param time the lifetime in units of 60 seconds (from ARO)
         */
        void MarkRegistered(uint16_t time);

        /**
         * \brief Changes the state to this entry to TENTATIVE.
         * It starts the tentative timer (20 seconds).
         */
        void MarkTentative();

        /**
         * \brief Change the state to this entry to GARBAGE.
         */
        void MarkGarbage();

        /**
         * \brief Is the entry REGISTERED.
         * \return true if the type of entry is REGISTERED, false otherwise
         */
        bool IsRegistered() const;

        /**
         * \brief Is the entry TENTATIVE.
         * \return true if the type of entry is TENTATIVE, false otherwise
         */
        bool IsTentative() const;

        /**
         * \brief Is the entry GARBAGE-COLLECTIBLE.
         * \return true if the type of entry GARBAGE-COLLECTIBLE, false otherwise
         */
        bool IsGarbage() const;

        /**
         * \brief Function called when timer timeout.
         */
        void FunctionTimeout();

        /**
         * \brief Get the ROVR field.
         * \return the ROVR
         */
        std::vector<uint8_t> GetRovr() const;

        /**
         * \brief Set the ROVR field.
         * \param rovr the ROVR value
         */
        void SetRovr(const std::vector<uint8_t>& rovr);

      private:
        /**
         * \brief The SixLowPanEntry type enumeration.
         */
        enum SixLowPanNdiscCacheEntryType_e
        {
            REGISTERED, /**< Have an explicit registered lifetime */
            TENTATIVE,  /**< Have a short lifetime, typically get converted to REGISTERED */
            GARBAGE     /**< Allow for garbage collection when low on memory */
        };

        /**
         * \brief The ROVR value.
         */
        std::vector<uint8_t> m_rovr;

        /**
         * \brief The state of the entry.
         */
        SixLowPanNdiscCacheEntryType_e m_type;

        /**
         * \brief Timer (used for REGISTERED entries).
         */
        Timer m_registeredTimer;

        /**
         * \brief Timer (used for TENTATIVE entries).
         */
        Timer m_tentativeTimer;
    };

  protected:
    /**
     * \brief Dispose this object.
     */
    void DoDispose() override;

  private:
    /**
     * \brief 6LoWPAN Neighbor Discovery Cache container
     */
    typedef std::unordered_map<Ipv6Address, SixLowPanNdiscCache::SixLowPanEntry*, Ipv6AddressHash>
        SixLowPanCache;

    /**
     * \brief 6LoWPAN Neighbor Discovery Cache container iterator
     */
    typedef std::unordered_map<Ipv6Address, SixLowPanNdiscCache::SixLowPanEntry*, Ipv6AddressHash>::
        iterator SixLowPanCacheI;
};

/**
 * \brief Stream insertion operator.
 *
 * \param os the reference to the output stream
 * \param entry the SixLowPanNdiscCache::SixLowPanEntry
 * \returns the reference to the output stream
 */
std::ostream& operator<<(std::ostream& os, const SixLowPanNdiscCache::SixLowPanEntry& entry);

} /* namespace ns3 */

#endif /* SIXLOW_NDISC_CACHE_H */
