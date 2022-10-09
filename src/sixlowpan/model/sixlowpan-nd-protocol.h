/*
 * Copyright (c) 2020 Università di Firenze, Italy
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

#ifndef SIXLOWPAN_ND_PROTOCOL_H
#define SIXLOWPAN_ND_PROTOCOL_H

#include "sixlowpan-nd-header.h"

#include "ns3/icmpv6-l4-protocol.h"
#include "ns3/ip-l4-protocol.h"
#include "ns3/ipv6-address.h"
#include "ns3/lollipop-counter.h"

#include <utility>

namespace ns3
{

class SixLowPanNdiscCache;
class SixLowPanNdPrefix;
class SixLowPanNdContext;
class SixLowPanNetDevice;

/**
 * @ingroup sixlowpan
 * @brief An optimization of the ND protocol for 6LoWPANs.
 */
class SixLowPanNdProtocol : public Icmpv6L4Protocol
{
  public:
    /**
     * 6LoWPAN-ND EARO registration status codes
     */
    enum RegStatus_e
    {
        SUCCESS = 0x0,       //!< Success
        DUPLICATE_ADDRESS,   //!< Duplicate Address
        NEIGHBOR_CACHE_FULL, //!< Neighbor Cache Full
        MOVED,               //!< Registration failed because it is not the most recent
        REMOVED,             //!< Binding state was removed.
        VALIDATION_REQUEST, //!< Registering Node is challenged for owning the Registered Address or
                            //!< for being an acceptable proxy for the registration.
        DUPLICATE_SOURCE_ADDRESS, //!< Address used as the source of the NS(EARO) conflicts with an
                                  //!< existing registration.
        INVALID_SOURCE_ADDRESS, //!< Address used as the source of the NS(EARO) is not a Link-Local
                                //!< Address.
        REGISTERED_ADDRESS_TOPOLOGICALLY_INCORRECT, //!< Address being registered is not usable on
                                                    //!< this link.
        SIXLBR_REGISTRY_SATURATED,                  //!< 6LBR Registry is saturated.
        VALIDATION_FAILED //!< The proof of ownership of the Registered Address is not correct.
    };

    /**
     * @brief 6LBR constants : min context change delay.
     */
    static const uint16_t MIN_CONTEXT_CHANGE_DELAY;

    /**
     * @brief 6LR constants : max RA transmission.
     */
    static const uint8_t MAX_RTR_ADVERTISEMENTS;

    /**
     * @brief 6LR constants : min delay between RA.
     */
    static const uint8_t MIN_DELAY_BETWEEN_RAS;

    /**
     * @brief 6LR constants : max delay between RA.
     */
    static const uint8_t MAX_RA_DELAY_TIME;

    /**
     * @brief 6LR constants : tentative neighbor cache entry lifetime.
     */
    static const uint8_t TENTATIVE_NCE_LIFETIME;

    /**
     * @brief router constants : multihop hoplimit.
     */
    static const uint8_t MULTIHOP_HOPLIMIT;

    /**
     * @brief Constructor.
     */
    SixLowPanNdProtocol();

    /**
     * @brief Destructor.
     */
    ~SixLowPanNdProtocol() override;

    /**
     * @brief Get the type ID.
     * @return The object TypeId.
     */
    static TypeId GetTypeId();

    TypeId GetInstanceTypeId() const override;
    void DoInitialize() override;
    void NotifyNewAggregate() override;

    /**
     * Assign a fixed random variable stream number to the random variables
     * used by this model.  Return the number of streams (possibly zero) that
     * have been assigned.
     *
     * @param stream first stream index to use
     * @return the number of stream indices assigned by this model
     */
    int64_t AssignStreams(int64_t stream);

    enum IpL4Protocol::RxStatus Receive(Ptr<Packet> p,
                                        const Ipv6Header& header,
                                        Ptr<Ipv6Interface> interface) override;

    Ptr<NdiscCache> CreateCache(Ptr<NetDevice> device, Ptr<Ipv6Interface> interface) override;

    bool Lookup(Ptr<Packet> p,
                const Ipv6Header& ipHeader,
                Ipv6Address dst,
                Ptr<NetDevice> device,
                Ptr<NdiscCache> cache,
                Address* hardwareDestination) override;

    void FunctionDadTimeout(Ipv6Interface* interface, Ipv6Address addr) override;

    /**
     * @brief Send a NS for 6LoWPAN ND (+ EARO, SLLAO).
     * @param addrToRegister source IPv6 address
     * @param dst destination IPv6 address
     * @param dstMac destination MAC address
     * @param time registration lifetime (EARO)
     * @param rovr ROVR (EARO)
     * @param tid TID (EARO)
     * @param sixDevice SixLowPan NetDevice
     */
    void SendSixLowPanNsWithEaro(Ipv6Address addrToRegister,
                                 Ipv6Address dst,
                                 Address dstMac,
                                 uint16_t time,
                                 const std::vector<uint8_t>& rovr,
                                 uint8_t tid,
                                 Ptr<NetDevice> sixDevice);

    /**
     * @brief Send a NA for 6LoWPAN ND (+ EARO).
     * @param src source IPv6 address
     * @param dst destination IPv6 address
     * @param target target IPv6 address
     * @param time registration lifetime (EARO)
     * @param rovr ROVR (EARO)
     * @param tid TID (EARO)
     * @param sixDevice SixLowPan NetDevice
     * @param status status (EARO)
     */
    void SendSixLowPanNaWithEaro(Ipv6Address src,
                                 Ipv6Address dst,
                                 Ipv6Address target,
                                 uint16_t time,
                                 const std::vector<uint8_t>& rovr,
                                 uint8_t tid,
                                 Ptr<NetDevice> sixDevice,
                                 uint8_t status);

    /**
     * @brief Send a RA for 6LoWPAN ND (+ PIO, 6CO, ABRO, SLLAO).
     * @param src source IPv6 address
     * @param dst destination IPv6 address
     * @param interface the interface from which the packet will be sent
     */
    void SendSixLowPanRA(Ipv6Address src, Ipv6Address dst, Ptr<Ipv6Interface> interface);

    /*
     * @brief Send a DAR for 6LoWPAN ND.
     * @param src source IPv6 address
     * @param dst destination IPv6 address
     * @param time registration lifetime
     * @param eui EUI-64
     * @param registered registered IPv6 address
     */
    //  void SendSixLowPanDAR (Ipv6Address src, Ipv6Address dst, uint16_t time, Mac64Address eui,
    //                         Ipv6Address registered);

    /**
     * @brief Function called to send RS + SLLAO.
     * @param src source IPv6 address
     * @param dst destination IPv6 address
     * @param linkAddr link-layer address (SLLAO)
     * @param retransmission RS retransmission number
     * @param retransmissionInterval RS retransmission interval
     */
    void RetransmitRS(Ipv6Address src,
                      Ipv6Address dst,
                      Address linkAddr,
                      uint8_t retransmission,
                      Time retransmissionInterval);

    /**
     * @brief Set an interface to be used as a 6LBR
     * @param device device to be used for announcement
     */
    void SetInterfaceAs6lbr(Ptr<SixLowPanNetDevice> device);

    /**
     * @brief Set a prefix to be announced on an interface (6LBR)
     * @param device device to be used for announcement
     * @param prefix announced prefix
     */
    void SetAdvertisedPrefix(Ptr<SixLowPanNetDevice> device, Ipv6Prefix prefix);

    /**
     * @brief Add a context to be advertised on an interface (6LBR)
     * @param device device to be used for advertisement
     * @param context advertised context
     */
    void AddAdvertisedContext(Ptr<SixLowPanNetDevice> device, Ipv6Prefix context);

    /**
     * @brief Remove a context to be advertised on an interface (6LBR)
     * @param device device to be used for advertisement
     * @param context advertised context
     */
    void RemoveAdvertisedContext(Ptr<SixLowPanNetDevice> device, Ipv6Prefix context);

    /**
     * @brief Checks if an interface is set as 6LBR
     * @param device the interface to check
     * @return true if the interface is configured as a 6LBR
     */
    bool IsBorderRouterOnInterface(Ptr<SixLowPanNetDevice> device) const;

    /**
     * @brief Checks if an address registration is in progress
     * @return true if an address registration is in progress
     */
    bool IsAddressRegistrationInProgress() const;

  protected:
    /**
     * @brief Dispose this object.
     */
    void DoDispose() override;

  private:
    /**
     * @brief RA advertised from routers for 6LoWPAN ND.
     */
    class SixLowPanRaEntry : public SimpleRefCount<SixLowPanRaEntry>
    {
      public:
        SixLowPanRaEntry();

        /**
         * Creates a RA entry from the RA header contents
         * @param raHeader the RA header
         * @param abroHdr the ABRO header
         * @param contextList the list of Contexts
         * @param prefixList the list of PIOs
         */
        SixLowPanRaEntry(Icmpv6RA raHeader,
                         Icmpv6OptionSixLowPanAuthoritativeBorderRouter abroHdr,
                         std::list<Icmpv6OptionSixLowPanContext> contextList,
                         std::list<Icmpv6OptionPrefixInformation> prefixList);

        ~SixLowPanRaEntry();

        /**
         * @brief Get the prefixes advertised for this interface.
         * @return a list of IPv6 prefixes
         */
        std::list<Ptr<SixLowPanNdPrefix>> GetPrefixes() const;

        /**
         * @brief Add a prefix to advertise on interface.
         * @param prefix prefix to advertise
         */
        void AddPrefix(Ptr<SixLowPanNdPrefix> prefix);

        /**
         * @brief Remove a prefix from the ones advertised on interface.
         * @param prefix prefix to remove from advertisements
         */
        void RemovePrefix(Ptr<SixLowPanNdPrefix> prefix);

        /**
         * @brief Get list of 6LoWPAN contexts advertised for this interface.
         * @return list of 6LoWPAN contexts
         */
        std::map<uint8_t, Ptr<SixLowPanNdContext>> GetContexts() const;

        /**
         * @brief Add a 6LoWPAN context to advertise on interface.
         * @param context 6LoWPAN context to advertise
         */
        void AddContext(Ptr<SixLowPanNdContext> context);

        /**
         * @brief Remove a 6LoWPAN context.
         * @param context 6LoWPAN context to remove
         */
        void RemoveContext(Ptr<SixLowPanNdContext> context);

        /**
         * Builds an Icmpv6RA from the stored data.
         * @return the Icmpv6RA.
         */
        Icmpv6RA BuildRouterAdvertisementHeader();

        /**
         * Builds a container of Icmpv6OptionPrefixInformation from the stored data.
         * @return the Icmpv6OptionPrefixInformation container.
         */
        std::list<Icmpv6OptionPrefixInformation> BuildPrefixInformationOptions();

        /**
         * @brief Is managed flag enabled ?
         * @return managed flag
         */
        bool IsManagedFlag() const;

        /**
         * @brief Set managed flag
         * @param managedFlag value
         */
        void SetManagedFlag(bool managedFlag);

        /**
         * @brief Is "other config" flag enabled ?
         * @return other config flag
         */
        bool IsOtherConfigFlag() const;

        /**
         * @brief Is "home agent" flag enabled ?
         * @return "home agent" flag
         */
        bool IsHomeAgentFlag() const;

        /**
         * @brief Set "home agent" flag.
         * @param homeAgentFlag value
         */
        void SetHomeAgentFlag(bool homeAgentFlag);

        /**
         * @brief Set "other config" flag
         * @param otherConfigFlag value
         */
        void SetOtherConfigFlag(bool otherConfigFlag);

        /**
         * @brief Get reachable time.
         * @return reachable time
         */
        uint32_t GetReachableTime() const;

        /**
         * @brief Set reachable time.
         * @param time reachable time
         */
        void SetReachableTime(uint32_t time);

        /**
         * @brief Get router lifetime.
         * @return router lifetime
         */
        uint32_t GetRouterLifeTime() const;

        /**
         * @brief Set router lifetime.
         * @param time router lifetime
         */
        void SetRouterLifeTime(uint32_t time);

        /**
         * @brief Get retransmission timer.
         * @return retransmission timer
         */
        uint32_t GetRetransTimer() const;

        /**
         * @brief Set retransmission timer.
         * @param timer retransmission timer
         */
        void SetRetransTimer(uint32_t timer);

        /**
         * @brief Get current hop limit.
         * @return current hop limit for the link
         */
        uint8_t GetCurHopLimit() const;

        /**
         * @brief Set current hop limit.
         * @param curHopLimit current hop limit for the link
         */
        void SetCurHopLimit(uint8_t curHopLimit);

        /**
         * @brief Get version value (ABRO).
         * @return the version value
         */
        uint32_t GetAbroVersion() const;

        /**
         * @brief Set version value (ABRO).
         * @param version the version value
         */
        void SetAbroVersion(uint32_t version);

        /**
         * @brief Get valid lifetime value (ABRO).
         * @return the valid lifetime (units of 60 seconds)
         */
        uint16_t GetAbroValidLifeTime() const;

        /**
         * @brief Set valid lifetime value (ABRO).
         * @param time the valid lifetime (units of 60 seconds)
         */
        void SetAbroValidLifeTime(uint16_t time);

        /**
         * @brief Get Border Router address (ABRO).
         * @return the Border Router address
         */
        Ipv6Address GetAbroBorderRouterAddress() const;

        /**
         * @brief Set Border Router address (ABRO).
         * @param border the Border Router address
         */
        void SetAbroBorderRouterAddress(Ipv6Address border);

        /**
         * @brief Parse an ABRO and records the appropriate params.
         * @param abro the Authoritative Border Router Option header
         * @return true if the parsing was correct.
         */
        bool ParseAbro(Icmpv6OptionSixLowPanAuthoritativeBorderRouter abro);

        /**
         * @brief Build an ABRO header.
         * @return the Authoritative Border Router Option header
         */
        Icmpv6OptionSixLowPanAuthoritativeBorderRouter MakeAbro();

      private:
        /**
         * @brief Advertised Prefixes.
         */
        std::list<Ptr<SixLowPanNdPrefix>> m_prefixes;

        /**
         * @brief List of 6LoWPAN contexts advertised.
         */
        std::map<uint8_t, Ptr<SixLowPanNdContext>> m_contexts;

        /**
         * @brief Managed flag. If true host use the stateful protocol for address
         * autoconfiguration.
         */
        bool m_managedFlag;

        /**
         * @brief Other configuration flag. If true host use stateful protocol for other
         * (non-address) information.
         */
        bool m_otherConfigFlag;

        /**
         * @brief Flag to add HA (home agent) flag in RA.
         */
        bool m_homeAgentFlag;

        /**
         * @brief Reachable time in milliseconds.
         */
        uint32_t m_reachableTime;

        /**
         * @brief Retransmission timer in milliseconds.
         */
        uint32_t m_retransTimer;

        /**
         * @brief Current hop limit (TTL).
         */
        uint32_t m_curHopLimit;

        /**
         * @brief Router life time in seconds.
         */
        uint32_t m_routerLifeTime;

        /**
         * @brief Version value for ABRO.
         */
        uint32_t m_abroVersion;

        /**
         * @brief Valid lifetime value for ABRO (units of 60 seconds).
         */
        uint16_t m_abroValidLifeTime;

        /**
         * @brief Border Router address for ABRO.
         */
        Ipv6Address m_abroBorderRouter;

        /**
         * Neighbors that are announcing this RA.
         * Pair of Ipv6Address of the neighbor, Time of the last RA received
         */
        std::map<Ipv6Address, Time> m_neighbors;
    };

    /**
     *  Role of the node: 6LN, 6LR, 6LBR
     *
     *  A Node starts either as a 6LN or as a 6LBR.
     *  A 6LN can become a 6LR.
     *
     */
    enum SixLowPanNodeStatus_e
    {
        SixLowPanNodeOnly,    //!< a 6LN that can not become a 6LR
        SixLowPanNode,        //!< a 6LN that can (and want to) become a 6LR
        SixLowPanRouter,      //!< a 6LR
        SixLowPanBorderRouter //!< a 6LBR
    };

    /**
     * @brief The amount of time (units of 60 seconds) that the router should retain the NCE for the
     * node.
     */
    uint16_t m_regTime;

    /**
     * @brief The advance to perform maintaining of RA's information and registration.
     */
    uint16_t m_advance;

    /**
     * @brief Status of the node.
     */
    SixLowPanNodeStatus_e m_nodeRole;

    Time m_routerLifeTime;       //!< Default Router Lifetime.
    Time m_pioPreferredLifeTime; //!< Default Prefix Information Preferred Lifetime.
    Time m_pioValidLifeTime;     //!< Default Prefix Information Valid Lifetime.
    Time m_contextValidLifeTime; //!< Default Context Valid Lifetime.
    Time m_abroValidLifeTime;    //!< Default ABRO Valid Lifetime.

    /**
     * @brief Random jitter before sending address registrations.
     */
    Ptr<RandomVariableStream> m_addressRegistrationJitter;

    /**
     * @brief Receive NS for 6LoWPAN ND method.
     * @param packet the packet.
     * @param src source address.
     * @param dst destination address.
     * @param interface the interface from which the packet is coming.
     */
    void HandleSixLowPanNS(Ptr<Packet> packet,
                           const Ipv6Address& src,
                           const Ipv6Address& dst,
                           Ptr<Ipv6Interface> interface);

    /**
     * @brief Receive NA for 6LoWPAN ND method.
     * @param packet the packet
     * @param src source address
     * @param dst destination address
     * @param interface the interface from which the packet is coming
     */
    void HandleSixLowPanNA(Ptr<Packet> packet,
                           const Ipv6Address& src,
                           const Ipv6Address& dst,
                           Ptr<Ipv6Interface> interface);

    /**
     * @brief Receive RS for 6LoWPAN ND method.
     * @param packet the packet
     * @param src source address
     * @param dst destination address
     * @param interface the interface from which the packet is coming
     */
    void HandleSixLowPanRS(Ptr<Packet> packet,
                           const Ipv6Address& src,
                           const Ipv6Address& dst,
                           Ptr<Ipv6Interface> interface);

    /**
     * @brief Receive RA for 6LoWPAN ND method.
     * @param packet the packet
     * @param src source address
     * @param dst destination address
     * @param interface the interface from which the packet is coming
     */
    void HandleSixLowPanRA(Ptr<Packet> packet,
                           const Ipv6Address& src,
                           const Ipv6Address& dst,
                           Ptr<Ipv6Interface> interface);

    /*
     * @brief Receive DAC for 6LoWPAN ND method.
     * @param packet the packet
     * @param src source address
     * @param dst destination address
     * @param interface the interface from which the packet is coming
     */
    // void HandleSixLowPanDAC (Ptr<Packet> packet, Ipv6Address const &src, Ipv6Address const &dst,
    //                         Ptr<Ipv6Interface> interface);

    /**
     * Creates a ROVR for the NetDevice
     *
     * @param device device to create the ROVR for
     */
    void BuildRovrForDevice(Ptr<NetDevice> device);

    /**
     * Check an RA for consistency with the ones in the RA cache
     * @param ra the RA to check
     * @return true if the RA is not consistent (must be discarded)
     */
    bool ScreeningRas(Ptr<SixLowPanRaEntry> ra);

    /**
     * Address re-registration procedure
     */
    void AddressReRegistration();

    /**
     * Address registration procedure
     */
    void AddressRegistration();

    /**
     * Address registration Success or Failure
     *
     * @param registrar node that performs the registration. Registered Node or a proxy.
     * @param tid Transaction ID
     */
    void AddressRegistrationSuccess(Ipv6Address registrar, LollipopCounter8 tid);

    /**
     * Address registration timeout handler
     */
    void AddressRegistrationTimeout();

    uint8_t m_addressRegistrationCounter; //!< Number of retries of an address registration.

    std::map<Ptr<NetDevice>, std::vector<uint8_t>> m_rovrContainer; //!< Container of ROVRs

    EventId m_addressRegistrationTimeoutEvent; //!< Address Registration timeout event.
    EventId m_addressRegistrationEvent;        //!< Address Registration event.
    EventId m_addressReRegistrationEvent;      //!< Address ReRegistration event.

    /**
     * Container of TIDs for each pair <Registered LinkLocal Address, Registrar address>
     * For LLaddr, the registrar is the node we're registering to. For Gaddr, it is "::".
     */
    std::map<std::pair<Ipv6Address, Ipv6Address>, LollipopCounter8> m_tidContainer;

    std::map<Ipv6Address, Time> m_neighborBlacklist; //!< Blacklist of the neighbors that didn't
                                                     //!< allow registration or didn't reply.

    std::map<Ipv6Address, Ptr<SixLowPanRaEntry>>
        m_raCache; //!< Router Advertisement cached entries (if the node is a 6L*).

    std::map<Ptr<SixLowPanNetDevice>, Ptr<SixLowPanRaEntry>>
        m_raEntries; //!< Router Advertisement entries (if the node is a 6LBR).

    /**
     * Structure holding data about a pending RA being processed
     */
    typedef struct
    {
        Ptr<SixLowPanRaEntry> pendingRa; //!< RA being processed
        Ipv6Address source;              //!< Origin of the RA (might be a 6LR)
        Icmpv6OptionLinkLayerAddress
            llaHdr; //!< Link-Layer address option from the RA (can be 6LR or 6LBR).
        Ptr<Ipv6Interface> incomingIf;                  //!< Interface that did receive the RA
        std::list<Ipv6Address> addressesToBeregistered; //!< Addresses pending registration.
        std::map<Ipv6Address, Icmpv6OptionPrefixInformation>
            prefixForAddress; //!< Prefixes used to build global addresses.
    } SixLowPanPendingRa;

    std::list<SixLowPanPendingRa>
        m_pendingRas; //!< RA waiting for processing (addresses registration).

    /**
     * Structure holding data about registered addresses
     */
    typedef struct
    {
        Time registrationTimeout;   //!< Registration expiration time
        Ipv6Address registeredAddr; //!< Registered address
        Ipv6Address abroAddress;    //!< Address of the ABRO (always global)
        Ipv6Address registrar; //!< Registering node (lladdr of 6LR or 6LBR for a lladdr, gaddr of
                               //!< 6LBR for gaddr)
        Address registrarMacAddr;     //!< Registering node MAC address
        Ptr<Ipv6Interface> interface; //!< Interface used for the registration
    } SixLowPanRegisteredAddress;

    std::list<SixLowPanRegisteredAddress>
        m_registeredAddresses; //!< Addresses that have been registered.

    /**
     * Structure holding data of the address being registered
     */
    typedef struct
    {
        bool isValid; //!< The data are valid (for timeouts and retransmissions)
        Ipv6Address addressPendingRegistration; //!< Address being Registered
        Ipv6Address abroAddress;                //!< Address of the ABRO (always global)
        Ipv6Address registrar;                  //!< Registering node address (always link-local)
        Address registrarMacAddr;               //!< Registering node MAC address
        bool newRegistration;     //!< new registration (true) or re-registration (false)
        Ptr<NetDevice> sixDevice; //!< The SixLowPanNetDevice to use for the registration
    } AddressPendingRegistration;

    AddressPendingRegistration m_addrPendingReg; //!< Address being Registered

    // RS retry backoff
    //  Time m_rtrSolicitationInterval;         //!< RS Retransmission interval
    Time m_maxRtrSolicitationInterval; //!< Maximum RS Retransmission interval
    //  Time m_currentRtrSolicitationInterval;  //!< Current RS Retransmission interval
    //  Ptr<UniformRandomVariable> m_rsRetransmissionDelay; //!< Random variable for RS
    //  retransmissions.
};

} /* namespace ns3 */

#endif
