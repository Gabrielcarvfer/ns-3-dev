/*
 * Copyright (c) 2005,2006 INRIA
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */

#ifndef CHANNEL_ACCESS_MANAGER_H
#define CHANNEL_ACCESS_MANAGER_H

#include "wifi-phy-common.h"
#include "wifi-phy-operating-channel.h"

#include "ns3/event-id.h"
#include "ns3/nstime.h"
#include "ns3/object.h"
#include "ns3/traced-callback.h"

#include <algorithm>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

class EmlsrUlTxopTest;
class EmlsrCcaBusyTest;

namespace ns3
{

class WifiPhy;
class PhyListener;
class Txop;
class FrameExchangeManager;
class WifiTxVector;
enum AcIndex : uint8_t; // opaque enum declaration

/**
 * @brief Enumeration values for the outcome of the check whether channel access is expected to be
 *        gained within a given time interval
 * @see ChannelAccessManager::GetExpectedAccessWithin
 * @ingroup wifi
 */
enum class WifiExpectedAccessReason : uint8_t
{
    ACCESS_EXPECTED = 0,
    NOT_REQUESTED,
    NOTHING_TO_TX,
    RX_END,
    BUSY_END,
    TX_END,
    NAV_END,
    ACK_TIMER_END,
    CTS_TIMER_END,
    SWITCHING_END,
    NO_PHY_END,
    SLEEP_END,
    OFF_END,
    BACKOFF_END
};

/**
 * @brief Manage a set of ns3::Txop
 * @ingroup wifi
 *
 * Handle a set of independent ns3::Txop, each of which represents
 * a single DCF within a MAC stack. Each ns3::Txop has a priority
 * implicitly associated with it (the priority is determined when the
 * ns3::Txop is added to the ChannelAccessManager: the first Txop to be
 * added gets the highest priority, the second, the second highest
 * priority, and so on.) which is used to handle "internal" collisions.
 * i.e., when two local Txop are expected to get access to the
 * medium at the same time, the highest priority local Txop wins
 * access to the medium and the other Txop suffers a "internal"
 * collision.
 */
class ChannelAccessManager : public Object
{
    /// Allow test cases to access private members
    friend class ::EmlsrUlTxopTest;
    friend class ::EmlsrCcaBusyTest;

  public:
    ChannelAccessManager();
    ~ChannelAccessManager() override;

    /**
     * @brief Get the type ID.
     * @return the object TypeId
     */
    static TypeId GetTypeId();

    /**
     * Set up (or reactivate) listener for PHY events on the given PHY. The new (or reactivated)
     * listener becomes the active listener and the previous active listener attached to another
     * PHY, if any, is deactivated.
     *
     * @param phy the WifiPhy to listen to
     */
    void SetupPhyListener(Ptr<WifiPhy> phy);
    /**
     * Remove current registered listener for PHY events on the given PHY.
     *
     * @param phy the WifiPhy to listen to
     */
    void RemovePhyListener(Ptr<WifiPhy> phy);
    /**
     * Deactivate current registered listener for PHY events on the given PHY. All notifications
     * but channel switch notifications coming from an inactive listener are ignored.
     *
     * @param phy the WifiPhy to listen to
     */
    void DeactivatePhyListener(Ptr<WifiPhy> phy);
    /**
     * Set the ID of the link this Channel Access Manager is associated with.
     *
     * @param linkId the ID of the link this Channel Access Manager is associated with
     */
    void SetLinkId(uint8_t linkId);
    /**
     * Set up the Frame Exchange Manager.
     *
     * @param feManager the Frame Exchange Manager
     */
    void SetupFrameExchangeManager(Ptr<FrameExchangeManager> feManager);

    /**
     * @param txop a new Txop.
     *
     * The ChannelAccessManager does not take ownership of this pointer so, the callee
     * must make sure that the Txop pointer will stay valid as long
     * as the ChannelAccessManager is valid. Note that the order in which Txop
     * objects are added to a ChannelAccessManager matters: the first Txop added
     * has the highest priority, the second Txop added, has the second
     * highest priority, etc.
     */
    void Add(Ptr<Txop> txop);

    /**
     * Determine if a new backoff needs to be generated as per letter a) of Section 10.23.2.2
     * of IEEE 802.11-2020 ("EDCA backoff procedure"). This method is called upon the occurrence
     * of events such as the enqueuing of a packet or the unblocking of some links after they
     * have been blocked for some reason (e.g., wait for ADDBA Response, wait for TX on another
     * EMLSR link to finish, etc.). The <i>checkMediumBusy</i> argument allows to generate a new
     * backoff regardless of the busy/idle state of the medium, as per Section 35.3.16.4 of
     * 802.11be D4.0.
     *
     * @param txop the Txop requesting to generate a backoff
     * @param hadFramesToTransmit whether packets available for transmission were queued just
     *                            before the occurrence of the event triggering this call
     * @param checkMediumBusy whether generation of backoff (also) depends on the busy/idle state
     *                        of the medium
     * @return true if backoff needs to be generated, false otherwise
     */
    bool NeedBackoffUponAccess(Ptr<Txop> txop, bool hadFramesToTransmit, bool checkMediumBusy);

    /**
     * @param txop a Txop
     *
     * Notify the ChannelAccessManager that a specific Txop needs access to the
     * medium. The ChannelAccessManager is then responsible for starting an access
     * timer and, invoking FrameExchangeManager::StartTransmission when the access
     * is granted if it ever gets granted.
     */
    void RequestAccess(Ptr<Txop> txop);

    /**
     * Access will never be granted to the medium _before_
     * the time returned by this method.
     *
     * @param ignoreNav flag whether NAV should be ignored
     *
     * @returns the absolute time at which access could start to be granted
     */
    Time GetAccessGrantStart(bool ignoreNav = false) const;

    /**
     * Return the time when the backoff procedure
     * started for the given Txop.
     *
     * @param txop the Txop
     *
     * @return the time when the backoff procedure started
     */
    Time GetBackoffStartFor(Ptr<Txop> txop) const;

    /**
     * Return the time when the backoff procedure
     * ended (or will end) for the given Txop.
     *
     * @param txop the Txop
     *
     * @return the time when the backoff procedure ended (or will end)
     */
    Time GetBackoffEndFor(Ptr<Txop> txop) const;

    /**
     * Check whether channel access is expected to be granted within the given delay. If it is,
     * ACCESS_EXPECTED is returned. If channel access is not expected to be granted because no
     * AC has requested channel access, NOT_REQUESTED is returned. If no AC has frames to send,
     * NOTHING_TO_TX is returned. If any of the times returned by DoGetAccessGrantStart() exceeds
     * the given deadline, the reason corresponding to the earliest of such times is returned.
     * Otherwise, it means that access cannot be granted in time due to the backoff slots to wait
     * and BACKOFF_END is returned.
     *
     * @see DoGetAccessGrantStart
     * @param delay the given delay
     * @return ACCESS_EXPECTED or the reason why channel access is not expected to be gained in time
     */
    WifiExpectedAccessReason GetExpectedAccessWithin(const Time& delay) const;

    /**
     * @return the time until the NAV has been set
     */
    Time GetNavEnd() const;

    /**
     * @param qosTxop a QosTxop that needs to be disabled
     * @param duration the amount of time during which the QosTxop is disabled
     *
     * Disable the given EDCA for the given amount of time. This EDCA will not be
     * granted channel access during this period and the backoff timer will be frozen.
     * After this period, the EDCA will start normal operations again by resuming
     * the backoff timer.
     */
    void DisableEdcaFor(Ptr<Txop> qosTxop, Time duration);

    /**
     * Set the member variable indicating whether the backoff should be invoked when an AC gains
     * the right to start a TXOP but it does not transmit any frame (e.g., due to constraints
     * associated with EMLSR operations), provided that the queue is not actually empty.
     *
     * @param enable whether to enable backoff generation when no TX is performed in a TXOP
     */
    void SetGenerateBackoffOnNoTx(bool enable);

    /**
     * @return whether the backoff should be invoked when an AC gains the right to start a TXOP
     *         but it does not transmit any frame (e.g., due to constraints associated with EMLSR
     *         operations), provided that the queue is not actually empty
     */
    bool GetGenerateBackoffOnNoTx() const;

    /**
     * Return the width of the largest primary channel that has been idle for the
     * given time interval before the given time, if any primary channel has been
     * idle, or zero, otherwise.
     *
     * @param interval the given time interval
     * @param end the given end time
     * @return the width of the largest primary channel that has been idle for the given time
     * interval before the given time, if any primary channel has been idle, or zero, otherwise
     */
    MHz_u GetLargestIdlePrimaryChannel(Time interval, Time end);

    /**
     * @param indices a set of indices (starting at 0) specifying the 20 MHz channels to test
     * @return true if per-20 MHz CCA indicates busy for at least one of the
     *         specified 20 MHz channels, false otherwise
     */
    bool GetPer20MHzBusy(const std::set<uint8_t>& indices) const;

    /**
     * @param duration expected duration of reception
     *
     * Notify the Txop that a packet reception started
     * for the expected duration.
     */
    void NotifyRxStartNow(Time duration);
    /**
     * Notify the Txop that a packet reception was just
     * completed successfully.
     */
    void NotifyRxEndOkNow();
    /**
     * Notify the Txop that a packet reception was just
     * completed unsuccessfuly.
     *
     * @param txVector the TXVECTOR used for transmission
     */
    void NotifyRxEndErrorNow(const WifiTxVector& txVector);
    /**
     * @param duration expected duration of transmission
     *
     * Notify the Txop that a packet transmission was
     * just started and is expected to last for the specified
     * duration.
     */
    void NotifyTxStartNow(Time duration);
    /**
     * @param duration expected duration of CCA busy period
     * @param channelType the channel type for which the CCA busy state is reported.
     * @param per20MhzDurations vector that indicates for how long each 20 MHz subchannel
     *        (corresponding to the index of the element in the vector) is busy and where a zero
     * duration indicates that the subchannel is idle. The vector is non-empty if  the PHY supports
     * 802.11ax or later and if the operational channel width is larger than 20 MHz.
     *
     * Notify the Txop that a CCA busy period has just started.
     */
    void NotifyCcaBusyStartNow(Time duration,
                               WifiChannelListType channelType,
                               const std::vector<Time>& per20MhzDurations);
    /**
     * @param phyListener the PHY listener that sent this notification
     * @param duration expected duration of channel switching period
     *
     * Notify the Txop that a channel switching period has just started.
     * During switching state, new packets can be enqueued in Txop/QosTxop
     * but they won't access to the medium until the end of the channel switching.
     */
    void NotifySwitchingStartNow(PhyListener* phyListener, Time duration);
    /**
     * Notify the Txop that the device has been put in sleep mode.
     */
    void NotifySleepNow();
    /**
     * Notify the Txop that the device has been put in off mode.
     */
    void NotifyOffNow();
    /**
     * Notify the Txop that the device has been resumed from sleep mode.
     */
    void NotifyWakeupNow();
    /**
     * Notify the Txop that the device has been resumed from off mode.
     */
    void NotifyOnNow();
    /**
     * @param duration the value of the received NAV.
     *
     * Called at end of RX
     */
    void NotifyNavResetNow(Time duration);
    /**
     * @param duration the value of the received NAV.
     *
     * Called at end of RX
     */
    void NotifyNavStartNow(Time duration);
    /**
     * Notify that ack timer has started for the given duration.
     *
     * @param duration the duration of the timer
     */
    void NotifyAckTimeoutStartNow(Time duration);
    /**
     * Notify that ack timer has reset.
     */
    void NotifyAckTimeoutResetNow();
    /**
     * Notify that CTS timer has started for the given duration.
     *
     * @param duration the duration of the timer
     */
    void NotifyCtsTimeoutStartNow(Time duration);
    /**
     * Notify that CTS timer has reset.
     */
    void NotifyCtsTimeoutResetNow();

    /**
     * Check if the device is busy sending or receiving,
     * or NAV or CCA busy.
     *
     * @return true if the device is busy,
     *         false otherwise
     */
    bool IsBusy() const;

    /**
     * Reset the state variables of this channel access manager.
     */
    void ResetState();
    /**
     * Reset the backoff for the given DCF/EDCAF.
     *
     * @param txop the given DCF/EDCAF
     */
    void ResetBackoff(Ptr<Txop> txop);

    /**
     * Reset the backoff for all the DCF/EDCAF. Additionally, cancel the access timeout event.
     */
    void ResetAllBackoffs();

    /**
     * Notify that the given PHY is about to switch to the given operating channel, which is
     * used by the given link. This notification is sent by the EMLSR Manager when a PHY object
     * switches operating channel to operate on another link.
     *
     * @param phy the PHY object that is going to switch channel
     * @param channel the new operating channel of the given PHY
     * @param linkId the ID of the link on which the given PHY is going to operate
     */
    void NotifySwitchingEmlsrLink(Ptr<WifiPhy> phy,
                                  const WifiPhyOperatingChannel& channel,
                                  uint8_t linkId);

  protected:
    void DoInitialize() override;
    void DoDispose() override;

  private:
    /**
     * Get current registered listener for PHY events on the given PHY.
     *
     * @param phy the given PHY
     * @return the current registered listener for PHY events on the given PHY
     */
    std::shared_ptr<PhyListener> GetPhyListener(Ptr<WifiPhy> phy) const;

    /**
     * Initialize the structures holding busy end times per channel type (primary, secondary, etc.)
     * and per 20 MHz channel. All values are set to the current time.
     */
    void InitLastBusyStructs();

    /**
     * Resize the structures holding busy end times per channel type (primary, secondary, etc.)
     * and per 20 MHz channel. If a value (e.g., the busy end time for secondary40 channel) already
     * exists, it is not changed; otherwise, it is set to the current time.
     */
    void ResizeLastBusyStructs();
    /**
     * Update backoff slots for all Txops.
     */
    void UpdateBackoff();

    /**
     * This overload is provided to enable caching the value returned by GetAccessGrantStart(),
     * which is independent of the given Txop object.
     *
     * @param txop the Txop
     * @param accessGrantStart the value returned by GetAccessGrantStart()
     *
     * @return the time when the backoff procedure started
     */
    Time GetBackoffStartFor(Ptr<Txop> txop, Time accessGrantStart) const;

    /**
     * This overload is provided to enable caching the value returned by GetAccessGrantStart(),
     * which is independent of the given Txop object.
     *
     * @param txop the Txop
     * @param accessGrantStart the value returned by GetAccessGrantStart()
     *
     * @return the time when the backoff procedure ended (or will end)
     */
    Time GetBackoffEndFor(Ptr<Txop> txop, Time accessGrantStart) const;

    /**
     * Return a map containing (Time, WifiExpectedAccessReason) pairs sorted in increasing order
     * of times. For each of the events preventing channel access (e.g., medium busy, RX state,
     * TX state, etc), a pair is present in the map indicating the latest known time for which
     * channel access cannot be granted due to that event. Therefore, the returned map does not
     * contain a pair for some WifiExpectedAccessReason enum values (ACCESS_EXPECTED, NOTHING_TO_TX,
     * NOT_REQUESTED and BACKOFF_END).
     *
     * @param ignoreNav whether NAV should be ignored
     * @return a map containing (Time, WifiExpectedAccessReason) pairs sorted in increasing order
     *         of times
     */
    std::multimap<Time, WifiExpectedAccessReason> DoGetAccessGrantStart(bool ignoreNav) const;

    /**
     * This method determines whether the medium has been idle during a period (of
     * non-null duration) immediately preceding the time this method is called. If
     * so, the last idle start time and end time for each channel type are updated.
     * Otherwise, no change is made by this method.
     * This method is normally called when we are notified of the start of a
     * transmission, reception, CCA Busy or switching to correctly maintain the
     * information about the last idle period.
     */
    void UpdateLastIdlePeriod();

    void DoRestartAccessTimeoutIfNeeded();

    /**
     * Called when access timeout should occur
     * (e.g. backoff procedure expired).
     */
    void AccessTimeout();

    /**
     * Grant access to Txop using DCF/EDCF contention rules
     */
    void DoGrantDcfAccess();

    /**
     * Return the Short Interframe Space (SIFS) for this PHY.
     *
     * @return the SIFS duration
     */
    virtual Time GetSifs() const;

    /**
     * Return the slot duration for this PHY.
     *
     * @return the slot duration
     */
    virtual Time GetSlot() const;

    /**
     * Return the EIFS duration minus a DIFS.
     *
     * @return the EIFS duration minus a DIFS
     */
    virtual Time GetEifsNoDifs() const;

    /**
     * Structure defining start time and end time for a given state.
     */
    struct Timespan
    {
        Time start{0}; //!< start time
        Time end{0};   //!< end time
    };

    /**
     * typedef for a vector of Txops
     */
    typedef std::vector<Ptr<Txop>> Txops;

    Txops m_txops;            //!< the vector of managed Txops
    Time m_lastAckTimeoutEnd; //!< the last Ack timeout end time
    Time m_lastCtsTimeoutEnd; //!< the last CTS timeout end time
    Time m_lastNavEnd;        //!< the last NAV end time
    Timespan m_lastRx;        //!< the last receive start and end time
    bool m_lastRxReceivedOk;  //!< the last receive OK
    Time m_lastTxEnd;         //!< the last transmit end time
    std::map<WifiChannelListType, Time>
        m_lastBusyEnd;                       //!< the last busy end time for each channel type
    std::vector<Time> m_lastPer20MHzBusyEnd; /**< the last busy end time per 20 MHz channel
                                                  (HE stations and channel width > 20 MHz only) */
    std::map<WifiChannelListType, Timespan>
        m_lastIdle;               //!< the last idle start and end time for each channel type
    Time m_lastSwitchingEnd;      //!< the last switching end time
    Time m_lastSleepEnd;          //!< the last sleep end time
    Time m_lastOffEnd;            //!< the last off end time
    Time m_eifsNoDifs;            //!< EIFS no DIFS time
    Timespan m_lastNoPhy;         //!< the last start and end time no PHY was operating on the link
    mutable Time m_cachedSifs;    //!< cached value for SIFS, to be only used without a PHY
    mutable Time m_cachedSlot;    //!< cached value for slot, to be only used without a PHY
    EventId m_accessTimeout;      //!< the access timeout ID
    bool m_generateBackoffOnNoTx; //!< whether the backoff should be invoked when the AC gains the
                                  //!< right to start a TXOP but it does not transmit any frame
                                  //!< (e.g., due to constraints associated with EMLSR operations),
                                  //!< provided that the queue is not actually empty
    bool m_proactiveBackoff; //!< whether a new backoff value is generated when a CCA busy period
                             //!< starts and the backoff counter is zero
    Time m_resetBackoffThreshold; //!< if no PHY operates on a link for a period greater than this
                                  //!< threshold, the backoff on that link is reset

    /// Information associated with each PHY that is going to operate on another EMLSR link
    struct EmlsrLinkSwitchInfo
    {
        WifiPhyOperatingChannel channel; //!< new operating channel
        uint8_t linkId; //!< ID of the EMLSR link on which the PHY is going to operate
    };

    /// Store information about the PHY objects that are going to operate on another EMLSR link
    std::unordered_map<Ptr<WifiPhy>, EmlsrLinkSwitchInfo> m_switchingEmlsrLinks;

    /// Maps each PHY listener to the associated PHY
    using PhyListenerMap = std::unordered_map<Ptr<WifiPhy>, std::shared_ptr<PhyListener>>;

    PhyListenerMap m_phyListeners;         //!< the PHY listeners
    Ptr<WifiPhy> m_phy;                    //!< pointer to the unique active PHY
    Ptr<FrameExchangeManager> m_feManager; //!< pointer to the Frame Exchange Manager
    uint8_t m_linkId;                      //!< the ID of the link this object is associated with
    uint8_t m_nSlotsLeft;                  //!< fire the NSlotsLeftAlert trace source when the
                                           //!< backoff counter with the minimum value among all
                                           //!< ACs reaches this value
    Time m_nSlotsLeftMinDelay; //!< the minimum gap between the end of a medium busy event and
                               //!< the time the NSlotsLeftAlert trace source can be fired

    /// default value for the NSlotsLeftMinDelay attribute, corresponds to a PIFS in 5GHz/6GHz bands
    static const Time DEFAULT_N_SLOTS_LEFT_MIN_DELAY;

    /**
     * TracedCallback signature for NSlotsLeft alerts.
     *
     * @param linkId the ID of this link
     * @param aci the index of the AC that triggered the NSlotsLeft alert
     * @param backoffDelay delay until backoff counts down to zero
     */
    typedef void (*NSlotsLeftCallback)(uint8_t linkId, AcIndex aci, const Time& backoffDelay);

    /// TracedCallback for NSlotsLeft alerts typedef
    using NSlotsLeftTracedCallback = TracedCallback<uint8_t, AcIndex, const Time&>;

    NSlotsLeftTracedCallback m_nSlotsLeftCallback; //!< traced callback for NSlotsLeft alerts
};

/**
 * @brief Stream insertion operator.
 *
 * @param os the stream
 * @param reason the expected access reason
 * @return a reference to the stream
 */
std::ostream& operator<<(std::ostream& os, const WifiExpectedAccessReason& reason);

} // namespace ns3

#endif /* CHANNEL_ACCESS_MANAGER_H */
