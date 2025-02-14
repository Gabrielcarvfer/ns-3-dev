/*
 * Copyright (c) 2009 IITP RAS
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Kirill Andreev <andreev@iitp.ru>
 */

#ifndef FLAME_PROTOCOL_MAC_H
#define FLAME_PROTOCOL_MAC_H

#include "ns3/mesh-wifi-interface-mac.h"

namespace ns3
{
namespace flame
{
class FlameProtocol;

/**
 * @ingroup flame
 *
 * @brief Interface MAC plugin for FLAME routing protocol
 */
class FlameProtocolMac : public MeshWifiInterfaceMacPlugin
{
  public:
    /**
     * Constructor
     *
     * @param protocol flame protocol object
     */
    FlameProtocolMac(Ptr<FlameProtocol> protocol);
    ~FlameProtocolMac() override;

    // Inherited from MAC plugin
    /**
     * Set parent of this instance
     * @param parent pointer to the parent MeshWifiInterfaceMac
     */
    void SetParent(Ptr<MeshWifiInterfaceMac> parent) override;
    /**
     * Receive and process a packet; packets are given a FlameTag packet tag
     * @param packet the packet received
     * @param header the header
     * @returns true if successful
     */
    bool Receive(Ptr<Packet> packet, const WifiMacHeader& header) override;
    /**
     * Process an outgoing frame.  Remove the FlameTag and increment stats
     * counters.
     * @param packet the packet received
     * @param header the header
     * @param from the MAC address of the sender
     * @param to the MAC address of the receiver
     * @returns true if successful
     */
    bool UpdateOutcomingFrame(Ptr<Packet> packet,
                              WifiMacHeader& header,
                              Mac48Address from,
                              Mac48Address to) override;

    /**
     * Update beacon is empty, because FLAME does not know anything about beacons
     * @param beacon the beacon
     */
    void UpdateBeacon(MeshWifiBeacon& beacon) const override
    {
    }

    /**
     * AssignStreams is empty, because this model doesn't use random variables
     * @param stream
     * @returns 0 (no streams used)
     */
    int64_t AssignStreams(int64_t stream) override
    {
        return 0;
    }

    /**
     * Get channel ID function
     * @returns the channel
     */
    uint16_t GetChannelId() const;
    /**
     * Report statistics
     * @param os the output stream
     */
    void Report(std::ostream& os) const;
    /// Reset statistics function
    void ResetStats();

  private:
    // MeshPointDevice parameters:
    Ptr<FlameProtocol> m_protocol;      ///< protocol
    Ptr<MeshWifiInterfaceMac> m_parent; ///< parent

    /// Statistics structure
    struct Statistics
    {
        uint16_t txUnicast;   ///< transmit unicast
        uint16_t txBroadcast; ///< transit broadcast
        uint32_t txBytes;     ///< transmit bytes
        uint16_t rxUnicast;   ///< receive unicast
        uint16_t rxBroadcast; ///< receive broadcast
        uint32_t rxBytes;     ///< receive bytes

        /**
         * Print function
         * @param os the output stream
         */
        void Print(std::ostream& os) const;
        /// constructor
        Statistics();
    };

    Statistics m_stats; ///< statistics
};
} // namespace flame
} // namespace ns3
#endif /* FLAME_PROTOCOL_MAC_H */
