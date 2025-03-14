/*
 * Copyright (c) 2008 INRIA
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */
#ifndef SIMPLE_CHANNEL_H
#define SIMPLE_CHANNEL_H

#include "mac48-address.h"

#include "ns3/channel.h"
#include "ns3/nstime.h"

#include <map>
#include <vector>

namespace ns3
{

class SimpleNetDevice;
class Packet;

/**
 * @ingroup channel
 * @brief A simple channel, for simple things and testing.
 *
 * This channel doesn't check for packet collisions and it
 * does not introduce any error.
 * By default, it does not add any delay to the packets.
 * Furthermore, it assumes that the associated NetDevices
 * are using 48-bit MAC addresses.
 *
 * This channel is meant to be used by ns3::SimpleNetDevices.
 */
class SimpleChannel : public Channel
{
  public:
    /**
     * @brief Get the type ID.
     * @return the object TypeId
     */
    static TypeId GetTypeId();
    SimpleChannel();

    /**
     * A packet is sent by a net device.  A receive event will be
     * scheduled for all net device connected to the channel other
     * than the net device who sent the packet
     *
     * @param p packet to be sent
     * @param protocol protocol number
     * @param to address to send packet to
     * @param from address the packet is coming from
     * @param sender netdevice who sent the packet
     *
     */
    virtual void Send(Ptr<Packet> p,
                      uint16_t protocol,
                      Mac48Address to,
                      Mac48Address from,
                      Ptr<SimpleNetDevice> sender);

    /**
     * Attached a net device to the channel.
     *
     * @param device the device to attach to the channel
     */
    virtual void Add(Ptr<SimpleNetDevice> device);

    /**
     * Blocks the communications from a NetDevice to another NetDevice.
     * The block is unidirectional
     *
     * @param from the device to BlackList
     * @param to the device wanting to block the other one
     */
    virtual void BlackList(Ptr<SimpleNetDevice> from, Ptr<SimpleNetDevice> to);

    /**
     * Un-Blocks the communications from a NetDevice to another NetDevice.
     * The block is unidirectional
     *
     * @param from the device to BlackList
     * @param to the device wanting to block the other one
     */
    virtual void UnBlackList(Ptr<SimpleNetDevice> from, Ptr<SimpleNetDevice> to);

    // inherited from ns3::Channel
    std::size_t GetNDevices() const override;
    Ptr<NetDevice> GetDevice(std::size_t i) const override;

  private:
    Time m_delay; //!< The assigned speed-of-light delay of the channel
    std::vector<Ptr<SimpleNetDevice>> m_devices; //!< devices connected by the channel
    std::map<Ptr<SimpleNetDevice>, std::vector<Ptr<SimpleNetDevice>>>
        m_blackListedDevices; //!< devices blocked on a device
};

} // namespace ns3

#endif /* SIMPLE_CHANNEL_H */
