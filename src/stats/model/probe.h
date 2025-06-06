/*
 * Copyright (c) 2011 Bucknell University
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: L. Felipe Perrone (perrone@bucknell.edu)
 *          Tiago G. Rodrigues (tgr002@bucknell.edu)
 */

#ifndef PROBE_H
#define PROBE_H

#include "data-collection-object.h"

#include "ns3/nstime.h"

namespace ns3
{

/**
 * @ingroup probes
 *
 * Base class for probes.
 *
 * This class provides general functionality to control each
 * probe and the data generated by it.
 */

class Probe : public DataCollectionObject
{
  public:
    /**
     * @brief Get the type ID.
     * @return the object TypeId
     */
    static TypeId GetTypeId();
    Probe();
    ~Probe() override;

    /**
     * @return true if Probe is currently enabled
     */
    bool IsEnabled() const override;

    /**
     * @brief connect to a trace source attribute provided by a given object
     *
     * @param traceSource the name of the attribute TraceSource to connect to
     * @param obj ns3::Object to connect to
     * @return true if the trace source was successfully connected
     */
    virtual bool ConnectByObject(std::string traceSource, Ptr<Object> obj) = 0;

    /**
     * @brief connect to a trace source provided by a config path
     *
     * @param path Config path to bind to
     *
     * Note, if an invalid path is provided, the probe will not be connected
     * to anything.
     */
    virtual void ConnectByPath(std::string path) = 0;

  protected:
    /// Time when logging starts.
    Time m_start;

    /// Time when logging stops.
    Time m_stop;
};

} // namespace ns3

#endif // PROBE_H
