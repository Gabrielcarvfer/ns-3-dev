/*
 * Copyright (c) 2007 INRIA
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */
#ifndef SIMULATION_SINGLETON_H
#define SIMULATION_SINGLETON_H

/**
 * @file
 * @ingroup singleton
 * ns3::SimulationSingleton declaration and template implementation.
 */

namespace ns3
{

/**
 * @ingroup singleton
 * This singleton class template ensures that the type
 * for which we want a singleton has a lifetime bounded
 * by the simulation run lifetime. That it, the underlying
 * type will be automatically deleted upon a call
 * to Simulator::Destroy.
 *
 * For a singleton with a lifetime bounded by the process,
 * not the simulation run, see Singleton.
 */
template <typename T>
class SimulationSingleton
{
  public:
    // Delete default constructor, copy constructor and assignment operator to avoid misuse
    SimulationSingleton() = delete;
    SimulationSingleton(const SimulationSingleton<T>&) = delete;
    SimulationSingleton& operator=(const SimulationSingleton<T>&) = delete;

    /**
     * Get a pointer to the singleton instance.
     *
     * This instance will be automatically deleted when the
     * simulation is destroyed by a call to Simulator::Destroy.
     *
     * @returns A pointer to the singleton instance.
     */
    static T* Get();

  private:
    /**
     * Get the singleton object, creating a new one if it doesn't exist yet.
     *
     * @internal
     * When a new object is created, this method schedules it's own
     * destruction using Simulator::ScheduleDestroy().
     *
     * @returns The address of the pointer holding the static instance.
     */
    static T** GetObject();

    /** Delete the static instance. */
    static void DeleteObject();
};

} // namespace ns3

/********************************************************************
 *  Implementation of the templates declared above.
 ********************************************************************/

#include "simulator.h"

namespace ns3
{

template <typename T>
T*
SimulationSingleton<T>::Get()
{
    T** ppobject = GetObject();
    return *ppobject;
}

template <typename T>
T**
SimulationSingleton<T>::GetObject()
{
    static T* pobject = nullptr;
    if (pobject == nullptr)
    {
        pobject = new T();
        Simulator::ScheduleDestroy(&SimulationSingleton<T>::DeleteObject);
    }
    return &pobject;
}

template <typename T>
void
SimulationSingleton<T>::DeleteObject()
{
    T** ppobject = GetObject();
    delete (*ppobject);
    *ppobject = nullptr;
}

} // namespace ns3

#endif /* SIMULATION_SINGLETON_H */
