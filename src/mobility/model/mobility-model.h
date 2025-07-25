/*
 * Copyright (c) 2006,2007 INRIA
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */
#ifndef MOBILITY_MODEL_H
#define MOBILITY_MODEL_H

#include "ns3/mobility-export.h"
#include "ns3/object.h"
#include "ns3/traced-callback.h"
#include "ns3/vector.h"

namespace ns3
{

/**
 * @ingroup mobility
 * @brief Keep track of the current position and velocity of an object.
 *
 * All space coordinates in this class and its subclasses are
 * understood to be meters or meters/s. i.e., they are all
 * metric international units.
 *
 * This is a base class for all specific mobility models.
 */
class MOBILITY_EXPORT MobilityModel : public Object
{
  public:
    /**
     * Register this type with the TypeId system.
     * @return the object TypeId
     */
    static TypeId GetTypeId();
    MobilityModel();
    ~MobilityModel() override = 0;

    /**
     * @return the current position
     */
    Vector GetPosition() const;
    /**
     * This method may be used if the position returned may depend on some
     * reference position provided.  For example, in a hierarchical mobility
     * model that is buildings-aware, the child mobility model may not be able
     * to determine if it is inside or outside of a building unless it knows
     * the parent position.
     *
     * @param referencePosition reference position to consider
     * @return the current position based on the provided referencePosition
     * \sa ns3::MobilityModel::DoGetPositionWithReference
     */
    Vector GetPositionWithReference(const Vector& referencePosition) const;
    /**
     * @param position the position to set.
     */
    void SetPosition(const Vector& position);
    /**
     * @return the current velocity.
     */
    Vector GetVelocity() const;
    /**
     * @param position a reference to another mobility model
     * @return the distance between the two objects. Unit is meters.
     */
    double GetDistanceFrom(Ptr<const MobilityModel> position) const;
    /**
     * @param other reference to another object's mobility model
     * @return the relative speed between the two objects. Unit is meters/s.
     */
    double GetRelativeSpeed(Ptr<const MobilityModel> other) const;
    /**
     * Assign a fixed random variable stream number to the random variables
     * used by this model. Return the number of streams (possibly zero) that
     * have been assigned.
     *
     * @param stream first stream index to use
     * @return the number of stream indices assigned by this model
     */
    int64_t AssignStreams(int64_t stream);

    /**
     *  TracedCallback signature.
     *
     * @param [in] model Value of the MobilityModel.
     */
    typedef void (*TracedCallback)(Ptr<const MobilityModel> model);

  protected:
    /**
     * Must be invoked by subclasses when the course of the
     * position changes to notify course change listeners.
     */
    void NotifyCourseChange() const;

  private:
    /**
     * @return the current position.
     *
     * Concrete subclasses of this base class must
     * implement this method.
     */
    virtual Vector DoGetPosition() const = 0;
    /**
     * @param referencePosition the reference position to consider
     * @return the current position.
     *
     * Unless subclasses override, this method will disregard the reference
     * position and return "DoGetPosition ()".
     */
    virtual Vector DoGetPositionWithReference(const Vector& referencePosition) const;
    /**
     * @param position the position to set.
     *
     * Concrete subclasses of this base class must
     * implement this method.
     */
    virtual void DoSetPosition(const Vector& position) = 0;
    /**
     * @return the current velocity.
     *
     * Concrete subclasses of this base class must
     * implement this method.
     */
    virtual Vector DoGetVelocity() const = 0;
    /**
     * The default implementation does nothing but return the passed-in
     * parameter.  Subclasses using random variables are expected to
     * override this.
     * @param start  starting stream index
     * @return the number of streams used
     */
    virtual int64_t DoAssignStreams(int64_t start);

    /**
     * Used to alert subscribers that a change in direction, velocity,
     * or position has occurred.
     */
    ns3::TracedCallback<Ptr<const MobilityModel>> m_courseChangeTrace;
};

} // namespace ns3

#endif /* MOBILITY_MODEL_H */
