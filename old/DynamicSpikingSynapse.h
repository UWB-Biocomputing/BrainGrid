/**
 ** \brief A Dynamic Spiking Synapse.
 **
 ** \class DynamicSpikingSynapse DynamicSpikingSynapse.h "DynamicSpikingSynapse.h"
 **
 ** \htmlonly  <h3>Model</h3> \endhtmlonly
 **
 ** The time varying state \f$x(t)\f$ of the synapse is increased by \f$W\cdot r \cdot u\f$ when a
 ** presynaptic spike hits the synapse and decays exponentially (time constant \f$\tau\f$) otherwise.
 ** \f$u\f$ and \f$r\f$ model the current state of facilitation and depression.\n
 ** A spike causes an exponential decaying postsynaptic response of the form \f$\exp(-t/\tau)\f$.
 **
 ** @authors Allan Ortiz & Cory Mayberry
 **
 **/

/**
 ** \file DynamicSpikingSynapse.h
 **
 ** \brief Header file for DynamicSpikingSynapse
 **
 **/

#pragma once

#ifndef _DYNAMICSPIKINGSYNAPSE_H_
#define _DYNAMICSPIKINGSYNAPSE_H_

#include "ISynapse.h"	

//! Implementation of ISynapse: A Dynamic Spiking Synapse
class DynamicSpikingSynapse : public ISynapse
{
public:

    //! Constructor, with params.
    DynamicSpikingSynapse( int source_x, int source_y, int sumX, int sumY, BGFLOAT& sum_point, BGFLOAT delay, BGFLOAT deltaT,
                           synapseType type );
	//! Constructor, with params. Loads directly from input stream
    DynamicSpikingSynapse( istream& is, BGFLOAT* pSummationMap, int width );
    virtual ~DynamicSpikingSynapse();

	//! Copy constructor.
    DynamicSpikingSynapse( const ISynapse &other );

    //! Advance a single time step.
    void advance();

    //! Update the internal state.
    bool updateInternal();

	//! Overloaded = operator.
    ISynapse& operator= ( const ISynapse &rhs );

};

#endif
