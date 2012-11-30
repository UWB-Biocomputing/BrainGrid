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

class DynamicSpikingSynapse : public ISynapse
{
public:

    //! Constructor, with params.
    DynamicSpikingSynapse( int source_x, int source_y, int sumX, int sumY, FLOAT& sum_point, FLOAT delay, FLOAT deltaT,
                           synapseType type );
    virtual ~DynamicSpikingSynapse();

	//! Copy constructor.
    DynamicSpikingSynapse( const DynamicSpikingSynapse &other );

    //! Advance a single time step.
    void advance();

    //! Update the internal state.
    bool updateInternal();

	//! Overloaded = operator.
    DynamicSpikingSynapse& operator= ( const DynamicSpikingSynapse &rhs );

    //! Read the neuron data from the stream
    static void read( istream& is, FLOAT* pSummationMap, int width, vector<DynamicSpikingSynapse>* pSynapseMap );

};

#endif
