/**
 ** \brief A Interface Synapse.
 **
 ** \class ISynapse ISynapse.h "ISynapse.h"
 **
 ** \htmlonly  <h3>Model</h3> \endhtmlonly
 **
 ** The time varying state \f$x(t)\f$ of the synapse is increased by \f$W\cdot r \cdot u\f$ when a
 ** presynaptic spike hits the synapse and decays exponentially (time constant \f$\tau\f$) otherwise.
 ** \f$u\f$ and \f$r\f$ model the current state of facilitation and depression.\n
 ** A spike causes an exponential decaying postsynaptic response of the form \f$\exp(-t/\tau)\f$.
 **
 **/

/**
 ** \file ISynapse.h
 **
 ** \brief Header file for ISynapse
 **
 **/

#pragma once

#ifndef _ISYNAPSE_H_
#define _ISYNAPSE_H_

#include "global.h"
#ifdef _WIN32
typedef unsigned _int8 uint8_t;
#endif

class ISynapse
{
public:
	//! The coordinates of the summation point.
    Coordinate summationCoord;

	//! The weight (scaling factor, strength, maximal amplitude) of the synapse.
    FLOAT W;


    //! Constructor, with params.
    ISynapse( int source_x, int source_y, int sumX, int sumY, FLOAT& sum_point, FLOAT delay, FLOAT deltaT,
                           synapseType type );
    virtual ~ISynapse();

    //! Copy constructor.
    ISynapse( const ISynapse &other );

	//! Advance a single time step.
    virtual void advance() = 0;

    //! Update the internal state.
    virtual bool updateInternal() = 0;

	//! Write the synapse data to the stream
    virtual void write( ostream& os );

    ////! Read the neuron data from the stream
    //virtual void read( istream& is, FLOAT* pSummationMap, int width, vector<ISynapse>* pSynapseMap ) = 0;

    //! Reset the synapse state.
    virtual void reset();

	//! Notify of a spike.
    void preSpikeHit();

	//! Check the input spike queue.
    bool isSpikeQueue();

	//! Add an input spike event to the queue.
    void addSpikeQueue();

	//! Initialize the input spike queue.
    void initSpikeQueue();
	
protected:
    //! This synapse's summation point's address.
    FLOAT& summationPoint;

    //! The location of the synapse.
    Coordinate synapseCoord;

    //! The time step size.
    FLOAT deltaT;

    //! The post-synaptic response is the result of whatever computation is going on in the synapse.
    FLOAT psr;
    //! The decay for the psr.
    FLOAT decay;
    //! The synaptic transmission delay, descretized into time steps.
    int total_delay; 
    #define BYTES_OF_DELAYQUEUE         ( sizeof(uint32_t) / sizeof(uint8_t) )
    #define LENGTH_OF_DELAYQUEUE        ( BYTES_OF_DELAYQUEUE * 8 )
    //! The delayed queue
    uint32_t delayQueue[1];
    //! The index indicating the current time slot in the delayed queue
    int delayIdx;
    //! Length of the delayed queue
    int ldelayQueue;
    //! Synapse type
    synapseType type;

    //! The synaptic time constant \f$\tau\f$ [units=sec; range=(0,100)].
    FLOAT tau;

    // dynamic synapse vars...........
    //! The time varying state variable \f$r\f$ for depression.
    FLOAT r;
    //! The time varying state variable \f$u\f$ for facilitation.
    FLOAT u;
    //! The time constant of the depression of the dynamic synapse [range=(0,10); units=sec].
    FLOAT D;
    //! The use parameter of the dynamic synapse [range=(1e-5,1)].
    FLOAT U;
    //! The time constant of the facilitation of the dynamic synapse [range=(0,10); units=sec].
    FLOAT F;
    //! The time of the last spike.
    uint64_t lastSpike;
};

#endif
