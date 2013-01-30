/**
 * \file ISynapse.h
 *
 * Header file for ISynapse
 * ISynapse is the base interface for a Synapse to be used with the Simulator.
 *
	DATE		VERSION		NAME		COMMENT
 *	11/24/2012	1.0			dwise		Initial stab at creating an ISynapse in the simulator
 */

#pragma once

#ifndef _ISYNAPSE_H_
#define _ISYNAPSE_H_

#include "global.h"
#ifdef _WIN32
typedef unsigned _int8 uint8_t;
#endif

//! Abstract interface class to provide basic framework needed for a Synapse to function with the simulator
class ISynapse
{
public:
    //! Constructor, with params.
    ISynapse( int source_x, int source_y, int sumX, int sumY, FLOAT& sum_point, FLOAT delay, FLOAT deltaT,
                           synapseType type );
	//! Constructor, with params. Loads directly from input stream
    ISynapse( istream& is, FLOAT* pSummationMap, int width );
    virtual ~ISynapse();

    //! Copy constructor.
    ISynapse( const ISynapse &other );

	//! Advance a single time step.
    virtual void advance() = 0;

    //! Update the internal state.
    virtual bool updateInternal() = 0;

	//! Overloaded = operator.
    virtual ISynapse& operator= ( const ISynapse &rhs ) = 0;

	//! Write the synapse data to the stream
    virtual void write( ostream& os );

    ////! Read the neuron data from the stream
    virtual void read( istream& is, FLOAT* pSummationMap, int width );

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
	

	//! The coordinates of the summation point.
    Coordinate summationCoord;

	//! The weight (scaling factor, strength, maximal amplitude) of the synapse.
    FLOAT W;

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

// NETWORK MODEL VARIABLES NMV-BEGIN {
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
// } NMV-END
};

#endif
