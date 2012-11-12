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

#ifndef _DYNAMICSPIKINGSYNAPSE_H_
#define _DYNAMICSPIKINGSYNAPSE_H_

#include "global.h"
#ifdef _WIN32
typedef unsigned _int8 uint8_t;
#endif

class DynamicSpikingSynapse
{
public:

    //! Constructor, with params.
    DynamicSpikingSynapse( int source_x, int source_y, int sumX, int sumY, FLOAT& sum_point, FLOAT delay, FLOAT deltaT,
                           synapseType type );
    ~DynamicSpikingSynapse();

    //! Copy constructor.
    DynamicSpikingSynapse( const DynamicSpikingSynapse &other );

    //! Overloaded = operator.
    DynamicSpikingSynapse& operator= ( const DynamicSpikingSynapse &rhs );

    //! Reset the synapse state.
    void reset();

    //! Notify of a spike.
    void preSpikeHit();

    //! Advance a single time step.
    void advance();

    //! Advance a single time step.
    void advance( FLOAT*& summationMap, int width );

    //! Initialize the input spike queue.
    void initSpikeQueue();

    //! Add an input spike event to the queue.
    void addSpikeQueue();

    //! Check the input spike queue.
    bool isSpikeQueue();

    //! Update the internal state.
    bool updateInternal();

    //! Write the synapse data to the stream
    void write( ostream& os );

    //! Read the neuron data from the stream
    static void read( istream& is, FLOAT* pSummationMap, int width, vector<DynamicSpikingSynapse>* pSynapseMap );

    //! This synapse's summation point's address.
    FLOAT& summationPoint;

    //! The coordinates of the summation point.
    Coordinate summationCoord;

    //! The location of the synapse.
    Coordinate synapseCoord;

    //! The time step size.
    FLOAT deltaT;

    //! The weight (scaling factor, strength, maximal amplitude) of the synapse.
    FLOAT W;
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
