/**
 * @file DelayList.h
 *
 * @brief Header file for DelayList
 */

/**
 ** \class DelayList DelayList.h "DelayList.h"
 **
 ** When a neuron fires, the spike is transmitted to the neurons that are connected to the fired neuron
 ** through synapses. The time needs to transmit the spike is called the synaptic transmission delay. 
 ** To realize the transmission delay in the simulation, we utilize a delay list. 
 ** The delay list is a global event queue consisting of an array, each element of which is the pointer 
 ** to a list of events. 
 ** An element of the array corresponds to a time when events happen, and there is an index that 
 ** points to the current simulation time. The index is incremented by the simulation proceeds, and 
 ** all events in the list pointed by the index will be executed. 
 ** The event is a structure, where index of the neuron, and synapse are stored to identify the synapse
 ** to be fired. We keep iNeuron and iSynapse to identify a synapse instead of keeping a pointer to a 
 ** synapse object. The reason of this is that synapse objects are stored in vectors, and memory management 
 ** of vector moves the location of these objects. Therefore, values of object pointers become invalid 
 ** and may cause memory fault. 
 **
 ** This delay list was originally implemented in CSIM, and removed in 2007 version, where transmission delay
 ** was handled by each synapse object. However, that implementation could not cope with dense spike trains. 
 ** We pick up code chunks of delay list from the CSIM and put them into one class. 
 **
 ** @author Fumitaka Kawasaki
 **/

#ifndef _DELAYLIST_H_
#define _DELAYLIST_H_

#include "global.h"
#include "DynamicSpikingSynapse.h"
#include <cstring>  //addded because it is needed for memset in g++ version 4.4.5

// maximal synaptic delay in seconds
#define MAX_SYNDELAY 0.1f
#define FRAC_SYN               50
#define MIN_LENGTH_OF_FREELIST 100

class DelayList
{
public:
	/** 
	 * Constructor
	 */
	DelayList( ); 

	/**
	 * Destructor
	 */
	~DelayList( );

	// an event structure
	struct event
	{
		event()
		{
			next = NULL;
		};
		int iNeuron;	// index of the neuron from
		int iSynapse;	// index of the synapse to
		event *next;
	};
	typedef event * t_pevent;

	//! Initialize the list
	void init( FLOAT deltaT );

	//! Free the memory for all event lists
	void destroy( );

	//! Reset the list
	void reset( );

	//! Clear the current time slot of the list
	void clear( event * e );

	//! Add an input spike event to the list
	void add( int total_delay, int iNeuron, int iSynapse );

	//! Get the pointer of event list of the current time slot
	event *get( );

	//! Increase m_delayIdx with loop around
	void inc( );

private:
	//! an array of event list
	event ** m_rgDelayList;

	//! an index indicating the current time slot in the delayed list
	int m_delayIdx;

	//! length of the delayed list
	int m_ldelayList;

	//! list of allocated memory, used for internal memory management
	vector<event *> m_allocList;

	//! recycled event list, used for internal memory management
	event *m_recycledEvent;

	//! free event array, used for internal memory management
	event *m_freeList;

	//! length of free event array
	int m_lfreeList;

	//! index of m_freeList
	int m_freeIdx;

	//! index of m_allocList
	unsigned m_allocIdx;

	//! minimum time interval between events of the simulation
	FLOAT m_deltaT;
};
#endif // _DELAYLIST_H_
