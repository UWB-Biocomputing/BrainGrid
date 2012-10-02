/**
 * @file DelayList.cpp
 *
 * @author Fumitaka Kawasaki
 *
 * @brief A delayed list class for synapse input spikes.
 */
#include "DelayList.h"

/**
 * Constructor
 */
DelayList::DelayList( )
{
	init( DEFAULT_dt );
}

/**
 * Destructor
 */
DelayList::~DelayList( )
{
	destroy( );
}

/**
 * Initialize the list
 *
 * @param deltaT
 */
void DelayList::init( FLOAT deltaT )
{
	// we have one list of fixed length to store all events depending on their delay
	m_deltaT = deltaT;
	m_ldelayList = static_cast<int>( ( MAX_SYNDELAY / m_deltaT + 0.5 ) + 1 );
	m_rgDelayList = new t_pevent[m_ldelayList];
	memset( m_rgDelayList, 0, m_ldelayList * sizeof( event* ) );
	m_delayIdx = 0;

	// build event lists (list of synapses who will receive a spike after a certain delay)
	m_lfreeList = MIN_LENGTH_OF_FREELIST + 10000 / FRAC_SYN;
	m_freeList = new event[m_lfreeList];

	// we do our memory management by ourselfs to avoid unnecessary malloc/new calls
	m_allocList.push_back( m_freeList );
	m_freeIdx = 0;
	m_allocIdx = 1;
	m_recycledEvent = 0;
}

/**
 * Delete all allocated memory
 */
void DelayList::destroy( )
{
	for ( unsigned int i = 0; i < m_allocList.size( ); i++ )
		delete[] m_allocList[i];

	delete[] m_rgDelayList;
}

/**
 * Reset the list
 */
void DelayList::reset( )
{
	// empty delayList and reset memory management
	memset( m_rgDelayList, 0, m_ldelayList * sizeof( event* ) );
	m_delayIdx = 0;
	m_freeList = m_allocList[0];
	m_freeIdx = 0;
	m_allocIdx = 1;
	m_recycledEvent = 0;
}

/**
 * Clear the current time slot of the list
 *
 * @param e
 */
void DelayList::clear( event * e )
{
	// Event memory management: put the event slots back to the
	// recycled list.
	e->next = m_recycledEvent;
	m_recycledEvent = m_rgDelayList[m_delayIdx];

	// Mark this slot of the delayList as processed
	m_rgDelayList[m_delayIdx] = NULL;
}

/**
 * Add an input spike to the list
 *
 * @param total_delay
 * @param iNeuron
 * @param iSynapse
 */
void DelayList::add( int total_delay, int iNeuron, int iSynapse )
{
	// Implementation note: We keep iNeuron and iSynapse to identify a synapse
	// instead of keeping a pointer to a synapse object. 
	// The reason of this is that synapse objects are stored in vectors, and
	// memory management of vector moves the location of these objects. 
	// Therefore, values of object pointers become invalid and may cause 
	// memory fault. 
	// 
	// calculate index where to insert the synapse into the m_rgDelayList
	int idx = m_delayIdx +  total_delay;
	if ( idx >= m_ldelayList )
		idx -= m_ldelayList;	

	// Event memory management: get a event slot
	event *e;
	if ( m_recycledEvent != 0 ) {
		// get a recycled event slot
		e = m_recycledEvent;
		m_recycledEvent = m_recycledEvent->next;
	}
	else if ( m_freeIdx < m_lfreeList ) {
		// get slot from the current (m_allocIdx) pre-allocated memory chunk
		e = &( m_freeList[m_freeIdx++] );
	}
	else if ( m_allocIdx < m_allocList.size( ) ) {
		// current (m_allocIdx) pre-allocated memory chunk used up: go to
		// next chunk
		m_freeList = m_allocList[m_allocIdx++];
		e = &( m_freeList[0] );
		m_freeIdx = 1;
	}
	else {
		// no more chunks available: alloc a new one
		m_freeList = new event[m_lfreeList];
		m_allocList.push_back( m_freeList );
		++m_allocIdx;
		e = &( m_freeList[0] );
		m_freeIdx = 1;
	}

	// insert the event into the list of event at position m_delayIdx of
  	// the m_rgDelayList
	e->iNeuron = iNeuron;
	e->iSynapse = iSynapse;	
	e->next = m_rgDelayList[idx];
	m_rgDelayList[idx] = e;
}

/**
 * Get the pointer of event list of the current time slot
 */
DelayList::event *DelayList::get( )
{
	return m_rgDelayList[m_delayIdx];	
}

/**
 * Increment time 
 */
void DelayList::inc( )
{
	if ( ++m_delayIdx >= m_ldelayList )
		m_delayIdx = 0;
}

