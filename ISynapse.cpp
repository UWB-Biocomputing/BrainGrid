/**
 ** \file ISynapse.cpp
 **
 ** \brief A interface for synapsse
 **/

#include "ISynapse.h"	

/**
 * Create a synapse and initialize all internal state vars according to the synapse type.
 * @post The synapse is setup according to parameters.
 * @param[in] source_x	The location(x) of the synapse.
 * @param[in] source_y	The location(y) of the synapse.
 * @param[in] sumX	The coordinates(x) of the summation point. 
 * @param[in] sumY	The coordinates(y) of the summation point.
 * @param[in] delay	The synaptic transmission delay (sec).
 * @param[in] new_deltaT The time step size (sec).
 * @param[in] s_type	Synapse type.
 */
ISynapse::ISynapse(int source_x, int source_y, 
                                             int sumX, int sumY, 
                                             FLOAT& sum_point,
                                             FLOAT delay, FLOAT new_deltaT, 
                                             synapseType s_type) :
    summationPoint( sum_point ),
    deltaT( new_deltaT ),
    W( 10.0e-9 ),
    psr( 0.0 ),
    decay(0),
    total_delay( static_cast<int>( delay / new_deltaT )), 
    tau( DEFAULT_tau ),
    r( 1.0 ),
    u(0),
    D( 1.0 ),
    U( DEFAULT_U ),
    F( 0.01 ),
    lastSpike( ULONG_MAX )
{
	synapseCoord.x = source_x;
	synapseCoord.y = source_y;
	summationCoord.x = sumX;
	summationCoord.y = sumY;

	// calculate the discrete delay (time steps)
	FLOAT tmpFLOAT = ( delay / new_deltaT);	//needed to be done in 2 lines or may cause incorrect results in linux
	total_delay = static_cast<int> (tmpFLOAT) + 1;

	// initialize spike queue
	initSpikeQueue();

	reset( );
}

ISynapse::~ISynapse() {
}

/**
 * Creates a synapse and initialize all internal state vars according to the source synapse.
 * @post The synapse is setup according to the state of the source synapse.
 * @param[in] other	The source synapse.
 */
ISynapse::ISynapse(const ISynapse &other) :
	summationPoint( other.summationPoint ), summationCoord( other.summationCoord ), synapseCoord( other.synapseCoord ),
			deltaT( other.deltaT ), W( other.W ), psr( other.psr ), decay( other.decay ), total_delay( other.total_delay ), 
					delayIdx( other.delayIdx ), ldelayQueue( other.ldelayQueue ), type( other.type ),
					tau( other.tau ), r( other.r ), u( other.u ), D( other.D ), U( other.U ), F( other.F ), lastSpike(
					other.lastSpike ) {
	delayQueue[0] = other.delayQueue[0];
}

/**
 * Write the synapse data to the stream
 * @param[in] os	The filestream to write
 */
void ISynapse::write( ostream& os ) {
	os.write( reinterpret_cast<const char*>(&summationCoord), sizeof(summationCoord) );
	os.write( reinterpret_cast<const char*>(&synapseCoord), sizeof(synapseCoord) );
	os.write( reinterpret_cast<const char*>(&deltaT), sizeof(deltaT) );
	os.write( reinterpret_cast<const char*>(&W), sizeof(W) );
	os.write( reinterpret_cast<const char*>(&psr), sizeof(psr) );
	os.write( reinterpret_cast<const char*>(&decay), sizeof(decay) );
	os.write( reinterpret_cast<const char*>(&total_delay), sizeof(total_delay) );
	os.write( reinterpret_cast<const char*>(delayQueue), sizeof(uint32_t) );
	os.write( reinterpret_cast<const char*>(&delayIdx), sizeof(delayIdx) );
	os.write( reinterpret_cast<const char*>(&ldelayQueue), sizeof(ldelayQueue) );
	os.write( reinterpret_cast<const char*>(&type), sizeof(type) );
	os.write( reinterpret_cast<const char*>(&tau), sizeof(tau) );
	os.write( reinterpret_cast<const char*>(&r), sizeof(r) );
	os.write( reinterpret_cast<const char*>(&u), sizeof(u) );
	os.write( reinterpret_cast<const char*>(&D), sizeof(D) );
	os.write( reinterpret_cast<const char*>(&U), sizeof(U) );
	os.write( reinterpret_cast<const char*>(&F), sizeof(F) );
	os.write( reinterpret_cast<const char*>(&lastSpike), sizeof(lastSpike) );
}

/**
 * Reset time varying state vars and recompute decay.
 */
void ISynapse::reset() {
	psr = 0.0;
	assert( updateInternal() );
	u = DEFAULT_U;
	r = 1.0;
	lastSpike = ULONG_MAX;
}

/**
 * Add an input spike event to the queue.
 */
void ISynapse::preSpikeHit() {
	// add an input spike event to the queue
	addSpikeQueue();
}

/**
 * Clear events in the queue, and initialize the queue. 
 */
void ISynapse::initSpikeQueue()
{
	size_t size = total_delay / ( sizeof(uint8_t) * 8 ) + 1;
	assert( size <= BYTES_OF_DELAYQUEUE );
	delayQueue[0] = 0;
	delayIdx = 0;
	ldelayQueue = LENGTH_OF_DELAYQUEUE;
}

/**
 * Add an input spike event to the queue according to the current index and delay.
 */
void ISynapse::addSpikeQueue()
{
	// calculate index where to insert the spike into delayQueue
	int idx = delayIdx +  total_delay;
	if ( idx >= ldelayQueue )
		idx -= ldelayQueue;

	// set a spike
	assert( !(delayQueue[0] & (0x1 << idx)) );
	delayQueue[0] |= (0x1 << idx);
}

/**
 * Check if there is an input spike in the queue.
 * @post The queue index is incremented. 
 * @return true if there is an input spike event.
 */
bool ISynapse::isSpikeQueue()
{
	bool r = delayQueue[0] & (0x1 << delayIdx);
	delayQueue[0] &= ~(0x1 << delayIdx);
	if ( ++delayIdx >= ldelayQueue )
		delayIdx = 0;
	return r;
}
