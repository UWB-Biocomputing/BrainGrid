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
                                             BGFLOAT& sum_point,
                                             BGFLOAT delay, BGFLOAT new_deltaT, 
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
	BGFLOAT tmpBGFLOAT = ( delay / new_deltaT);	//needed to be done in 2 lines or may cause incorrect results in linux
	total_delay = static_cast<int> (tmpBGFLOAT) + 1;

	// initialize spike queue
	initSpikeQueue();

	reset( );
}

/**
* Destructor
*
*/
ISynapse::~ISynapse() {
}

/**
 * Create a synapse and initialize all internal state vars according to input stream and summation map given
 * Allows creation of Synapse by utilizing the read from input stream function "read"
 * @param[in] is	The filestream to read
 * @param[in] pSummationMap		The map of all the summation points
 * @param[in] width		The width of the summantion map
 */
ISynapse::ISynapse( istream& is, BGFLOAT* pSummationMap, int width ) : summationPoint( pSummationMap[0] ) {
	// initialize spike queue
	initSpikeQueue();
	reset( );

	read(is, pSummationMap, width);
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
 * Read the synapse data from the stream
 * @param[in] is	The filestream to read
 * @param[in] pSummationMap		The map of all the summation points
 * @param[in] width		The width of the summantion map
 */
void ISynapse::read( istream& is, BGFLOAT* pSummationMap, int width ) {
	Coordinate t_summationCoord, t_synapseCoord;
	BGFLOAT t_deltaT, t_W, t_psr, t_decay, t_tau, t_r, t_u, t_D, t_U, t_F;
	int t_total_delay, t_delayIdx, t_ldelayQueue;
	uint32_t t_delayQueue[1];
	uint64_t t_lastSpike;
	synapseType t_type;

	is.read( reinterpret_cast<char*>(&t_summationCoord), sizeof(t_summationCoord) );
	is.read( reinterpret_cast<char*>(&t_synapseCoord), sizeof(t_synapseCoord) );
	is.read( reinterpret_cast<char*>(&t_deltaT), sizeof(t_deltaT) );
	is.read( reinterpret_cast<char*>(&t_W), sizeof(t_W) );
	is.read( reinterpret_cast<char*>(&t_psr), sizeof(t_psr) );
	is.read( reinterpret_cast<char*>(&t_decay), sizeof(t_decay) );
	is.read( reinterpret_cast<char*>(&t_total_delay), sizeof(t_total_delay) );
	is.read( reinterpret_cast<char*>(t_delayQueue), sizeof(uint32_t) );
	is.read( reinterpret_cast<char*>(&t_delayIdx), sizeof(t_delayIdx) );
	is.read( reinterpret_cast<char*>(&t_ldelayQueue), sizeof(t_ldelayQueue) );
	is.read( reinterpret_cast<char*>(&t_type), sizeof(t_type) );
	is.read( reinterpret_cast<char*>(&t_tau), sizeof(t_tau) );
	is.read( reinterpret_cast<char*>(&t_r), sizeof(t_r) );
	is.read( reinterpret_cast<char*>(&t_u), sizeof(t_u) );
	is.read( reinterpret_cast<char*>(&t_D), sizeof(t_D) );
	is.read( reinterpret_cast<char*>(&t_U), sizeof(t_U) );
	is.read( reinterpret_cast<char*>(&t_F), sizeof(t_F) );
	is.read( reinterpret_cast<char*>(&t_lastSpike), sizeof(t_lastSpike) );

	// locate summation point
	summationPoint = *&(pSummationMap[t_summationCoord.x + t_summationCoord.y * width]);

	// copy read values
	W = t_W;
	psr = t_psr;
	decay = t_decay;
	total_delay = t_total_delay;
	delayQueue[0] = t_delayQueue[0];
	delayIdx = t_delayIdx;
	ldelayQueue = t_ldelayQueue;
	tau = t_tau;
	r = t_r;
	u = t_u;
	D = t_D;
	U = t_U;
	F = t_F;
	lastSpike = t_lastSpike;
	summationCoord.x = t_summationCoord.x;
	summationCoord.y = t_summationCoord.y;
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
