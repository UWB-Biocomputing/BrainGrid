/**
 ** \file DynamicSpikingSynapse.cpp
 **
 ** \authors Allan Ortiz & Cory Mayberry
 **
 ** \brief A dynamic spiking synapse (Makram et al (1998))
 **/

#include "DynamicSpikingSynapse.h"	

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
DynamicSpikingSynapse::DynamicSpikingSynapse(int source_x, int source_y, 
                                             int sumX, int sumY, 
                                             FLOAT& sum_point,
                                             FLOAT delay, FLOAT new_deltaT, 
                                             synapseType s_type) :
    ISynapse(source_x, source_y, sumX, sumY, sum_point, delay, new_deltaT, s_type)
{
	type = s_type;
	switch (type) {
	case II:
		U = 0.32;
		D = 0.144;
		F = 0.06;
		tau = 6e-3;
		delay = 0.8e-3;
		break;
	case IE:
		U = 0.25;
		D = 0.7;
		F = 0.02;
		tau = 6e-3;
		delay = 0.8e-3;
		break;
	case EI:
		U = 0.05;
		D = 0.125;
		F = 1.2;
		tau = 3e-3;
		delay = 0.8e-3;
		break;
	case EE:
		U = 0.5;
		D = 1.1;
		F = 0.05;
		tau = 3e-3;
		delay = 1.5e-3;
		break;
	default:
		assert( false );
	}
}

DynamicSpikingSynapse::~DynamicSpikingSynapse() {
}

/**
 * Creates a synapse and initialize all internal state vars according to the source synapse.
 * @post The synapse is setup according to the state of the source synapse.
 * @param[in] other	The source synapse.
 */
DynamicSpikingSynapse::DynamicSpikingSynapse(const DynamicSpikingSynapse &other) : ISynapse(other) {
}

/**
 * If an input spike is in the queue, adjust synapse parameters, calculate psr.
 * Decay the post spike response(psr), and apply it to the summation point.
 */
void DynamicSpikingSynapse::advance() {
	// is an input in the queue?
	if (isSpikeQueue()) {
		// adjust synapse paramaters
		if (lastSpike != ULONG_MAX) {
			FLOAT isi = (g_simulationStep - lastSpike) * deltaT ;
			r = 1 + ( r * ( 1 - u ) - 1 ) * exp( -isi / D );
			u = U + u * ( 1 - U ) * exp( -isi / F );
		}
		psr += ( ( W / decay ) * u * r );// calculate psr
		lastSpike = g_simulationStep; // record the time of the spike
	}

	// decay the post spike response
	psr *= decay;
	// and apply it to the summation point
#ifdef USE_OMP
#pragma omp atomic
#endif
	summationPoint += psr;
#ifdef USE_OMP
#pragma omp flush (summationPoint)
#endif
}

/**
 * Recompute decay.
 * @return true if success.
 */
bool DynamicSpikingSynapse::updateInternal() {
	if (tau > 0) {
		decay = exp( -deltaT / tau );
		return true;
	}
	return false;
}

/**
 * Reconstruct self using placement new and copy constructor.
 * @param[in] rhs	Overloaded = operator.
 */
DynamicSpikingSynapse& DynamicSpikingSynapse::operator=(const DynamicSpikingSynapse & rhs) {
	if (this == &rhs) // avoid aliasing
	return *this;

	this->~DynamicSpikingSynapse( ); // destroy self
	new ( this ) DynamicSpikingSynapse( rhs ); //reconstruct self using placement new and copy constructor

	return *this;
}

/**
 * Read the synapse data from the stream
 * @param[in] os	The filestream to read
 */
void DynamicSpikingSynapse::read( istream& is, FLOAT* pSummationMap, int width, vector<DynamicSpikingSynapse>* pSynapseMap ) {
	Coordinate t_summationCoord, t_synapseCoord;
	FLOAT t_deltaT, t_W, t_psr, t_decay, t_tau, t_r, t_u, t_D, t_U, t_F;
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
	FLOAT* sp = &(pSummationMap[t_summationCoord.x + t_summationCoord.y * width]);

	// create synapse
	DynamicSpikingSynapse syn(t_synapseCoord.x, t_synapseCoord.y, t_summationCoord.x, t_summationCoord.y, *sp, DEFAULT_delay_weight, t_deltaT, t_type);

	// copy read values
	syn.W = t_W;
	syn.psr = t_psr;
	syn.decay = t_decay;
	syn.total_delay = t_total_delay;
	syn.delayQueue[0] = t_delayQueue[0];
	syn.delayIdx = t_delayIdx;
	syn.ldelayQueue = t_ldelayQueue;
	syn.tau = t_tau;
	syn.r = t_r;
	syn.u = t_u;
	syn.D = t_D;
	syn.U = t_U;
	syn.F = t_F;
	syn.lastSpike = t_lastSpike;

	pSynapseMap[t_synapseCoord.x + t_synapseCoord.y * width].push_back(syn);
}