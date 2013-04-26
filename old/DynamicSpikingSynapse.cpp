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
                                             BGFLOAT& sum_point,
                                             BGFLOAT delay, BGFLOAT new_deltaT, 
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

/**
 * Read the synapse data from the stream
 * @param[in] os	The filestream to read
 */
DynamicSpikingSynapse::DynamicSpikingSynapse( istream& is, BGFLOAT* pSummationMap, int width ) : 
	ISynapse(is, pSummationMap, width) {
}

/**
* Destructor
*
*/
DynamicSpikingSynapse::~DynamicSpikingSynapse() {
}

/**
 * Creates a synapse and initialize all internal state vars according to the source synapse.
 * @post The synapse is setup according to the state of the source synapse.
 * @param[in] other	The source synapse.
 */
DynamicSpikingSynapse::DynamicSpikingSynapse(const ISynapse &other) : ISynapse(other) {
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
			BGFLOAT isi = (g_simulationStep - lastSpike) * deltaT ;
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
//PAB: atomic above has implied flush (following statement generates error -- can't be member variable)
//#pragma omp flush (summationPoint)
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
ISynapse& DynamicSpikingSynapse::operator=(const ISynapse & rhs) {
	if (this == &rhs) // avoid aliasing
	return *this;

	this->~DynamicSpikingSynapse( ); // destroy self
	new ( this ) DynamicSpikingSynapse( rhs ); //reconstruct self using placement new and copy constructor

	return *this;
}
