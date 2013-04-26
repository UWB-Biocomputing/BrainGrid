/**
 ** \file LifNeuron.cpp
 **
 ** \authors Allan Ortiz & Cory Mayberry
 **
 ** \brief A leaky-integrate-and-fire neuron
 **/

#include "LifNeuron.h"

/**
 * Create a neuron and initialize all internal state vars using 
 * the default values in global.cpp.
 * @post The neuron is setup according to the default values. 
 */
LifNeuron::LifNeuron() : INeuron()
{
}

/**
* Destructor
*
*/
LifNeuron::~LifNeuron() {
}

/**
 * If the neuron is refractory, decrement the remaining refractory period.
 * If \f$V_m\f$ exceeds \f$V_{thresh}\f$ a spike is emmited. 
 * Otherwise, decay \f$Vm\f$ and add inputs.
 * @param[in] summationPoint
 */
void LifNeuron::advance(BGFLOAT& summationPoint) {
	if (nStepsInRefr > 0) { // is neuron refractory?
		--nStepsInRefr;
	} else if (Vm >= Vthresh) { // should it fire?
		fire( );
	} else {
		summationPoint += I0; // add IO
#ifdef USE_OMP
		int tid = OMP(omp_get_thread_num());
		summationPoint += ( (*rgNormrnd[tid])( ) * Inoise ); // add noise
#else
		summationPoint += ( (*rgNormrnd[0])( ) * Inoise ); // add noise
#endif
		Vm = C1 * Vm + C2 * summationPoint; // decay Vm and add inputs
	}
	// clear synaptic input for next time step
	summationPoint = 0;
}

/**
 * Propagate the spike to the synapse and reset the neuron.
 * If STORE_SPIKEHISTORY is set, spike time is recorded. 
 */
void LifNeuron::fire()
{
	// Note that the neuron has fired!
	hasFired = true;

#ifdef STORE_SPIKEHISTORY
	// record spike time
	spikeHistory.push_back(g_simulationStep);
#endif // STORE_SPIKEHISTORY

	// increment spike count
	spikeCount++;

	// calculate the number of steps in the absolute refractory period
	nStepsInRefr = static_cast<int> ( Trefract / deltaT + 0.5 );

	// reset to 'Vreset'
	Vm = Vreset;
}

/**
 * Init consts C1,C2 for exponential Euler integration,
 * and calculate const IO.
 */
void LifNeuron::updateInternal() {
	/* init consts C1,C2 for exponential Euler integration */
	if (Tau > 0) {
		C1 = exp( -deltaT / Tau );
		C2 = Rm * ( 1 - C1 );
	} else {
		C1 = 0.0;
		C2 = Rm;
	}
	/* calculate const IO */
	if (Rm > 0)
		I0 = Iinject + Vrest / Rm;
	else {
		assert(false);
	}
}