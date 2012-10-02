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
LifNeuron::LifNeuron() :
    deltaT(DEFAULT_dt), 
    Cm(DEFAULT_Cm),
    Rm(DEFAULT_Rm),
    Vthresh(DEFAULT_Vthresh), 
    Vrest(DEFAULT_Vrest), 
    Vreset(DEFAULT_Vreset), 
    Vinit(Vreset), 
    Trefract(DEFAULT_Trefract), 
    Inoise(DEFAULT_Inoise), 
    Iinject(DEFAULT_Iinject),
    Tau(DEFAULT_Cm * DEFAULT_Rm)
{
	reset( );
}

/**
 * Set the neuron parameters according to the input parameters.
 * @param[in] new_Iinject	A constant current to be injected into the LIF neuron.
 * @param[in] new_Inoise	The standard deviation of the noise to be added each integration time constant. 
 * @param[in] new_Vthresh	If \f$V_m\f$ exceeds \f$V_{thresh}\f$ a spike is emmited.
 * @param[in] new_Vresting	The resting membrane voltage.
 * @param[in] new_Vreset	The voltage to reset \f$V_m\f$ to after a spike.
 * @param[in] new_Vinit		The initial condition for \f$V_m\f$ at time \f$t=0\f$
 * @param[in] new_deltaT	The simulation time step size.
 */
void LifNeuron::setParams(FLOAT new_Iinject, FLOAT new_Inoise, FLOAT new_Vthresh, FLOAT new_Vresting,
		FLOAT new_Vreset, FLOAT new_Vinit, FLOAT new_deltaT) {
	Iinject = new_Iinject;
	Inoise = new_Inoise;
	Vthresh = new_Vthresh;
	Vrest = new_Vresting;
	Vreset = new_Vreset;
	Vinit = new_Vinit;
	deltaT = new_deltaT;
	reset( );
}

LifNeuron::~LifNeuron() {
}

/**
 * If the neuron is refractory, decrement the remaining refractory period.
 * If \f$V_m\f$ exceeds \f$V_{thresh}\f$ a spike is emmited. 
 * Otherwise, decay \f$Vm\f$ and add inputs.
 * @param[in] summationPoint
 */
void LifNeuron::advance(FLOAT& summationPoint) {
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
 * Reset time varying state vars.
 * If STORE_SPIKEHISTORY is set, reset spike history.
 */
void LifNeuron::reset(void) {
	hasFired = false;
#ifdef STORE_SPIKEHISTORY
	spikeHistory.clear( );
#endif // STORE_SPIKEHISTORY
	nStepsInRefr = 0;
	Vm = Vinit;
	updateInternal( );
	spikeCount = 0;
}

/**
 * Init consts C1,C2 for exponential Euler integration,
 * and calculate const IO.
 */
void LifNeuron::updateInternal(void) {
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

/**
 * @return a string with the Vm.
 */
string LifNeuron::toStringVm() {
	stringstream ss;
	ss << Vm;
	return ss.str( );
}

/**
 * @return a terse representation.
 */
string LifNeuron::toString() {
	stringstream ss;
	if (hasFired) ss << "!! FIRED !!    ";
	ss << "Vm: " << Vm;
	if (isRefractory( )) ss << "      refract: " << nStepsInRefr;
	return ss.str( );
}

/**
 * @return the complete state of the neuron.
 */
string LifNeuron::toStringAll() {
	stringstream ss;
	ss << "Cm: " << Cm << " "; // membrane capacitance
	ss << "Rm: " << Rm << " "; // membrane resistance
	ss << "Vthresh: " << Vthresh << " "; // if Vm exceeds, Vthresh, a spike is emitted
	ss << "Vrest: " << Vrest << " "; // the resting membrane voltage
	ss << "Vreset: " << Vreset << " "; // The voltage to reset Vm to after a spike
	ss << "Vinit: " << Vinit << endl; // The initial condition for V_m at t=0
	ss << "Trefract: " << Trefract << " "; // the number of steps in the refractory period
	ss << "Inoise: " << Inoise << " "; // the stdev of the noise to be added each delta_t
	ss << "Iinject: " << Iinject << " "; // A constant current to be injected into the LIF neuron
	ss << "nStepsInRefr: " << nStepsInRefr << endl; // the number of steps left in the refractory period
	ss << "Vm: " << Vm << " "; // the membrane voltage
	ss << "hasFired: " << hasFired << " "; // it done fired?
	ss << "C1: " << C1 << " ";
	ss << "C2: " << C2 << " ";
	ss << "I0: " << I0 << " ";
	return ss.str( );
}

/**
 * @return true if the neuron is in its absolute refractory period, else otherwise
 */
bool LifNeuron::isRefractory(void) {
	return ( nStepsInRefr > 0 );
}

#ifdef STORE_SPIKEHISTORY
/**
 * @return the number of spikes emitted.
 */
int LifNeuron::nSpikes(void) {
	return spikeHistory.size( );
}

/**
 * @return the count of spikes that occur at or after begin_time.
 */
int LifNeuron::nSpikesSince(uint64_t begin_step)
{
	int count = 0;
	for (size_t i = 0; i < spikeHistory.size(); i++)
	{
		if (spikeHistory[i] >= begin_step) count++;
	}
	return count;
}

/**
 * @returns a pointer to a vector of spike times.
 */
vector<uint64_t>* LifNeuron::getSpikes(void) {
	return &spikeHistory;
}
#endif // STORE_SPIKEHISTORY

/**
 * Set spikeCount to 0.
 */
void LifNeuron::clearSpikeCount(void) {
	spikeCount = 0;
}

/**
 * @return the spike count.
 */
int LifNeuron::getSpikeCount(void) {
	return spikeCount;
}

/**
 * @param[in] os	The filestream to write
 */
void LifNeuron::write(ostream& os) {
	os.write( reinterpret_cast<const char*>(&deltaT), sizeof(deltaT) );
	os.write( reinterpret_cast<const char*>(&Cm), sizeof(Cm) );
	os.write( reinterpret_cast<const char*>(&Rm), sizeof(Rm) );
	os.write( reinterpret_cast<const char*>(&Vthresh), sizeof(Vthresh) );
	os.write( reinterpret_cast<const char*>(&Vrest), sizeof(Vrest) );
	os.write( reinterpret_cast<const char*>(&Vreset), sizeof(Vreset) );
	os.write( reinterpret_cast<const char*>(&Vinit), sizeof(Vinit) );
	os.write( reinterpret_cast<const char*>(&Trefract), sizeof(Trefract) );
	os.write( reinterpret_cast<const char*>(&Inoise), sizeof(Inoise) );
	os.write( reinterpret_cast<const char*>(&Iinject), sizeof(Iinject) );
	os.write( reinterpret_cast<const char*>(&Isyn), sizeof(Isyn) );
	os.write( reinterpret_cast<const char*>(&nStepsInRefr), sizeof(nStepsInRefr) );
	os.write( reinterpret_cast<const char*>(&C1), sizeof(C1) );
	os.write( reinterpret_cast<const char*>(&C2), sizeof(C2) );
	os.write( reinterpret_cast<const char*>(&I0), sizeof(I0) );
	os.write( reinterpret_cast<const char*>(&Vm), sizeof(Vm) );
	os.write( reinterpret_cast<const char*>(&hasFired), sizeof(hasFired) );
	os.write( reinterpret_cast<const char*>(&Tau), sizeof(Tau) );
}

/**
 * @param[in] os	The filestream to read
 */
void LifNeuron::read(istream& is) {
	is.read( reinterpret_cast<char*>(&deltaT), sizeof(deltaT) );
	is.read( reinterpret_cast<char*>(&Cm), sizeof(Cm) );
	is.read( reinterpret_cast<char*>(&Rm), sizeof(Rm) );
	is.read( reinterpret_cast<char*>(&Vthresh), sizeof(Vthresh) );
	is.read( reinterpret_cast<char*>(&Vrest), sizeof(Vrest) );
	is.read( reinterpret_cast<char*>(&Vreset), sizeof(Vreset) );
	is.read( reinterpret_cast<char*>(&Vinit), sizeof(Vinit) );
	is.read( reinterpret_cast<char*>(&Trefract), sizeof(Trefract) );
	is.read( reinterpret_cast<char*>(&Inoise), sizeof(Inoise) );
	is.read( reinterpret_cast<char*>(&Iinject), sizeof(Iinject) );
	is.read( reinterpret_cast<char*>(&Isyn), sizeof(Isyn) );
	is.read( reinterpret_cast<char*>(&nStepsInRefr), sizeof(nStepsInRefr) );
	is.read( reinterpret_cast<char*>(&C1), sizeof(C1) );
	is.read( reinterpret_cast<char*>(&C2), sizeof(C2) );
	is.read( reinterpret_cast<char*>(&I0), sizeof(I0) );
	is.read( reinterpret_cast<char*>(&Vm), sizeof(Vm) );
	is.read( reinterpret_cast<char*>(&hasFired), sizeof(hasFired) );
	is.read( reinterpret_cast<char*>(&Tau), sizeof(Tau) );
}
