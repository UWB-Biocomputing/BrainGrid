/**
 ** \file INeuron.cpp
 **/

#include "INeuron.h"

/**
 * Create a neuron and initialize all internal state vars using 
 * the default values in global.cpp.
 * @post The neuron is setup according to the default values. 
 */
INeuron::INeuron() :
    deltaT(DEFAULT_dt), 
    Cm(DEFAULT_Cm),
    Rm(DEFAULT_Rm),
    Vthresh(DEFAULT_Vthresh), 
    Vrest(DEFAULT_Vrest), 
    Vreset(DEFAULT_Vreset), 
    Vinit(DEFAULT_Vreset), 
    Trefract(DEFAULT_Trefract), 
    Inoise(DEFAULT_Inoise), 
    Iinject(DEFAULT_Iinject),
    Tau(DEFAULT_Cm * DEFAULT_Rm) 
{ 
}

/**
* Destructor
*
*/
INeuron::~INeuron() {
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
void INeuron::setParams(BGFLOAT new_Iinject, BGFLOAT new_Inoise, BGFLOAT new_Vthresh, BGFLOAT new_Vresting,
	BGFLOAT new_Vreset, BGFLOAT new_Vinit, BGFLOAT new_deltaT) {
	Iinject = new_Iinject;
	Inoise = new_Inoise;
	Vthresh = new_Vthresh;
	Vrest = new_Vresting;
	Vreset = new_Vreset;
	Vinit = new_Vinit;
	deltaT = new_deltaT;
	reset( );
}

/**
 * Reset time varying state vars.
 * If STORE_SPIKEHISTORY is set, reset spike history.
 */
void INeuron::reset() {
	hasFired = false;
#ifdef STORE_SPIKEHISTORY
	spikeHistory.clear( );
#endif // STORE_SPIKEHISTORY
	nStepsInRefr = 0;
	Vm = Vinit;
	this->updateInternal( );
	spikeCount = 0;
}

/**
 * @return a string with the Vm.
 */
string INeuron::toStringVm() {
	stringstream ss;
	ss << Vm;
	return ss.str( );
}

/**
 * @return a terse representation.
 */
string INeuron::toString() {
	stringstream ss;
	if (hasFired) ss << "!! FIRED !!    ";
	ss << "Vm: " << Vm;
	if (isRefractory( )) ss << "      refract: " << nStepsInRefr;
	return ss.str( );
}

/**
 * @return the complete state of the neuron.
 */
string INeuron::toStringAll() {
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
bool INeuron::isRefractory() {
	return ( nStepsInRefr > 0 );
}

#ifdef STORE_SPIKEHISTORY
/**
 * @return the number of spikes emitted.
 */
int INeuron::nSpikes() {
	return spikeHistory.size( );
}

/**
 * @return the count of spikes that occur at or after begin_time.
 */
int INeuron::nSpikesSince(uint64_t begin_step)
{
	int count = 0;
	for (size_t i = 0; i < spikeHistory.size(); i++)
	{
		if (spikeHistory[i] >= begin_step) 
			count++;
	}
	return count;
}

/**
 * @returns a pointer to a vector of spike times.
 */
vector<uint64_t>* INeuron::getSpikes() {
	return &spikeHistory;
}
#endif // STORE_SPIKEHISTORY

/**
 * Set spikeCount to 0.
 */
void INeuron::clearSpikeCount() {
	spikeCount = 0;
}

/**
 * @return the spike count.
 */
int INeuron::getSpikeCount() {
	return spikeCount;
}

/**
 * Write the neuron data to the stream
 * @param[in] os	The filestream to write
 */
void INeuron::write(ostream& os) {
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
 * Read the neuron data from the stream
 * @param[in] is	The filestream to read
 */
void INeuron::read(istream& is) {
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
