/**
 **
 ** \brief A leaky-integrate-and-fire (I&F) neuron.
 **
 ** \class LifNeuron LifNeuron.h "LifNeuron.h"
 **
 ** \latexonly  \subsubsection*{Model} \endlatexonly
 ** \htmlonly  <h3>Model</h3> \endhtmlonly
 **
 ** A standard leaky-integrate-and-fire neuron model is implemented
 ** where the membrane potential \f$V_m\f$ of a neuron is given by
 ** \f[
 **   \tau_m \frac{d V_m}{dt} = -(V_m-V_{resting}) + R_m \cdot (I_{syn}(t)+I_{inject}+I_{noise})
 ** \f]
 ** where \f$\tau_m=C_m\cdot R_m\f$ is the membrane time constant,
 ** \f$R_m\f$ is the membrane resistance, \f$I_{syn}(t)\f$ is the
 ** current supplied by the synapses, \f$I_{inject}\f$ is a
 ** non-specific background current and \f$I_{noise}\f$ is a
 ** Gaussian random variable with zero mean and a given variance
 ** noise.
 **
 ** At time \f$t=0\f$ \f$V_m\f$ is set to \f$V_{init}\f$. If
 ** \f$V_m\f$ exceeds the threshold voltage \f$V_{thresh}\f$ it is
 ** reset to \f$V_{reset}\f$ and hold there for the length
 ** \f$T_{refract}\f$ of the absolute refractory period.
 **
 ** \latexonly  \subsubsection*{Implementation} \endlatexonly
 **
 ** \htmlonly  <h3>Implementation</h3> \endhtmlonly
 ** The exponential Euler method is used for numerical integration.
 **
 **	@authors Allan Ortiz & Cory Mayberry
 **/

/**
 * \file LifNeuron.h
 *
 * Header file for LifNeuron
 */

#ifndef _LIFNEURON_H_
#define _LIFNEURON_H_

#include <cstdlib>
#include "global.h"

class LifNeuron {
public:

	//! Constructor.
	LifNeuron();
	~LifNeuron();

	//! Set the neuron parameters.
	void setParams(FLOAT new_Iinject, FLOAT new_Inoise, FLOAT new_Vthresh, FLOAT new_Vresting,
			FLOAT new_Vreset, FLOAT new_Vinit, FLOAT new_deltaT);

	//! Process another time step.
	void advance(FLOAT& summationPoint);

	//! Reset to initial state.
	void reset();

	//! Return a terse representation.
	string toString();

	//! Return a string with the Vm.
	string toStringVm();

	//! Return the complete state of the neuron.
	string toStringAll();

	//! Return refractory state.
	bool isRefractory();

	//! Returns the actual membrane voltage.
	inline FLOAT getVm() const;

	//! Emit a spike.
	void fire();

	//! Update internal constants.
	void updateInternal();

	//! The simulation time step size.
	FLOAT deltaT;

	//! The membrane capacity \f$C_m\f$ [range=(0,1); units=F;]
	FLOAT Cm;

	//! The membrane resistance \f$R_m\f$ [units=Ohm; range=(0,1e30)]
	FLOAT Rm;

	//! If \f$V_m\f$ exceeds \f$V_{thresh}\f$ a spike is emmited. [units=V; range=(-10,100);]
	FLOAT Vthresh;

	//! The resting membrane voltage. [units=V; range=(-1,1);]
	FLOAT Vrest;

	//! The voltage to reset \f$V_m\f$ to after a spike. [units=V; range=(-1,1);]
	FLOAT Vreset;

	//! The initial condition for \f$V_m\f$ at time \f$t=0\f$. [units=V; range=(-1,1);]
	FLOAT Vinit;

	//! The length of the absolute refractory period. [units=sec; range=(0,1);]
	FLOAT Trefract;

	//! The standard deviation of the noise to be added each integration time constant. [range=(0,1); units=A;]
	FLOAT Inoise;

	//! A constant current to be injected into the LIF neuron. [units=A; range=(-1,1);]
	FLOAT Iinject;

	//! The synaptic input current.
	FLOAT Isyn;

	//! The remaining number of time steps for the absolute refractory period.
	int nStepsInRefr;

	//! Internal constant for the exponential Euler integration of \f$V_m\f$.
	FLOAT C1;
	//! Internal constant for the exponential Euler integration of \f$V_m\f$.
	FLOAT C2;
	//! Internal constant for the exponential Euler integration of \f$V_m\f$.
	FLOAT I0;

	//! The membrane voltage \f$V_m\f$ [readonly; units=V;]
	FLOAT Vm;

	//! A boolean which tracks whether the neuron has fired
	bool hasFired;

	//! The membrane time constant \f$(R_m \cdot C_m)\f$
	FLOAT Tau;

#ifdef STORE_SPIKEHISTORY
	//! The history of spike times.
	vector <uint64_t> spikeHistory;

	//! Return the number of spikes emitted.
	int nSpikes(void);

	//! Returns a pointer to a vector of spike times
	vector<uint64_t>* getSpikes(void);

	//! Return the count of spikes that occur at or after begin_time
	int nSpikesSince(uint64_t begin_step);
#endif // STORE_SPIKEHISTORY

	//! The number of spikes since the last growth cycle
	int spikeCount;

	//! Clear the spike count
	void clearSpikeCount(void);

	//! Return the spike count
	int getSpikeCount(void);

	//! Write the neuron data to the stream
	void write(ostream& os);

	//! Read the neuron data from the stream
	void read(istream& is);
};

#endif
