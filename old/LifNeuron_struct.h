  /**
 **
 ** \brief A leaky-integrate-and-fire (I&F) neuron structure.
 **
 ** \struct LifNeuron_struct LifNeuron_struct.h "LifNeuron_struct.h"
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
 * \file LifNeuron_struct.h
 *
 * Header file for LifNeuron_struct
 */

#ifndef _LIFNEURON_STRUCT_H_
#define _LIFNEURON_STRUCT_H_

#include <cstdlib>
#include "global.h"
#include "LifNeuron.h"
#include "DynamicSpikingSynapse_struct.h"

// forward declaration
struct LifNeuron_struct;

/**
 * Allocate data members in the LifNeuron_struct.
 * @param neuron
 * @param count
 */
void allocNeuronStruct(LifNeuron_struct* neuron, int count);

/**
 * Deallocate data members in the LifNeuron_struct.
 * @param neuron
 */
void deleteNeuronStruct(LifNeuron_struct* neuron);

/**
 * Copy INeuron data into a LifNeuron_struct for GPU processing.
 * @param in
 * @param out
 * @param idx
 */
void copyNeuronToStruct(INeuron* in, LifNeuron_struct* out, int idx);

/**
 * Copy LifNeuron_struct array data into a INeuron.
 * @param in
 * @param out
 * @param idx
 */
void copyNeuronStructToNeuron(LifNeuron_struct* in, INeuron* out, int idx);

/**
 * Copy a neuronArray into a neuronMap
 * @param neuron
 * @param pNeuronList
 * @param numNeurons
 */
void neuronArrayToMap(LifNeuron_struct* neuron, vector<INeuron*>* pNeuronList, int numNeurons);

struct LifNeuron_struct {
	
	//! The simulation time step size.
	BGFLOAT* deltaT;

	//! The pooling location for all incoming spikes.
	PBGFLOAT* summationPoint;

	//! The membrane capacity \f$C_m\f$ [range=(0,1); units=F;]
	BGFLOAT* Cm;

	//! The membrane resistance \f$R_m\f$ [units=Ohm; range=(0,1e30)]
	BGFLOAT* Rm;

	//! If \f$V_m\f$ exceeds \f$V_{thresh}\f$ a spike is emmited. [units=V; range=(-10,100);]
	BGFLOAT* Vthresh;

	//! The resting membrane voltage. [units=V; range=(-1,1);]
	BGFLOAT* Vrest;

	//! The voltage to reset \f$V_m\f$ to after a spike. [units=V; range=(-1,1);]
	BGFLOAT* Vreset;

	//! The initial condition for \f$V_m\f$ at time \f$t=0\f$. [units=V; range=(-1,1);]
	BGFLOAT* Vinit;

	//! The length of the absolute refractory period. [units=sec; range=(0,1);]
	BGFLOAT* Trefract;

	//! The standard deviation of the noise to be added each integration time constant. [range=(0,1); units=A;]
	BGFLOAT* Inoise;

	//! A Random multiplier to Inoise. [range=(0,1)]
	float** randNoise;

	//! A constant current to be injected into the LIF neuron. [units=A; range=(-1,1);]
	BGFLOAT* Iinject;

	//! The synaptic input current.
	BGFLOAT* Isyn;

	//! The remaining number of time steps for the absolute refractory period.
	int* nStepsInRefr;

	//! Internal constant for the exponential Euler integration of \f$V_m\f$.
	BGFLOAT* C1;
	//! Internal constant for the exponential Euler integration of \f$V_m\f$.
	BGFLOAT* C2;
	//! Internal constant for the exponential Euler integration of \f$V_m\f$.
	BGFLOAT* I0;

	//! The membrane voltage \f$V_m\f$ [readonly; units=V;]
	BGFLOAT* Vm;

	 //! A boolean which tracks whether the neuron has fired
        bool* hasFired;

	//! The membrane time constant \f$(R_m \cdot C_m)\f$
	BGFLOAT* Tau;

	//! The number of spikes since the last growth cycle.
	int* spikeCount;

	//! The beginning index of the outgoing dynamic spiking synapse array.
	int* outgoingSynapse_begin;

	//! The number of outgoing synapses.
	int* synapseCount;

	//! The beginning index of the incoming dynamic spiking synapse array.
	int* incomingSynapse_begin;

	//! The number of incoming synapses.
	int* inverseCount;	

	//! The number of neurons.
	int* numNeurons;

	//! the TSim from the input file
	int* stepDuration;    // NOTE: unused. TODO: delete??
};

#endif
