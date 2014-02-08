/** 
 * @authors Aaron Oziel, Sean Blackbourn
 * 
 *  In this file you will find usage statistics for every variable inside the
 *  the LIFCingleThreaded.cpp file. These statistics can be used to help 
 *  determine if a variable is being used, where it is being used, and how it
 *  is being used in each function (either Modified or Accessed).
 *  
 *  For Example
 *
 *  Usage:
 *  function():		Modified OR Accessed
 *  function():     Accessed   
 */
#pragma once

#ifndef _ALLNEURONS_H_
#define _ALLNEURONS_H_

#include "Global.h"

// Struct to hold all data necessary for all the Neurons.
struct AllNeurons
{
    public:
        /*! The number of neurons stored.
         *  
         *  Usage:
         *  advanceNeurons()	Accessed
		 *  updateHistory()		Accessed
		 *  getSpikeCounts()	Accessed
		 *  clearSpikeCounts()	Accessed
         */
        int size;

        /*! A boolean which tracks whether the neuron has fired
         *  
         *  Usage:
         *  advanceNeurons()	Modified
		 *  fire()				Modified
         */
        bool *hasFired;

        /*! The length of the absolute refractory period. [units=sec; range=(0,1);]
         *  
         *  Usage:
         *  fire()				Accessed
         */
        BGFLOAT *Trefract;

        /*! If \f$V_m\f$ exceeds \f$V_{thresh}\f$ a spike is emmited. [units=V; range=(-10,100);]
         *  
         *  Usage:
         *  advanceNeuron()		Modified
         */
        BGFLOAT *Vthresh;

        /*! The resting membrane voltage. [units=V; range=(-1,1);]
         *  
         *  Usage:
         *  updateNeuron()		Accessed
         */
        BGFLOAT *Vrest;

        /*! The voltage to reset \f$V_m\f$ to after a spike. [units=V; range=(-1,1);]
         *  
         *  Usage:
         *  fire()				Accessed
         */
        BGFLOAT *Vreset;

        /*! The initial condition for \f$V_m\f$ at time \f$t=0\f$. [units=V; range=(-1,1);]
         *  
         *  Usage:
         *  NOT USED
         */
        BGFLOAT *Vinit;
		
        /*! The simulation time step size.
         *  
         *  Usage:
		 *  updateNeuron()		Accessed
         *  fire()				Accessed
         */
        BGFLOAT *deltaT;

        /*! The membrane capacitance \f$C_m\f$ [range=(0,1); units=F;]
         *  
         *  Usage:
         *  NOT USED
         */
        BGFLOAT *Cm;

        /*! The membrane resistance \f$R_m\f$ [units=Ohm; range=(0,1e30)]
         *  
         *  Usage:
         *  updateNeuron()		Accessed
         */
        BGFLOAT *Rm;

        /*! The standard deviation of the noise to be added each integration time constant. 
		 *	[range=(0,1); units=A;]
         *  
         *  Usage:
         *  advanceNeuron()		Accessed
         */
        BGFLOAT *Inoise;

        /*! A constant current to be injected into the LIF neuron. [units=A; range=(-1,1);]
         *  
         *  Usage:
         *  updateNeuron()		Accessed
         */
        BGFLOAT *Iinject;

        /*! What the hell is this used for???
         *  The synaptic input current.
         *  
         *  Usage:
         *  NOT USED
         */
        BGFLOAT *Isyn;

        /*! The remaining number of time steps for the absolute refractory period.
         *  
         *  Usage:
         *  advanceNeuron()		Modified
		 *  fire()				Modified
         */
        int *nStepsInRefr;

        /*! Internal constant for the exponential Euler integration of f$V_m\f$.
         *  
         *  Usage:
         *  updateNeuron()		Modified
		 *  advanceNeuron()		Accessed
         */
        BGFLOAT *C1;
        /*! Internal constant for the exponential Euler integration of \f$V_m\f$.
         *  
         *  Usage:
         *  
         */
        BGFLOAT *C2;
        /*! Internal constant for the exponential Euler integration of \f$V_m\f$.
         *  
         *  Usage:
         *  updateNeuron()		Modified
		 *  advanceNeuron()		Accessed
         */
        BGFLOAT *I0;

        /*! The membrane voltage \f$V_m\f$ [readonly; units=V;]
         *  
         *  Usage:
         *  advanceNeuron()		Modified
		 *  fire()				Modified
         */
        BGFLOAT *Vm;

        /*! The membrane time constant \f$(R_m \cdot C_m)\f$
         *  
         *  Usage:
         *  updateNeuron()		Accessed
         */
        BGFLOAT *Tau;

        /*! The number of spikes since the last growth cycle
         *  
         *  Usage:
		 *  fire()				Modified
		 *  updateHistory()		Accessed
		 *  getSpikeCounts()	Accessed
		 *  clearSpikeCounts()	Modified
         */
        int *spikeCount;

        /*! The total number of spikes since the start of the simulation
         *  
         *  Usage:
         *  cleanupSim()		Accessed
		 *  fire()				Modified
         */
        int *totalSpikeCount;        

		/*! TODO
		 *
		 *  Usage:
         *  cleanupSim()		Accessed
		 *  fire()				Modified
         */
        uint64_t **spike_history;

        /*! The neuron type map (INH, EXC).
         *  
         *  Usage:
         *  logSimStep()		Accessed
		 *  synType()			Accessed
         */
        neuronType *neuron_type_map;

        /*! List of summation points for each neuron
         *  
         *  Usage:
         *  advanceNeuron()		Modified
		 *  updateWeights()		Accessed
         */
        BGFLOAT *summation_map;

        /*! The starter existence map (T/F).
         *  
         *  Usage:
         *  logSimStep()		Accessed
         */
        bool *starter_map;

        AllNeurons();
        AllNeurons(const int size);
        ~AllNeurons();

};

#endif
