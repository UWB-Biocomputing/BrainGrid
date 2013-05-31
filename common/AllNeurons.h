#pragma once

#ifndef _ALLNEURONS_H_
#define _ALLNEURONS_H_

#include "Global.h"

// Struct to hold all data nessesary for all the Neurons.
struct AllNeurons
{
    public:
        //! The number of neurons stored.
        uint32_t size;

        //! A boolean which tracks whether the neuron has fired
        vector<GPU_COMPAT_BOOL> hasFired; // used as a bool (must be int sized/cannot be bool because of incompatibility with AMP)

        //! The length of the absolute refractory period. [units=sec; range=(0,1);]
        vector<BGFLOAT> Trefract;

        //! If \f$V_m\f$ exceeds \f$V_{thresh}\f$ a spike is emmited. [units=V; range=(-10,100);]
        vector<BGFLOAT> Vthresh;

        //! The resting membrane voltage. [units=V; range=(-1,1);]
        vector<BGFLOAT> Vrest;

        //! The voltage to reset \f$V_m\f$ to after a spike. [units=V; range=(-1,1);]
        vector<BGFLOAT> Vreset;

        //! The initial condition for \f$V_m\f$ at time \f$t=0\f$. [units=V; range=(-1,1);]
        vector<BGFLOAT> Vinit;
        //! The simulation time step size.
        vector<TIMEFLOAT> deltaT; // must be double for compatibility with GPU code

        //! The membrane capacitance \f$C_m\f$ [range=(0,1); units=F;]
        vector<BGFLOAT> Cm;

        //! The membrane resistance \f$R_m\f$ [units=Ohm; range=(0,1e30)]
        vector<BGFLOAT> Rm;

        //! The standard deviation of the noise to be added each integration time constant. [range=(0,1); units=A;]
		vector<BGFLOAT> Inoise;

        //! A constant current to be injected into the LIF neuron. [units=A; range=(-1,1);]
        vector<BGFLOAT> Iinject;

        // What the hell is this used for???
        //! The synaptic input current.
        vector<BGFLOAT> Isyn;

        //! The remaining number of time steps for the absolute refractory period.
        vector<uint32_t> nStepsInRefr;

        //! Internal constant for the exponential Euler integration of \f$V_m\f$.
        vector<BGFLOAT> C1;
        //! Internal constant for the exponential Euler integration of \f$V_m\f$.
        vector<BGFLOAT> C2;
        //! Internal constant for the exponential Euler integration of \f$V_m\f$.
        vector<BGFLOAT> I0;

        //! The membrane voltage \f$V_m\f$ [readonly; units=V;]
        vector<BGFLOAT> Vm;

        //! The membrane time constant \f$(R_m \cdot C_m)\f$
        vector<BGFLOAT> Tau;

        //! The number of spikes since the last growth cycle
        vector<uint32_t> spikeCount;

        //! The total number of spikes since the start of the simulation
        vector<uint32_t> totalSpikeCount;        

        uint64_t **spike_history;

        //! The neuron type map (INH, EXC).
        vector<neuronType> neuron_type_map;

        // Each synapse (for this neuron) keeps a total summation here
        vector<BGFLOAT> summation;

        //! The starter existence map (T/F).
        bool *starter_map;

        AllNeurons();
        AllNeurons(const int size);
        ~AllNeurons();

};

#endif
