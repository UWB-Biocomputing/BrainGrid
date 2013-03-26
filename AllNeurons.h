#pragma once

#ifndef _ALLNEURONS_H_
#define _ALLNEURONS_H_

#include "global.h"

//	struct to hold all data nessesary for all the Neurons.
struct AllNeurons
{
    public:
        //! The number of neurons stored.
        int size;

        //! A boolean which tracks whether the neuron has fired
        bool *hasFired;

        //! The length of the absolute refractory period. [units=sec; range=(0,1);]
        BGFLOAT *Trefract;

        //! If \f$V_m\f$ exceeds \f$V_{thresh}\f$ a spike is emmited. [units=V; range=(-10,100);]
        BGFLOAT *Vthresh;

        //! The resting membrane voltage. [units=V; range=(-1,1);]
        BGFLOAT *Vrest;

        //! The voltage to reset \f$V_m\f$ to after a spike. [units=V; range=(-1,1);]
        BGFLOAT *Vreset;

        //! The initial condition for \f$V_m\f$ at time \f$t=0\f$. [units=V; range=(-1,1);]
        BGFLOAT *Vinit;
        //! The simulation time step size.
        BGFLOAT *deltaT;

        //! The membrane capacitance \f$C_m\f$ [range=(0,1); units=F;]
        BGFLOAT *Cm;

        //! The membrane resistance \f$R_m\f$ [units=Ohm; range=(0,1e30)]
        BGFLOAT *Rm;

        //! The standard deviation of the noise to be added each integration time constant. [range=(0,1); units=A;]
        BGFLOAT *Inoise;

        //! A constant current to be injected into the LIF neuron. [units=A; range=(-1,1);]
        BGFLOAT *Iinject;

        // What the hell is this used for???
        //! The synaptic input current.
        BGFLOAT *Isyn;

        //! The remaining number of time steps for the absolute refractory period.
        int *nStepsInRefr;

        //! Internal constant for the exponential Euler integration of \f$V_m\f$.
        BGFLOAT *C1;
        //! Internal constant for the exponential Euler integration of \f$V_m\f$.
        BGFLOAT *C2;
        //! Internal constant for the exponential Euler integration of \f$V_m\f$.
        BGFLOAT *I0;

        //! The membrane voltage \f$V_m\f$ [readonly; units=V;]
        BGFLOAT *Vm;

        //! The membrane time constant \f$(R_m \cdot C_m)\f$
        BGFLOAT *Tau;

        //! The number of spikes since the last growth cycle
        int *spikeCount;
        
        uint64_t **spike_history;

        //! The neuron type map (INH, EXC).
        neuronType *neuron_type_map;

        // List of summation points for each neuron
        BGFLOAT *summation_map;

        //! The starter existence map (T/F).
        bool *starter_map;

        AllNeurons();
        AllNeurons(const int size);
        ~AllNeurons();

};

#endif
