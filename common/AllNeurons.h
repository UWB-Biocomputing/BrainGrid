/** 
 * @authors Aaron Oziel, Sean Blackbourn
 * 
 *  In this file you will find usage statistics for every variable inthe BrainGrid 
 *  project as we find them. These statistics can be used to help 
 *  determine if a variable is being used, where it is being used, and how it
 *  is being used in each class::function() (either Modified or Accessed).
 *  
 *  For Example
 *
 *  Usage:
 *  Class::function():			Modified OR Accessed
 *  OtherClass::function():     Accessed   
 */
#pragma once

#ifndef _ALLNEURONS_H_
#define _ALLNEURONS_H_

#include "Global.h"

// Struct to hold all data necessary for all the Neurons.
struct AllNeurons
{
    public:

        /*! A boolean which tracks whether the neuron has fired
         *  
         *  Usage: LOCAL VARIABLE
         *  LIFSingleThreadedModel::advanceNeurons()		Modified
		 *  LIFSingleThreadedModel::fire()					Modified
		 *	GpuSim_struct.cu::advanceNeuronsDevice()		Modified
         */
        bool *hasFired;

        /*! The length of the absolute refractory period. [units=sec; range=(0,1);]
         *  
         *  Usage: LOCAL CONSTANT depending on a type of neuron
         *  LIFModel::createAllNeurons						Initialized
         *  LIFSingleThreadedModel::fire()					Accessed
		 *	GpuSim_struct.cu::advanceNeuronsDevice()		Accessed
         */
        BGFLOAT *Trefract;

        /*! If \f$V_m\f$ exceeds \f$V_{thresh}\f$ a spike is emmited. [units=V; range=(-10,100);]
         *  
         *  Usage: LOCAL CONSTANT
         *  LIFModel::createAllNeurons						Initialized
         *  LIFSingleThreadedModel::advanceNeuron()			Accessed
		 *	GpuSim_struct.cu::advanceNeuronsDevice()		Accessed
         */
        BGFLOAT *Vthresh;

        /*! The resting membrane voltage. [units=V; range=(-1,1);]
         *  
         *  Usage: LOCAL CONSTANT
         *  LIFModel::createAllNeurons											Initialized
         *  LIFSingleThreadedModel::initNeuronConstsFromParamValues()			Accessed
		 *	GpuSim_struct.cu::NOT USED
         */
        BGFLOAT *Vrest;

        /*! The voltage to reset \f$V_m\f$ to after a spike. [units=V; range=(-1,1);]
         *  
         *  Usage: LOCAL CONSTANT
         *  LIFModel::createAllNeurons						Initialized
         *  LIFSingleThreadedModel::fire()					Accessed
		 *	GpuSim_struct.cu::advanceNeuronsDevice()		Accessed
         */
        BGFLOAT *Vreset;

        /*! The initial condition for \f$V_m\f$ at time \f$t=0\f$. [units=V; range=(-1,1);]
         *
         *  Usage: LOCAL CONSTANT
         *  LIFModel::createAllNeurons				Initialized
         *  LIFModel::createAllNeurons()			Accessed
         *  GpuSim_struct.cu::NOT USED
         */
        BGFLOAT *Vinit;

        /*! The membrane capacitance \f$C_m\f$ [range=(0,1); units=F;]
         *  Used to initialize Tau (no use after that)
         *
         *  Usage: GLOBAL CONSTANT
         *  LIFModel::setNeuronDefaults				Initialized and accessed
         */
        BGFLOAT *Cm;

        /*! The membrane resistance \f$R_m\f$ [units=Ohm; range=(0,1e30)]
         *  
         *  Usage: GLOBAL CONSTANT
         *  LIFModel::setNeuronDefaults									Initialized and accessed
         *  LIFSingleThreadedModel::initNeuronConstsFromParamValues()	Accessed
		 *	GpuSim_struct.cu::NOT USED
         */
        BGFLOAT *Rm;

        /*! The standard deviation of the noise to be added each integration time constant. 
		 *	[range=(0,1); units=A;]
         *  
         *  Usage: LOCAL CONSTANT
         *  LIFModel::createAllNeurons						Initialized
         *  LIFSingleThreadedModel::advanceNeuron()			Accessed
		 *	GpuSim_struct.cu::advanceNeuronsDevice()		Accessed
         */
        BGFLOAT *Inoise;

        /*! A constant current to be injected into the LIF neuron. [units=A; range=(-1,1);]
         *  
         *  Usage: LOCAL CONSTANT
         *  LIFModel::createAllNeurons									Initialized
         *  LIFSingleThreadedModel::initNeuronConstsFromParamValues()	Accessed
		 *	GpuSim_struct.cu::NOT USED
         */
        BGFLOAT *Iinject;

        /*! What the hell is this used for???
		 *  It does not seem to be used; seems to be a candidate for deletion.
		 *  Possibly from the old code before using a separate summation point
         *  The synaptic input current.
         *  
         *  Usage: NOT USED ANYWHERE
         */
        BGFLOAT *Isyn;

        /*! The remaining number of time steps for the absolute refractory period.
         *  
         *  Usage: LOCAL VARIABLE
         *  LIFSingleThreadedModel::advanceNeuron()			Modified
		 *  LIFSingleThreadedModel::fire()					Modified
		 *	GpuSim_struct.cu::advanceNeuronsDevice()		Modified
         */
        int *nStepsInRefr;

        /*! Internal constant for the exponential Euler integration of f$V_m\f$.
         *  
         *  Usage: GLOBAL CONSTANT
         *  LIFSingleThreadedModel::initNeuronConstsFromParamValues()	Initialized
		 *  LIFSingleThreadedModel::advanceNeuron()						Accessed
		 *	GpuSim_struct.cu::advanceNeuronsDevice()					Accessed
         */
        BGFLOAT *C1;
        /*! Internal constant for the exponential Euler integration of \f$V_m\f$.
         *  
         *  Usage: GLOBAL CONSTANT
         *  LIFSingleThreadedModel::initNeuronConstsFromParamValues()	Initialized
		 *  LIFSingleThreadedModel::advanceNeuron()						Accessed
		 *	GpuSim_struct.cu::advanceNeuronsDevice()					Accessed
         */
        BGFLOAT *C2;
        /*! Internal constant for the exponential Euler integration of \f$V_m\f$.
         *  
         *  Usage: LOCAL CONSTANT
         *  LIFSingleThreadedModel::initNeuronConstsFromParamValues()	Initialized
		 *  LIFSingleThreadedModel::advanceNeuron()						Accessed
		 *	GpuSim_struct.cu::advanceNeuronsDevice()					Accessed
         */
        BGFLOAT *I0;
        /*! The membrane voltage \f$V_m\f$ [readonly; units=V;]
         *  
         *  Usage: LOCAL VARIABLE
         *  LIFSingleThreadedModel::advanceNeuron()			Modified
		 *  LIFSingleThreadedModel::fire()					Modified
		 *	GpuSim_struct.cu::advanceNeuronsDevice()		Modified
         */
        BGFLOAT *Vm;
        /*! The membrane time constant \f$(R_m \cdot C_m)\f$
         *  
         *  Usage: GLOBAL CONSTANT
         *  LIFModel::setNeuronDefaults									Initialized
         *  LIFSingleThreadedModel::initNeuronConstsFromParamValues()	Accessed
		 *	GpuSim_struct.cu::NOT USED
         */
        BGFLOAT *Tau;

        /*! The number of spikes since the last growth cycle
         *  
         *  Usage: LOCAL VARIABLE
		 *  LIFSingleThreadedModel::fire()					Modified
		 *  LIFSingleThreadedModel::updateHistory()			Accessed
		 *  LIFSingleThreadedModel::getSpikeCounts()		Accessed
		 *  LIFSingleThreadedModel::clearSpikeCounts()		Modified
		 *	GpuSim_struct.cu::getSpikeCounts()				Accessed
		 *	GpuSim_struct.cu::clearSpikeCounts()			Accessed
 		 *	GpuSim_struct.cu::advanceNeuronsDevice()		Modified
         */
        int *spikeCount;

		/*! Step count for each spike fired by each neuron
		 *
		 *  Usage: LOCAL VARIABLE
         *  LIFSingleThreadedModel::cleanupSim()			Accessed
		 *  LIFSingleThreadedModel::fire()					Modified
		 *	GpuSim_struct.cu::NOT USED
         */
        uint64_t **spike_history;

        /*! The neuron type map (INH, EXC).
         *  
         *  Usage: LOCAL CONSTANT
         *  LIFModel::generateNeuronTypeMap					Initialized
         *  LIFSingleThreadedModel::logSimStep()			Accessed
		 *  LIFSingleThreadedModel::synType()				Accessed
		 *	GpuSim_struct.cu::NOT USED
         */
        neuronType *neuron_type_map;

        /*! List of summation points for each neuron
         *  
         *  Usage: LOCAL VARIABLE
         *  LIFSingleThreadedModel::advanceNeuron()			Modified
		 *  LIFSingleThreadedModel::updateWeights()			Accessed
		 *	GpuSim_struct.cu::NOT USED
         */
        BGFLOAT *summation_map;

        /*! The starter existence map (T/F).
         *  
         *  Usage: LOCAL CONSTANT
         *  LIFModel::initStarterMap					Initialized
         *  LIFSingleThreadedModel::logSimStep()		Accessed
		 *	GpuSim_struct.cu::NOT USED
         */
        bool *starter_map;

	

        AllNeurons();
        AllNeurons(const int size);
        ~AllNeurons();

    private:

        int size;
};

#endif
