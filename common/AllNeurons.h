/** 
 * @authors Aaron Oziel, Sean Blackbourn
 * 
 * @class AllNeurons AllNeurons.h "AllNeurons.h"
 * @brief A container of all neuron data
 *
 *  The container holds neuron parameters of all neurons. 
 *  Each kind of neuron parameter is stored in a 1D array, of which length
 *  is number of all neurons. Each array of a neuron parameter is pointed by a 
 *  corresponding member variable of the neuron parameter in the class.
 *
 *  In this file you will find usage statistics for every variable in the BrainGrid 
 *  project as we find them. These statistics can be used to help 
 *  determine if a variable is being used, where it is being used, and how it
 *  is being used in each class::function()
 *  
 *  For Example
 *
 *  Usage:
 *  - LOCAL VARIABLE -- a variable for individual neuron
 *  - LOCAL CONSTANT --  a constant for individual neuron
 *  - GLOBAL VARIABLE -- a variable for all neurons
 *  - GLOBAL CONSTANT -- a constant for all neurons
 *
 *  Class::function(): --- Initialized, Modified OR Accessed
 *
 *  OtherClass::function(): --- Accessed   
 *
 *  Note: All GLOBAL parameters can be scalars. Also some LOCAL CONSTANT can be categorized 
 *  depending on neuron types. 
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
         *  - AllNeurons::AllNeurons() --- Initialized
         *  - LIFModel::readNeuron() --- Modified
         *  - LIFModel::writeNeuron() --- Accessed
         *  - LIFSingleThreadedModel::advanceNeurons() --- Accessed & Modified
         *  - LIFSingleThreadedModel::fire() --- Modified
         *  - GpuSim_struct.cu::advanceNeuronsDevice() --- Modified
         */
        bool *hasFired;

        /*! The length of the absolute refractory period. [units=sec; range=(0,1);]
         *  
         *  Usage: LOCAL CONSTANT depending on a type of neuron
         *  - LIFModel::createAllNeurons() --- Initialized
         *  - LIFModel::setNeuronDefaults() --- Initialized
         *  - LIFModel::readNeuron() --- Modified
         *  - LIFModel::writeNeuron() --- Accessed
         *  - LIFSingleThreadedModel::fire() --- Accessed
         *  - GpuSim_struct.cu::advanceNeuronsDevice() --- Accessed
         */
        BGFLOAT *Trefract;

        /*! If \f$V_m\f$ exceeds \f$V_{thresh}\f$ a spike is emmited. [units=V; range=(-10,100);]
         *  
         *  Usage: LOCAL CONSTANT
         *  - LIFModel::createAllNeurons() --- Initialized
         *  - LIFModel::setNeuronDefaults() --- Initialized
         *  - LIFModel::readNeuron() --- Modified
         *  - LIFModel::writeNeuron() --- Accessed
         *  - LIFSingleThreadedModel::advanceNeuron() --- Accessed
         *  - GpuSim_struct.cu::advanceNeuronsDevice() --- Accessed
         *  - Hdf5Recorder::saveSimState() --- Accessed
         *  - XmlRecorder::saveSimState() --- Accessed
         */
        BGFLOAT *Vthresh;

        /*! The resting membrane voltage. [units=V; range=(-1,1);]
         *  
         *  Usage: LOCAL CONSTANT
         *  - LIFModel::createAllNeurons() --- Initialized
         *  - LIFModel::setNeuronDefaults() --- Initialized
         *  - LIFModel::readNeuron() --- Modified
         *  - LIFModel::writeNeuron() --- Accessed
         *  - LIFModel::initNeuronConstsFromParamValues() --- Accessed
         */
        BGFLOAT *Vrest;

        /*! The voltage to reset \f$V_m\f$ to after a spike. [units=V; range=(-1,1);]
         *  
         *  Usage: LOCAL CONSTANT
         *  - LIFModel::createAllNeurons() --- Initialized
         *  - LIFModel::setNeuronDefaults() --- Initialized
         *  - LIFModel::readNeuron() --- Modified
         *  - LIFModel::writeNeuron() --- Accessed
         *  - LIFSingleThreadedModel::fire() --- Accessed
         *  - GpuSim_struct.cu::advanceNeuronsDevice() --- Accessed
         */
        BGFLOAT *Vreset;

        /*! The initial condition for \f$V_m\f$ at time \f$t=0\f$. [units=V; range=(-1,1);]
         *
         *  Usage: LOCAL CONSTANT
         *  - LIFModel::createAllNeurons() --- Initialized & Accessed
         *  - LIFModel::setNeuronDefaults() --- Initialized
         *  - LIFModel::readNeuron() --- Modified
         *  - LIFModel::writeNeuron() --- Accessed
         */
        BGFLOAT *Vinit;

        /*! The membrane capacitance \f$C_m\f$ [range=(0,1); units=F;]
         *  Used to initialize Tau (no use after that)
         *
         *  Usage: GLOBAL CONSTANT
         *  - LIFModel::setNeuronDefaults() --- Initialized and accessed
         *  - LIFModel::readNeuron() --- Modified
         *  - LIFModel::writeNeuron() --- Accessed
         */
        BGFLOAT *Cm;

        /*! The membrane resistance \f$R_m\f$ [units=Ohm; range=(0,1e30)]
         *  
         *  Usage: GLOBAL CONSTANT
         *  - LIFModel::setNeuronDefaults() --- Initialized
         *  - LIFModel::initNeuronConstsFromParamValues() --- Accessed
         *  - LIFModel::readNeuron() --- Modified
         *  - LIFModel::writeNeuron() --- Accessed
         */
        BGFLOAT *Rm;

        /*! The standard deviation of the noise to be added each integration time constant. [range=(0,1); units=A;]
         *  
         *  Usage: LOCAL CONSTANT
         *  - LIFModel::createAllNeurons() --- Initialized
         *  - LIFModel::setNeuronDefaults() --- Initialized
         *  - LIFModel::readNeuron() --- Modified
         *  - LIFModel::writeNeuron() --- Accessed
         *  - LIFSingleThreadedModel::advanceNeuron() --- Accessed
         *  - GpuSim_struct.cu::advanceNeuronsDevice() --- Accessed
         */
        BGFLOAT *Inoise;

        /*! A constant current to be injected into the LIF neuron. [units=A; range=(-1,1);]
         *  
         *  Usage: LOCAL CONSTANT
         *  - LIFModel::createAllNeurons() --- Initialized
         *  - LIFModel::setNeuronDefaults() --- Initialized
         *  - LIFModel::readNeuron() --- Modified
         *  - LIFModel::writeNeuron() --- Accessed
         *  - LIFModel::initNeuronConstsFromParamValues() --- Accessed
         */
        BGFLOAT *Iinject;

        /*! What the hell is this used for???
         *  It does not seem to be used; seems to be a candidate for deletion.
         *  Possibly from the old code before using a separate summation point
         *  The synaptic input current.
         *  
         *  Usage: NOT USED ANYWHERE
         *  - LIFModel::readNeuron() --- Modified
         *  - LIFModel::writeNeuron() --- Accessed
         */
        BGFLOAT *Isyn;

        /*! The remaining number of time steps for the absolute refractory period.
         *  
         *  Usage: LOCAL VARIABLE
         *  - AllNeurons::AllNeurons() --- Initialized
         *  - LIFModel::readNeuron() --- Modified
         *  - LIFModel::writeNeuron() --- Accessed
         *  - LIFSingleThreadedModel::advanceNeuron() --- Accessed & Modified
         *  - LIFSingleThreadedModel::fire() --- Modified
         *  - GpuSim_struct.cu::advanceNeuronsDevice() --- Accessed & Modified
         */
        int *nStepsInRefr;

        /*! Internal constant for the exponential Euler integration of f$V_m\f$.
         *  
         *  Usage: GLOBAL CONSTANT
         *  - LIFModel::initNeuronConstsFromParamValues() --- Initialized
         *  - LIFModel::readNeuron() --- Modified
         *  - LIFModel::writeNeuron() --- Accessed
         *  - LIFSingleThreadedModel::advanceNeuron() --- Accessed
         *  - GpuSim_struct.cu::advanceNeuronsDevice() --- Accessed
         */
        BGFLOAT *C1;

        /*! Internal constant for the exponential Euler integration of \f$V_m\f$.
         *  
         *  Usage: GLOBAL CONSTANT
         *  - LIFModel::initNeuronConstsFromParamValues() --- Initialized
         *  - LIFModel::readNeuron() --- Modified
         *  - LIFModel::writeNeuron() --- Accessed
         *  - LIFSingleThreadedModel::advanceNeuron() --- Accessed
         *  - GpuSim_struct.cu::advanceNeuronsDevice() --- Accessed
         */
        BGFLOAT *C2;

        /*! Internal constant for the exponential Euler integration of \f$V_m\f$.
         *  
         *  Usage: LOCAL CONSTANT
         *  - LIFModel::initNeuronConstsFromParamValues() --- Initialized
         *  - LIFModel::readNeuron() --- Modified
         *  - LIFModel::writeNeuron() --- Accessed
         *  - LIFSingleThreadedModel::advanceNeuron() --- Accessed
         *  - GpuSim_struct.cu::advanceNeuronsDevice() --- Accessed
         */
        BGFLOAT *I0;

        /*! The membrane voltage \f$V_m\f$ [readonly; units=V;]
         *  
         *  Usage: LOCAL VARIABLE
         *  - LIFModel::createAllNeurons() --- Initialized
         *  - LIFModel::readNeuron() --- Modified
         *  - LIFModel::writeNeuron() --- Accessed
         *  - LIFSingleThreadedModel::advanceNeuron() --- Accessed & Modified
         *  - LIFSingleThreadedModel::fire() --- Modified
         *  - GpuSim_struct.cu::advanceNeuronsDevice() --- Accessed & Modified
         */
        BGFLOAT *Vm;

        /*! The membrane time constant \f$(R_m \cdot C_m)\f$
         *  
         *  Usage: GLOBAL CONSTANT
         *  - LIFModel::setNeuronDefaults() --- Initialized
         *  - LIFModel::readNeuron() --- Modified
         *  - LIFModel::writeNeuron() --- Accessed
         *  - LIFModel::initNeuronConstsFromParamValues() --- Accessed
         */
        BGFLOAT *Tau;

        /*! The number of spikes since the last growth cycle
         *  
         *  Usage: LOCAL VARIABLE
         *  - AllNeurons::AllNeurons() --- Initialized
         *  - LIFModel::updateHistory() --- Accessed
         *  - LIFModel::clearSpikeCounts() --- Modified
         *  - LIFSingleThreadedModel::fire() --- Modified
         *  - GpuSim_struct.cu::clearSpikeCounts() --- Modified
         *  - LIFGPUModel::copyDeviceSpikeCountsToHost() --- Accessed
         *  - GpuSim_struct.cu::advanceNeuronsDevice() --- Modified
         *  - Hdf5Recorder::compileHistories() --- Accessed
         *  - XmlRecorder::compileHistories() --- Accessed
         */
        int *spikeCount;

        /*! Step count for each spike fired by each neuron
         *
         *  Usage: LOCAL VARIABLE
         *  - LIFModel::createAllNeurons() --- Initialized
         *  - LIFSingleThreadedModel::fire() --- Modified
         *  - GpuSim_struct.cu::advanceNeuronsDevice() --- Modified
         *  - LIFGPUModel::copyDeviceSpikeHistoryToHost() --- Accessed
         *  - Hdf5Recorder::compileHistories() --- Accessed
         *  - XmlRecorder::compileHistories() --- Accessed
         */
        uint64_t **spike_history;

        /*! The neuron type map (INH, EXC).
         *  
         *  Usage: LOCAL CONSTANT
         *  - LIFModel::generateNeuronTypeMap --- Initialized
         *  - LIFModel::logSimStep() --- Accessed
         *  - LIFSingleThreadedModel::synType() --- Accessed
         *  - GpuSim_struct.cu::synType() --- Accessed
         *  - Hdf5Recorder::saveSimState() --- Accessed
         */
        neuronType *neuron_type_map;

        /*! List of summation points for each neuron
         *  
         *  Usage: LOCAL CONSTANT
         *  - AllNeurons::AllNeurons() --- Initialized
         *  - LIFModel::loadMemory() --- Accessed
         *  - LIFSingleThreadedModel::advanceNeuron() --- Accessed
         *  - LIFGPUModel::setupSim() --- Accessed
         *  - GpuSim_struct.cu::advanceNeuronsDevice() --- Accessed
         *  - GpuSim_struct.cu::setSynapseSummationPointDevice() --- Accessed
         *  - GpuSim_struct.cu::updateNetworkDevice() --- Accessed
         *  - Network::Network() --- Accessed
         */
        BGFLOAT *summation_map;

        /*! The starter existence map (T/F).
         *  
         *  Usage: LOCAL CONSTANT
         *  - LIFModel::initStarterMap() --- Initialized
         *  - LIFModel::createAllNeurons() --- Accessed
         *  - LIFModel::logSimStep() --- Accessed
         *  - LIFModel::getStarterNeuronMatrix() --- Accessed
         *  - Hdf5Recorder::saveSimState() --- Accessed
         *  - XmlRecorder::saveSimState() --- Accessed
         */
        bool *starter_map;

        AllNeurons();
        AllNeurons(const int size);
        ~AllNeurons();

    private:

        int size;
};

#endif
