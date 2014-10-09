/**
 * @class AllSynapsesDevice AllSynapsesDevice.h "AllSynapsesDevice."
 * @brief The device side representation of AllSynapses structure
 *
 * This is the device side representation of AllSynapses structure, 
 * which was converting the AllSynapses structure from being allocated 
 * as a ragged array (1D array of pointers to 1D arrays) 
 * to single blocks of memory (as regular 2D arrays are allocated). 
 * This eliminated a memory access for every synapse state variable access.
 *
 *  In this file you will find usage statistics for every variable inthe BrainGrid
 *  project as we find them. These statistics can be used to help
 *  determine if a variable is being used, where it is being used, and how it
 *  is being used in each class::function()
 *
 *  For Example
 *
 *  Usage:
 *  - LOCAL VARIABLE -- a variable for individual synapse
 *  - LOCAL CONSTANT --  a constant for individual synapse
 *  - GLOBAL VARIABLE -- a variable for all synapses
 *  - GLOBAL CONSTANT -- a constant for all synapses
 *
 *  Class::function(): --- Initialized, Modified OR Accessed
 *
 *  OtherClass::function(): --- Accessed
 *
 *  Note: All GLOBAL parameters can be scalars. Also some LOCAL CONSTANT can be categorized
 *  depending on synapse types.
 */

#pragma once

#ifndef _ALLSYNAPSESDEVICE_H_
#define _ALLSYNAPSESDEVICE_H_

#include "Global.h"

struct AllSynapsesDevice
{
    public:
        AllSynapsesDevice();
        AllSynapsesDevice(const int num_neurons, const int max_synapses);
        ~AllSynapsesDevice();
 
        /*! The coordinates of the summation point.
         *
         *  Usage: LOCAL CONSTANT
         *  - LIFGPUModel::copySynapseHostToDevice --- Copied from host
         *  - LIFGPUModel::copySynapseDeviceToHost --- Copied to host
         *  - LIFGPUModel::copyDeviceSynapseSumCoordToHost() --- Accessed
         *  - GpuSim_Struct::createSynapseImap() --- Accessed
         *  - GpuSim_Struct::setSynapseSummationPointDevice() --- Accessed
         *  - GpuSim_Struct::updateNetworkDevice() --- Accessed
         *  - GpuSim_Struct::createSynapse() --- Initialized
         */
        Coordinate *summationCoord;

        /*! The weight (scaling factor, strength, maximal amplitude) of the synapse.
         *
         *  Usage: LOCAL VARIABLE
         *  - LIFGPUModel::copySynapseHostToDevice --- Copied from host
         *  - LIFGPUModel::copySynapseDeviceToHost --- Copied to host
         *  - GpuSim_Struct::advanceSynapsesDevice() --- Accessed
         *  - GpuSim_Struct::updateNetworkDevice() --- Modified
         *  - GpuSim_Struct::addSynapse() --- Modified
         *  - GpuSim_Struct::createSynapse() --- Initialized
         */
        BGFLOAT *W;

        /*! This synapse's summation point's address.
         *
         *  Usage: LOCAL CONSTANT
         *  - GpuSim_Struct::setSynapseSummationPointDevice() --- Initialized
         *  - GpuSim_Struct::calcSummationMap() --- Accessed
         *  - GpuSim_Struct::eraseSynapse() --- Modified (= NULL)
         *  - GpuSim_Struct::createSynapse() --- Initialized
	 */
        BGFLOAT **summationPoint;

        /*! The location of the synapse.
         *
         *  Usage: NOT USED ANYWHERE
         *  - LIFGPUModel::copySynapseHostToDevice --- Copied from host
         *  - LIFGPUModel::copySynapseDeviceToHost --- Copied to host
         *  - GpuSim_Struct::createSynapse() --- Initialized
         */
        Coordinate *synapseCoord;

        /*! The post-synaptic response is the result of whatever computation 
         *  is going on in the synapse.
         *
         *  Usage: LOCAL VARIABLE
         *  - LIFGPUModel::copySynapseHostToDevice --- Copied from host
         *  - LIFGPUModel::copySynapseDeviceToHost --- Copied to host
         *  - GpuSim_Struct::createSynapse() --- Initialized
         *  - GpuSim_Struct::advanceSynapsesDevice() --- Modified
         *  - GpuSim_Struct::calcSummationMap() --- Accessed
         */
        BGFLOAT *psr;
        
        /*! The decay for the psr.
         *
         *  Usage: LOCAL CONSTANT depending on synapse type
         *  - LIFGPUModel::copySynapseHostToDevice --- Copied from host
         *  - LIFGPUModel::copySynapseDeviceToHost --- Copied to host
         *  - GpuSim_Struct::createSynapse() --- Modified
         *  - GpuSim_Struct::advanceSynapsesDevice() --- Accessed
         */
        BGFLOAT *decay;
        
        /*! The synaptic transmission delay, descretized into time steps.
         *
         *  Usage: LOCAL CONSTANT depending on synapse type
         *  - LIFGPUModel::copySynapseHostToDevice --- Copied from host
         *  - LIFGPUModel::copySynapseDeviceToHost --- Copied to host
         *  - GpuSim_Struct::createSynapse() --- Initialized
         *  - GpuSim_Struct::advanceNeuronsDevice() --- Accessed
         */
        int *total_delay;
        
        /*! The delayed queue
         *
         *  Usage: LOCAL CONSTANT
         *  - LIFGPUModel::copySynapseHostToDevice --- Copied from host
         *  - LIFGPUModel::copySynapseDeviceToHost --- Copied to host
         *  - GpuSim_Struct::createSynapse() --- Initialized
         *  - GpuSim_Struct::advanceNeuronsDevice() --- Accessed
         *  - GpuSim_Struct::advanceSynapseDevice() --- Accessed
         */
        uint32_t *delayQueue;
        
        /*! The index indicating the current time slot in the delayed queue
         *
         *  Usage: LOCAL VARIABLE
         *  - LIFGPUModel::copySynapseHostToDevice --- Copied from host
         *  - LIFGPUModel::copySynapseDeviceToHost --- Copied to host
         *  - GpuSim_Struct::advanceNeuronsDevice() --- Accessed
         *  - GpuSim_Struct::advanceSynapseDevice() --- Accessed & Modified
         *  - GpuSim_Struct::createSynapse() --- Initialized
         */
        int *delayIdx;
        
        /*! Length of the delayed queue
         *
         *  Usage: GLOBAL CONSTANT
         *  - LIFGPUModel::copySynapseHostToDevice --- Copied from host
         *  - LIFGPUModel::copySynapseDeviceToHost --- Copied to host
         *  - GpuSim_Struct::advanceNeuronsDevice() --- Accessed
         *  - GpuSim_Struct::advanceSynapsesDevice() --- Accessed
         *  - GpuSim_Struct::createSynapse() --- Initialzied
         */
        int *ldelayQueue;
        
    	/*! Synapse type
         *
         *  Usage: LOCAL CONSTANT
         *  - LIFGPUModel::copySynapseHostToDevice --- Copied from host
         *  - LIFGPUModel::copySynapseDeviceToHost --- Copied to host
         *  - GpuSim_Struct::createSynapse() --- Initialized
         */
        synapseType *type;

        /*! The synaptic time constant \f$\tau\f$ [units=sec; range=(0,100)].
         *
         *  Usage: LOCAL CONSTANT depending on synapse type
         *  - LIFGPUModel::copySynapseHostToDevice --- Copied from host
         *  - LIFGPUModel::copySynapseDeviceToHost --- Copied to host
         *  - GpuSim_Struct::createSynapse() --- Initialized
         */
        BGFLOAT *tau;

        // dynamic synapse vars...........
        /*! The time varying state variable \f$r\f$ for depression.
         *
         *  Usage: LOCAL VARIABLE
         *  - LIFGPUModel::copySynapseHostToDevice --- Copied from host
         *  - LIFGPUModel::copySynapseDeviceToHost --- Copied to host
         *  - GpuSim_Struct::createSynapse() --- Initialized
         *  - GpuSim_Struct::advanceSynapsesDevice() --- Modified
         */
        BGFLOAT *r;
        
        /*! The time varying state variable \f$u\f$ for facilitation.
         *
         *  Usage: LOCAL VARIABL
         *  - LIFGPUModel::copySynapseHostToDevice --- Copied from host
         *  - LIFGPUModel::copySynapseDeviceToHost --- Copied to host
         *  - GpuSim_Struct::createSynapse() --- Initialized
         *  - GpuSim_Struct::advanceSynapsesDevice() --- Modified
         */
        BGFLOAT *u;
        
        /*! The time constant of the depression of the dynamic synapse [range=(0,10); units=sec].
         *
         *  Usage: LOCAL CONSTANT depending on synapse type
         *  - LIFGPUModel::copySynapseHostToDevice --- Copied from host
         *  - LIFGPUModel::copySynapseDeviceToHost --- Copied to host
         *  - GpuSim_Struct::createSynapse() --- Initialized
         *  - GpuSim_Struct::advanceSynapsesDevice() --- Modified
         */
        BGFLOAT *D;
        
        /*! The use parameter of the dynamic synapse [range=(1e-5,1)].
         *
         *  Usage: LOCAL CONSTANT depending on synapse type
         *  - LIFGPUModel::copySynapseHostToDevice --- Copied from host
         *  - LIFGPUModel::copySynapseDeviceToHost --- Copied to host
         *  - GpuSim_Struct::createSynapse() --- Initialized
         *  - GpuSim_Struct::advanceSynapsesDevice() --- Modified
         */
        BGFLOAT *U;
        
        /*! The time constant of the facilitation of the dynamic synapse [range=(0,10); units=sec].
         *
         *  Usage: LOCAL CONSTANT depending on synapse type
         *  - LIFGPUModel::copySynapseHostToDevice --- Copied from host
         *  - LIFGPUModel::copySynapseDeviceToHost --- Copied to host
         *  - GpuSim_Struct::createSynapse() --- Initialized
         *  - GpuSim_Struct::advanceSynapsesDevice() --- Modified
         */
        BGFLOAT *F;
        
        /*! The time of the last spike.
         *
         *  Usage: LOCAL VARIABLE
         *  - LIFGPUModel::copySynapseHostToDevice --- Copied from host
         *  - LIFGPUModel::copySynapseDeviceToHost --- Copied to host
         *  - GpuSim_Struct::createSynapse() --- Initialized
         *  - GpuSim_Struct::advanceSynapseDevice() --- Accessed & Modified
         */
        uint64_t *lastSpike;

    	/*! The boolean value indicating the entry in the array is in use.
         *
         *  Usage: LOCAL VARIABLE
         *  - LIFGPUModel::copySynapseHostToDevice --- Copied from host
         *  - LIFGPUModel::copySynapseDeviceToHost --- Copied to host
         *  - LIFGPUModel::copyDeviceSynapseSumCoordToHost() --- Accessed
         *  - LIFGPUModel::createSynapseImap() --- Accessed
         *  - GpuSim_Struct::advanceNeuronsDevice() --- Accessed
         *  - GpuSim_Struct::setSynapseSummationPointDevice() --- Accessed
         *  - GpuSim_Struct::updateNetworkDevice() --- Accessed
         *  - GpuSim_Struct::updateNetworkDevice() --- Modified
         *  - GpuSim_Struct::addSynapse() --- Accessed
         *  - GpuSim_Struct::createSynapse() --- Modified
         */
        bool *in_use;

        /*! The number of active synapses for each neuron.
         *
         *  Usage: LOCAL VARIABLE
         *  - LIFGPUModel::copySynapseHostToDevice --- Copied from host
         *  - LIFGPUModel::copySynapseDeviceToHost --- Copied to host
         *  - LIFGPUModel::copyDeviceSynapseCountsToHost --- Accessed
         *  - LIFGPUModel::copyDeviceSynapseCountsToHost() --- Accessed
         *  - LIFGPUModel::createSynapseImap() --- Accessed
         *  - GpuSim_Struct::advanceNeuronsDevice() --- Accessed
         *  - GpuSim_Struct::setSynapseSummationPointDevice() --- Accessed
         *  - GpuSim_Struct::updateNetworkDevice() --- Accessed
         *  - GpuSim_Struct::eraseSynapse() --- Modified
         *  - GpuSim_Struct::addSynapse() --- Modified
         */
        size_t *synapse_counts;

        /*! The total number of active synapses.
         *
         *  Usage: GLOBAL VARIABLE
         *  - LIFGPUModel::copySynapseHostToDevice --- Copied from host
         *  - LIFGPUModel::copySynapseDeviceToHost --- Copied to host
         *  - LIFGPUModel::advance() --- Accessed
         *  - LIFGPUModel::createSynapseImap() --- Modified
         */
        size_t total_synapse_counts;

    	/*! The maximum number of synapses for each neurons.
         *
         *  Usage: GLOBAL CONSTANT
         *  - LIFGPUModel::copySynapseHostToDevice --- Copied from host
         *  - LIFGPUModel::copySynapseDeviceToHost --- Copied to host
         *  - LIFGPUModel::createSynapseImap() --- Accessed
         *  - GpuSim_Struct::addSynapse --- Accessed
         *  - GpuSim_Struct::createSynapse --- Accessed
         */
        size_t max_synapses;
private:
        /*! The manimum total number of synapses.
         *
         * Usage: Used by constructor and destructor
         */
	uint32_t max_total_synapses;
};

#endif
