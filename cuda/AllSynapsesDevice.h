/**
 * This is the device side representation of AllSynapses structure, 
 * which was converting the AllSynapses structure from being allocated 
 * as a ragged array (1D array of pointers to 1D arrays) 
 * to single blocks of memory (as regular 2D arrays are allocated). 
 * This eliminated a memory access for every synapse state variable access.
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
         * LIFGPUModel::copySynapseHostToDevice		Copied from host
         * LIFGPUModel::copySynapseDeviceToHost		Copied to host
         * LIFGPUModel::copyDeviceSynapseSumCoordToHost	Accessed
         * updateNetworkDevice				Accessed
         * createSynapse				Initialized
         */
        Coordinate *summationCoord;

        /*! The weight (scaling factor, strength, maximal amplitude) of the synapse.
         *
         * LIFGPUModel::copySynapseHostToDevice		Copied from host
         * LIFGPUModel::copySynapseDeviceToHost		Copied to host
         * advanceSynapsesDevice			Accessed
         * updateNetworkDevice				Mofified
         * addSynapse					Initailized
         * createSynapse				Initialized
         */
        BGFLOAT *W;

        /*! This synapse's summation point's address.
         *
         * LIFGPUModel::copySynapseHostToDevice		Initialized
         * calcSummationMap				Accessed
         * eraseSynapse					Modified to NULL
         * createSynapse				Initialized
	 */
        BGFLOAT **summationPoint;

        /*! The location of the synapse.
         *
         * LIFGPUModel::copySynapseHostToDevice		Copied from host
         * LIFGPUModel::copySynapseDeviceToHost		Copied to host
         * createSynapse				Initialized
         */
        Coordinate *synapseCoord;

        /*! The post-synaptic response is the result of whatever computation 
         *  is going on in the synapse.
         *
         * LIFGPUModel::copySynapseHostToDevice		Copied from host
         * LIFGPUModel::copySynapseDeviceToHost		Copied to host
         * advanceSynapsesDevice			Modified
         * calcSummationMap				Accessed
         * createSynapse				Initialized
         */
        BGFLOAT *psr;
        
        /*! The decay for the psr.
         *
         * LIFGPUModel::copySynapseHostToDevice		Copied from host
         * LIFGPUModel::copySynapseDeviceToHost		Copied to host
         * advanceSynapsesDevice			Accessed
         * createSynapse				Initialized
         */
        BGFLOAT *decay;
        
        /*! The synaptic transmission delay, descretized into time steps.
         *
         * LIFGPUModel::copySynapseHostToDevice		Copied from host
         * LIFGPUModel::copySynapseDeviceToHost		Copied to host
         * advanceNeuronsDevice				Accessed
         * createSynapse				Initialized
         */
        int *total_delay;
        
        /*! The delayed queue
         *
         * LIFGPUModel::copySynapseHostToDevice		Copied from host
         * LIFGPUModel::copySynapseDeviceToHost		Copied to host
         * advanceNeuronsDevice				Modified
         * advanceSynapsesDevice			Modified
         * createSynapse				Initialized
         */
        uint32_t *delayQueue;
        
	/*! The index indicating the current time slot in the delayed queue
         *
         * LIFGPUModel::copySynapseHostToDevice		Copied from host
         * LIFGPUModel::copySynapseDeviceToHost		Copied to host
         * advanceNeuronsDevice				Accessed
         * advanceSynapsesDevice			Modified
         * createSynapse				Initialized
         */
        int *delayIdx;
        
	/*! Length of the delayed queue
         *
         * LIFGPUModel::copySynapseHostToDevice		Copied from host
         * LIFGPUModel::copySynapseDeviceToHost		Copied to host
         * advanceNeuronsDevice				Accessed
         * advanceSynapsesDevice			Accessed
         * createSynapse				Initialized
         */
        int *ldelayQueue;
        
    	/*! Synapse type
         *
         * LIFGPUModel::copySynapseHostToDevice		Copied from host
         * LIFGPUModel::copySynapseDeviceToHost		Copied to host
         * createSynapse				Initialized
         */
        synapseType *type;

        /*! The synaptic time constant \f$\tau\f$ [units=sec; range=(0,100)].
         *
         * LIFGPUModel::copySynapseHostToDevice		Copied from host
         * LIFGPUModel::copySynapseDeviceToHost		Copied to host
         * createSynapse				Initialized
         */
        BGFLOAT *tau;

        // dynamic synapse vars...........
        /*! The time varying state variable \f$r\f$ for depression.
         *
         * LIFGPUModel::copySynapseHostToDevice		Copied from host
         * LIFGPUModel::copySynapseDeviceToHost		Copied to host
         * advanceSynapsesDevice			Modified
         * createSynapse				Initialized
         */
        BGFLOAT *r;
        
        /*! The time varying state variable \f$u\f$ for facilitation.
         *
         * LIFGPUModel::copySynapseHostToDevice		Copied from host
         * LIFGPUModel::copySynapseDeviceToHost		Copied to host
         * advanceSynapsesDevice			Modified
         * createSynapse				Initialized
         */
        BGFLOAT *u;
        
        /*! The time constant of the depression of the dynamic synapse [range=(0,10); units=sec].
         *
         * LIFGPUModel::copySynapseHostToDevice		Copied from host
         * LIFGPUModel::copySynapseDeviceToHost		Copied to host
         * advanceSynapsesDevice			Accessed
         * createSynapse				Initialized
         */
        BGFLOAT *D;
        
        /*! The use parameter of the dynamic synapse [range=(1e-5,1)].
         *
         * LIFGPUModel::copySynapseHostToDevice		Copied from host
         * LIFGPUModel::copySynapseDeviceToHost		Copied to host
         * advanceSynapsesDevice			Accessed
         * createSynapse				Initialized
         */
        BGFLOAT *U;
        
        /*! The time constant of the facilitation of the dynamic synapse [range=(0,10); units=sec].
         *
         * LIFGPUModel::copySynapseHostToDevice		Copied from host
         * LIFGPUModel::copySynapseDeviceToHost		Copied to host
         * advanceSynapsesDevice			Accessed
         * createSynapse				Initialized
         */
        BGFLOAT *F;
        
        /*! The time of the last spike.
         *
         * LIFGPUModel::copySynapseHostToDevice		Copied from host
         * LIFGPUModel::copySynapseDeviceToHost		Copied to host
         * advanceSynapsesDevice			Modified
         * createSynapse				Initialized
         */
        uint64_t *lastSpike;

    	/*! The boolean value indicating the entry in the array is in use.
         *
         * LIFGPUModel::copySynapseHostToDevice		Copied from host
         * LIFGPUModel::copySynapseDeviceToHost		Copied to host
         * LIFGPUModel::copyDeviceSynapseSumCoordToHost	Accessed
         * advanceNeuronsDevice				Accessed
         * updateNetworkDevice				Accessed
         * eraseSynapse					Modified 
	 * addSynapse					Accessed
         * createSynapse				Modified
         */
        bool *in_use;

        /*! The number of active synapses for each neuron.
         *
         * LIFGPUModel::copySynapseHostToDevice		Copied from host
         * LIFGPUModel::copySynapseDeviceToHost		Copied to host
         * LIFGPUModel::copyDeviceSynapseCountsToHost	Accessed
         * advanceNeuronsDevice				Accessed
         * updateNetworkDevice				Accessed
         * eraseSynapse					Modified
         * addSynapse					Modified
         */
        size_t *synapse_counts;

        /*! The total number of active synapses.
         *
         * LIFGPUModel::copySynapseHostToDevice		Copied from host
         * LIFGPUModel::copySynapseDeviceToHost		Copied to host
         */
        size_t total_synapse_counts;

    	/*! The maximum number of synapses for each neurons.
         *
         * LIFGPUModel::copySynapseHostToDevice		Copied from host
         * LIFGPUModel::copySynapseDeviceToHost		Copied to host
         * addSynapse					Accessed
         * createSynapse				Accessed
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
