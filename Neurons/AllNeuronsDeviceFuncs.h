#pragma once

#include "AllIZHNeurons.h"
#include "AllSTDPSynapses.h"
#include "AllSynapsesDeviceFuncs.h"

#if defined(__CUDACC__)

/**
 *  CUDA code for advancing LIF neurons
 *
 *  @param[in] totalNeurons          Number of neurons.
 *  @param[in] maxSpikes             Maximum number of spikes per neuron per epoch.
 *  @param[in] deltaT                Inner simulation step duration.
 *  @param[in] simulationStep        The current simulation step.
 *  @param[in] randNoise             Pointer to device random noise array.
 *  @param[in] allNeuronsDevice      Pointer to Neuron structures in device memory.
 *  @param[in] allSynapsesDevice     Pointer to Synapse structures in device memory.
 *  @param[in] synapseIndexMap       Inverse map, which is a table indexed by an input neuron and maps to the synapses that provide input to that neuron.
 *  @param[in] fAllowBackPropagation True if back propagaion is allowed.
 *  @param[in] iStepOffset           Offset from the current simulation step.
 */
extern __global__ void advanceLIFNeuronsDevice(const int totalNeurons, const int maxSpikes, const BGFLOAT deltaT,
                                               const uint64_t simulationStep, const float* randNoise,
                                               AllIFNeuronsDeviceProperties* allNeuronsDevice,
                                               AllSpikingSynapsesDeviceProperties* allSynapsesDevice,
                                               SynapseIndexMap* synapseIndexMapDevice, const bool fAllowBackPropagation,
                                               const int iStepOffset);

/**
 *  CUDA code for advancing izhikevich neurons
 *
 *  @param[in] totalNeurons          Number of neurons.
 *  @param[in] maxSpikes             Maximum number of spikes per neuron per epoch.
 *  @param[in] deltaT                Inner simulation step duration.
 *  @param[in] simulationStep        The current simulation step.
 *  @param[in] randNoise             Pointer to device random noise array.
 *  @param[in] allNeuronsDevice      Pointer to Neuron structures in device memory.
 *  @param[in] allSynapsesDevice     Pointer to Synapse structures in device memory.
 *  @param[in] synapseIndexMap       Inverse map, which is a table indexed by an input neuron and maps to the synapses that provide input to that neuron.
 *  @param[in] fAllowBackPropagation True if back propagaion is allowed.
 *  @param[in] iStepOffset           Offset from the current simulation step.
 */
extern __global__ void advanceIZHNeuronsDevice(const int totalNeurons, const int maxSpikes,
                                               const BGFLOAT deltaT, const uint64_t simulationStep, const float* randNoise,
                                               AllIZHNeuronsDeviceProperties* allNeuronsDevice,
                                               AllSpikingSynapsesDeviceProperties* allSynapsesDevice,
                                               SynapseIndexMap* synapseIndexMapDevice,
                                               const bool fAllowBackPropagation, const int iStepOffset);

#endif
