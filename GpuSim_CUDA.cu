// GpuSim_CUDA.cu
// by Sean McCallum, 2010

#include "global.h"
#include "DynamicSpikingSynapse_struct.h"
#include "LifNeuron_struct.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <fstream>
using namespace std;


//Forward Declarations
extern "C" void advanceGPU(			LifNeuron_struct* neuronArrays, 
									int neuron_count,
									DynamicSpikingSynapse_struct* synapseArray,
									int synapse_count,
									int endTime,
									int currentStep,
									int maxSteps,
									int width);

// Make pointers to device neuron arrays
FLOAT* neur_C1_d;					
FLOAT* neur_C2_d;
FLOAT* neur_I0_d;
FLOAT* neur_Inoise_d;
FLOAT* neur_Trefract_d;
FLOAT* neur_Vm_d;
FLOAT* neur_Vreset_d;
FLOAT* neur_Vthresh_d;
int* neur_nStepsInRefr_d;
int* neur_spikeCount_d;
FLOAT* neur_summationPoint_d;
FLOAT* neur_randNoise_d;

// Determine sizes for GPU device arrays
size_t neur_FLOATS_size;
size_t neur_ints_size;

__device__ void LifNeuronFireDevice(FLOAT* neur_Trefract_d,
									FLOAT* neur_Vm_d,
									FLOAT* neur_Vreset_d,
									int* neur_spikeCount_d,
									int* neur_nStepsInRefr_d,
									int idx,
									double deltaT,
									double simulationTime);

__global__ void advanceNeuronsDevice(FLOAT* neur_C1_d,
									 FLOAT* neur_C2_d,
									 FLOAT* neur_I0_d,
									 FLOAT* neur_Inoise_d,
									 FLOAT* neur_Trefract_d,
									 FLOAT* neur_Vm_d,
									 FLOAT* neur_Vreset_d,
									 FLOAT* neur_Vthresh_d,
									 int* neur_nStepsInRefr_d,
									 int* neur_spikeCount_d,
									 FLOAT* neur_summationPoint_d,
									 FLOAT* neur_randNoise_d,
									 int n,
									 double deltaT,
									 double simulationTime);

void advanceGPU(LifNeuron_struct* neuronArrays, 
				 int neuron_count,
				 DynamicSpikingSynapse_struct* synapseArray,
				 int synapse_count,
				 int endTime,
				 int currentStep,
				 int maxSteps,
				 int width)
{
	if (g_simulationTime == 0.0){

		// Get sizes of arrays
		neur_FLOATS_size = neuron_count * sizeof (FLOAT);
		neur_ints_size = neuron_count * sizeof (int);

		// Allocate on GPU device
		cudaMalloc ( ( void ** ) &neur_C1_d, neur_FLOATS_size );
		cudaMalloc ( ( void ** ) &neur_C2_d, neur_FLOATS_size );
		cudaMalloc ( ( void ** ) &neur_I0_d, neur_FLOATS_size );
		cudaMalloc ( ( void ** ) &neur_Inoise_d, neur_FLOATS_size );
		cudaMalloc ( ( void ** ) &neur_Trefract_d, neur_FLOATS_size );
		cudaMalloc ( ( void ** ) &neur_Vm_d, neur_FLOATS_size );
		cudaMalloc ( ( void ** ) &neur_Vreset_d, neur_FLOATS_size );
		cudaMalloc ( ( void ** ) &neur_Vthresh_d, neur_FLOATS_size );
		cudaMalloc ( ( void ** ) &neur_nStepsInRefr_d, neur_ints_size );		
		cudaMalloc ( ( void ** ) &neur_summationPoint_d, neur_FLOATS_size );
		cudaMalloc ( ( void ** ) &neur_randNoise_d, neur_FLOATS_size );
		cudaMalloc ( ( void ** ) &neur_spikeCount_d, neur_ints_size );

		// Copy host neuron and synapse arrays into GPU device
		cudaMemcpy ( neur_C1_d, neuronArrays->C1, neur_FLOATS_size, cudaMemcpyHostToDevice );
		cudaMemcpy ( neur_C2_d, neuronArrays->C2, neur_FLOATS_size, cudaMemcpyHostToDevice );
		cudaMemcpy ( neur_I0_d, neuronArrays->I0, neur_FLOATS_size, cudaMemcpyHostToDevice );
		cudaMemcpy ( neur_Inoise_d, neuronArrays->Inoise, neur_FLOATS_size, cudaMemcpyHostToDevice );
		cudaMemcpy ( neur_Trefract_d, neuronArrays->Trefract, neur_FLOATS_size, cudaMemcpyHostToDevice );
		cudaMemcpy ( neur_Vm_d, neuronArrays->Vm, neur_FLOATS_size, cudaMemcpyHostToDevice );
		cudaMemcpy ( neur_Vreset_d, neuronArrays->Vreset, neur_FLOATS_size, cudaMemcpyHostToDevice );
		cudaMemcpy ( neur_Vthresh_d, neuronArrays->Vthresh, neur_FLOATS_size, cudaMemcpyHostToDevice );
		cudaMemcpy ( neur_nStepsInRefr_d, neuronArrays->nStepsInRefr, neur_ints_size, cudaMemcpyHostToDevice );
		cudaMemcpy ( neur_summationPoint_d, neuronArrays->summationPoint, neur_FLOATS_size, cudaMemcpyHostToDevice );
		cudaMemcpy ( neur_spikeCount_d, neuronArrays->spikeCount, neur_ints_size, cudaMemcpyHostToDevice );
	}	
	
	// Configure dimensions for CUDA scheduling per sim size
	int blocksx, blocksy, threadsx, threadsy;
	if (neuronArrays->numNeurons == 100){
		blocksx = 2;
		blocksy = 1;
		threadsx = 64;
		threadsy = 1;
	}
	if (neuronArrays->numNeurons == 625){
		blocksx = 5;
		blocksy = 5;
		threadsx = 8;
		threadsy = 4;
	}
	if (neuronArrays->numNeurons == 10000){
		blocksx = 8;
		blocksy = 5;
		threadsx = 16;
		threadsy = 16;
	}
	dim3 dimGrid(blocksx, blocksy);
	dim3 dimBlock(threadsx, threadsy);

	DEBUG(cout << "Looping kernels in advanceGPU() function" << endl;);
	while (g_simulationTime < endTime)
    {
		// Generate random noise
		for (int i = 0; i < neuron_count; i++){
			neuronArrays->randNoise[i] = normrnd();
		}		
		// Copy random noise to GPU device
		cudaMemcpy ( neur_randNoise_d, neuronArrays->randNoise, neur_FLOATS_size, cudaMemcpyHostToDevice );

        // Advance neurons
		advanceNeuronsDevice <<< dimGrid, dimBlock >>> (	
														neur_C1_d, 
														neur_C2_d,
														neur_I0_d,
														neur_Inoise_d,
														neur_Trefract_d,
														neur_Vm_d,
														neur_Vreset_d,
														neur_Vthresh_d,
														neur_nStepsInRefr_d,
														neur_spikeCount_d,
														neur_summationPoint_d,
														neur_randNoise_d,
														neuron_count,
														neuronArrays->deltaT,
														g_simulationTime );

		// Copy processed data from GPU device memory to host
		cudaMemcpy ( neuronArrays->spikeCount, neur_spikeCount_d, neur_ints_size, cudaMemcpyDeviceToHost );

		// Go through spike counts and add a g_simulationTime to the psi->pNeuronList->at(i).spikeHistory
		for (int i = 0; i < neuron_count; i++){
			if ( neuronArrays->spikeCount[i] > 0) {
				vector<FLOAT>& spikeHist = *(neuronArrays->spikeHistories[i]);
				spikeHist.push_back(g_simulationTime);
				neuronArrays->spikeCount[i] = 0;   // reset for next advance
			}
		}
		// Copy zeroed array back to GPU
		cudaMemcpy ( neur_spikeCount_d, neuronArrays->spikeCount, neur_ints_size, cudaMemcpyHostToDevice );

		// Advance the clock
		g_simulationTime += neuronArrays->deltaT;
    }
	
	// Determine whether to free device mem.
	if (currentStep == maxSteps && g_simulationTime >= endTime){
		cudaFree( neur_C1_d );
		cudaFree( neur_C2_d );
		cudaFree( neur_I0_d );
		cudaFree( neur_Inoise_d );
		cudaFree( neur_Vm_d );
		cudaFree( neur_Trefract_d );
		cudaFree( neur_Vreset_d );
		cudaFree( neur_Vthresh_d );
		cudaFree( neur_nStepsInRefr_d );		
		cudaFree( neur_summationPoint_d );
	}
}


// CUDA code for firing a neuron ------------------------------------------------------------------------
__device__ void LifNeuronFireDevice (	FLOAT* neur_Trefract_d,
										FLOAT* neur_Vm_d,
										FLOAT* neur_Vreset_d,
										int* neur_spikeCount_d,
										int* neur_nStepsInRefr_d,
										int idx, 
										double deltaT,
										double simulationTime ) {

	// Note the occurrence of a spike
	neur_spikeCount_d[idx]++;

	// calculate the number of steps in the absolute refractory period
	neur_nStepsInRefr_d[idx] = static_cast<int> ( neur_Trefract_d[idx] / deltaT + 0.5 );

	// reset to 'Vreset'
	neur_Vm_d[idx] = neur_Vreset_d[idx];
}

// CUDA code for advancing neurons-----------------------------------------------------------------------
__global__ void advanceNeuronsDevice(FLOAT* neur_C1_d, 
									 FLOAT* neur_C2_d,
									 FLOAT* neur_I0_d,
									 FLOAT* neur_Inoise_d,
									 FLOAT* neur_Trefract_d,
									 FLOAT* neur_Vm_d,
									 FLOAT* neur_Vreset_d,
									 FLOAT* neur_Vthresh_d,
									 int* neur_nStepsInRefr_d,
									 int* neur_spikeCount_d,
									 FLOAT* neur_summationPoint_d,
									 FLOAT* neur_randNoise_d,
									 int n, 
									 double deltaT,
									 double simulationTime) {

    // determine which neuron this thread is processing
	//int idx = threadIdx.x;
    int idx = ((blockIdx.x + gridDim.x * blockIdx.y) * (blockDim.x * blockDim.y)) + 
				(threadIdx.x + threadIdx.y * blockDim.x);

	// Reset fired status
    //neur_hasFired_d[idx] = false;

    if ( neur_nStepsInRefr_d[idx] > 0 ) { // is neuron refractory?
        --neur_nStepsInRefr_d[idx];
	} else if ( neur_Vm_d[idx] >= neur_Vthresh_d[idx]) { // should it fire?
		LifNeuronFireDevice (	neur_Trefract_d,
								neur_Vm_d,
								neur_Vreset_d,
								neur_spikeCount_d,
								neur_nStepsInRefr_d,
								idx, 
								deltaT,
								simulationTime );
    } else {
		neur_summationPoint_d[idx] += neur_I0_d[idx]; // add IO		
		neur_summationPoint_d[idx] += (neur_randNoise_d[idx] * neur_Inoise_d[idx]); // add cheap noise
        neur_Vm_d[idx] = neur_C1_d[idx] * neur_Vm_d[idx] + neur_C2_d[idx] * neur_summationPoint_d[idx]; // decay Vm and add inputs
    }

    // clear synaptic input for next time step
	neur_summationPoint_d[idx] = 0;
}
