/**
 *      @file GpuSInputPoisson.h
 *
 *      @brief A class that performs stimulus input (implementation Poisson on GPU).
 */

/**
 **
 ** \class GpuSInputPoisson GpuSInputPoisson.h "GpuSInputPoisson.h"
 **
 ** \latexonly  \subsubsection*{Implementation} \endlatexonly
 ** \htmlonly   <h3>Implementation</h3> \endhtmlonly
 **
 ** The GpuSInputPoisson performs providing stimulus input to the network for each time step.
 ** In this version, a layer of synapses are added, which accept external spike trains.
 ** Each synapse gets an indivisual spike train (Poisson distribution) characterized
 ** by mean firing rate, and each synapse has individual weight value.
 **
 ** \latexonly  \subsubsection*{Credits} \endlatexonly
 ** \htmlonly   <h3>Credits</h3> \endhtmlonly
 **
 ** This simulator is a rewrite of CSIM (2006) and other work (Stiber and Kawasaki (2007?))
 **
 **
 **     @author Fumitaka Kawasaki
 **/

#pragma once

#ifndef _GPUSINPUTPOISSON_H_
#define _GPUSINPUTPOISSON_H_

#include "SInputPoisson.h"
#include "GPUSpikingModel.h"

class GpuSInputPoisson : public SInputPoisson
{
public:
    //! The constructor for GpuSInputPoisson.
    GpuSInputPoisson(SimulationInfo* psi, TiXmlElement* parms);
    ~GpuSInputPoisson();

    //! Initialize data.
    virtual void init(IModel* model, AllNeurons &neurons, SimulationInfo* psi);

    //! Terminate process.
    virtual void term(IModel* model, SimulationInfo* psi);

    //! Process input stimulus for each time step.
    virtual void inputStimulus(IModel* model, SimulationInfo* psi, BGFLOAT* summationPoint);

private:
    //! Allocate GPU device memory and copy values
    void allocDeviceValues( IModel* model, SimulationInfo* psi, int *nISIs );

    //! Dellocate GPU device memory
    void deleteDeviceValues( IModel* model, SimulationInfo* psi );

    //! Synapse structures in device memory.
    AllDSSynapses* allSynapsesDevice;
 
    //! Pointer to synapse index map in device memory.
    SynapseIndexMap* synapseIndexMapDevice;

    //! Pointer to device interval counter.
    int* nISIs_d;

    //! Pointer to device masks for stimulus input
    bool* masks_d;
};

#if defined(__CUDACC__)
//! Device function that processes input stimulus for each time step.
extern __global__ void initSynapsesDevice( int n, AllDSSynapses* allSynapsesDevice, BGFLOAT *pSummationMap, int width, const BGFLOAT deltaT, BGFLOAT weight );
extern __global__ void inputStimulusDevice( int n, int* nISIs_d, bool* masks_d, BGFLOAT deltaT, BGFLOAT lambda, curandState* devStates_d, AllDSSynapses* allSynapsesDevice );
extern __global__ void applyI2SummationMap( int n, BGFLOAT* summationPoint_d, AllDSSynapses* allSynapsesDevice );
extern __global__ void setupSeeds( int n, curandState* devStates_d, unsigned long seed );
#endif

#endif // _GPUSINPUTPOISSON_H_
