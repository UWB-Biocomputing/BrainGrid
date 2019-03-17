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
 ** Some models in this simulator is a rewrite of CSIM (2006) and other 
 ** work (Stiber and Kawasaki (2007?))
 **
 **
 **     @author Fumitaka Kawasaki
 **/

#pragma once

#ifndef _GPUSINPUTPOISSON_H_
#define _GPUSINPUTPOISSON_H_

#include "SInputPoisson.h"
#include "curand_kernel.h"

class GpuSInputPoisson : public SInputPoisson
{
public:
    //! The constructor for GpuSInputPoisson.
    GpuSInputPoisson(SimulationInfo* psi, TiXmlElement* parms);
    ~GpuSInputPoisson();

    //! Initialize data.
    virtual void init(SimulationInfo* psi, vector<ClusterInfo *> &vtClrInfo);

    //! Terminate process.
    virtual void term(SimulationInfo* psi, vector<ClusterInfo *> &vtClrInfo);

    //! Process input stimulus for each time step.
    virtual void inputStimulus(const SimulationInfo* psi, ClusterInfo *pci, int iStepOffset);

    //! Advance input stimulus state.
    virtual void advanceSInputState(const ClusterInfo *pci, int iStep);
private:
    //! Allocate GPU device memory and copy values
    void allocDeviceValues( SimulationInfo* psi, vector<ClusterInfo *> &vtClrInfo, int *nISIs );

    //! Dellocate GPU device memory
    void deleteDeviceValues( vector<ClusterInfo *> &vtClrInfo );
};

#if defined(__CUDACC__)
//! Device function that processes input stimulus for each time step.
extern __global__ void inputStimulusDevice( int n, int* nISIs_d, bool* masks_d, BGFLOAT deltaT, BGFLOAT lambda, curandState* devStates_d, AllDSSynapsesProps* allSynapsesProps, CLUSTER_INDEX_TYPE clusterID, int iStepOffset );
extern __global__ void applyI2SummationMap( int n, BGFLOAT* summationPoint_d, AllDSSynapsesProps* allSynapsesDevice );
extern __global__ void setupSeeds( int n, curandState* devStates_d, unsigned long seed );

/**
 * Adds a synapse to the network.  Requires the locations of the source and
 * destination neurons.
 *
 * @param synapsesDevice         Pointer to the Synapses object in device memory.
 * @param allSynapsesProps       Pointer to the Synapse structures in device memory.
 * @param pSummationMap          Pointer to the summation point.
 * @param width                  Width of neuron map (assumes square).
 * @param deltaT                 The simulation time step size.
 * @param weight                 Synapse weight.
 */
extern __global__ void initSynapsesDevice( IAllSynapses* synapsesDevice, int n, AllDSSynapsesProps* allSynapsesProps, BGFLOAT *pSummationMap, int width, const BGFLOAT deltaT, BGFLOAT weight );

#endif

#endif // _GPUSINPUTPOISSON_H_
