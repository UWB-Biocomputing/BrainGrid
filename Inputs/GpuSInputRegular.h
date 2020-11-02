/**
 *      @file GpuSInputRegular.h
 *
 *      @brief A class that performs stimulus input (implementation Regular on GPU).
 */

/**
 **
 ** \class GpuSInputRegular GpuSInputRegular.h "GpuSInputRegular.h"
 **
 ** \latexonly  \subsubsection*{Implementation} \endlatexonly
 ** \htmlonly   <h3>Implementation</h3> \endhtmlonly
 **
 ** The GpuSInputRegular performs providing stimulus input to the network for each time step on GPU.
 ** Inputs are series of current pulses, which are characterized by a duration, an interval
 ** and input values.
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

#ifndef _GPUSINPUTREGULAR_H_
#define _GPUSINPUTREGULAR_H_

#include "SInputRegular.h"

class GpuSInputRegular : public SInputRegular
{
public:
    //! The constructor for SInputRegular.
    GpuSInputRegular(SimulationInfo* psi, BGFLOAT duration, BGFLOAT interval, string &sync, BGFLOAT weight, vector<BGFLOAT> &maskIndex);
    virtual ~GpuSInputRegular();

    //! Initialize data.
    virtual void init(SimulationInfo* psi, vector<ClusterInfo *> &vtClrInfo);

    //! Terminate process.
    virtual void term(SimulationInfo* psi, vector<ClusterInfo *> const&vtClrInfo);

    //! Process input stimulus for each time step.
    virtual void inputStimulus(const SimulationInfo* psi, ClusterInfo *pci, int iStepOffset);

    // Process input stimulus for each time step.
    virtual void advanceSInputState(const ClusterInfo *pci, int iStep);

private:
    //! Allocate GPU device memory and copy values
    void allocDeviceValues( SimulationInfo* psi, ClusterInfo* pci, int *nShiftValues );

    //! Dellocate GPU device memory
    void deleteDeviceValues( ClusterInfo *pci );
};

//! Device function that processes input stimulus for each time step.
#if defined(__CUDACC__)
__global__ void inputStimulusDevice( int n, bool* masks_d, int* nShiftValues_d, int nStepsInCycle, int nStepsCycle, int nStepsDuration, AllDSSynapsesProps* allSynapsesProps, CLUSTER_INDEX_TYPE clusterID, int iStepOffset );
extern __global__ void applyI2SummationMap( int n, BGFLOAT* summationPoint_d, AllDSSynapsesProps* allSynapsesDevice );
#endif

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
#endif // _GPUSINPUTREGULAR_H_
