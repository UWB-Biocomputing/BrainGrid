/**
 ** \brief A class that performs stimulus input (implementation Poisson).
 **
 ** \class GpuSInputPoisson GpuSInputPoisson.h "GpuSInputPoisson.h"
 **
 ** \latexonly  \subsubsection*{Implementation} \endlatexonly
 ** \htmlonly   <h3>Implementation</h3> \endhtmlonly
 **
 ** The GpuSInputPoisson SInputPoisson performs providing stimulus input to the network for each time step.
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

/**
 ** \file GpuSInputPoisson.h
 **
 ** \brief Header file for GpuSInputPoisson.
 **/

#pragma once

#ifndef _GPUSINPUTPOISSON_H_
#define _GPUSINPUTPOISSON_H_

#include "SInputPoisson.h"
#include "AllSynapsesDevice.h"
#include "LIFGPUModel.h"

class GpuSInputPoisson : public SInputPoisson
{
public:
    //! The constructor for GpuSInputPoisson.
    GpuSInputPoisson(SimulationInfo* psi, TiXmlElement* parms);
    ~GpuSInputPoisson();

    //! Initialize data.
    virtual void init(Model* model, AllNeurons &neurons, SimulationInfo* psi);

    //! Terminate process.
    virtual void term(Model* model, SimulationInfo* psi);

    //! Process input stimulus for each time step.
    virtual void inputStimulus(Model* model, SimulationInfo* psi, BGFLOAT* summationPoint);

private:
    //! Allocate GPU device memory and copy values
    void allocDeviceValues( Model* model, SimulationInfo* psi, int *nISIs );

    //! Dellocate GPU device memory
    void deleteDeviceValues( Model* model, SimulationInfo* psi );

    //! Synapse structures in device memory.
    AllSynapsesDevice* allSynapsesDevice;
 
    //! Pointer to synapse index map in device memory.
    LIFGPUModel::SynapseIndexMap* synapseIndexMapDevice;

    //! Pointer to device interval counter.
    int* nISIs_d;
};

#endif // _GPUSINPUTPOISSON_H_