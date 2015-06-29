/** 
 * @authors Aaron Oziel, Sean Blackbourn
 * 
 * @class AllDynamicSTDPSynapses AllDynamicSTDPSynapses.h "AllDynamicSTDPSynapses.h"
 * @brief A container of all synapse data
 *
 *  The container holds synapse parameters of all synapses. 
 *  Each kind of synapse parameter is stored in a 2D array. Each item in the first 
 *  dimention of the array corresponds with each neuron, and each item in the second
 *  dimension of the array corresponds with a synapse parameter of each synapse of the neuron. 
 *  Bacause each neuron owns different number of synapses, the number of synapses 
 *  for each neuron is stored in a 1D array, synapse_counts.
 *
 *  For CUDA implementation, we used another structure, AllDSSynapsesDevice, where synapse
 *  parameters are stored in 1D arrays instead of 2D arrays, so that device functions
 *  can access these data less latency. When copying a synapse parameter, P[i][j],
 *  from host to device, it is stored in P[i * max_synapses_per_neuron + j] in 
 *  AllDSSynapsesDevice structure.
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
 *  
 *  The AllDynamicSTDPSynapses inherited properties from the AllDSSynapses and the AllSTDPSynapses
 *  classes (multiple inheritance), and both the AllDSSynapses and the AllSTDPSynapses classes are
 *  the subclass of the AllSpikingSynapses class. Therefore, this is known as a diamond class
 *  inheritance, which causes the problem of ambibuous hierarchy compositon. To solve the
 *  problem, we can use the virtual inheritance. 
 *  However, the virtual inheritance will bring another problem. That is, we cannot static cast
 *  from a pointer to the AllSynapses class to a pointer to the AllDSSynapses or the AllSTDPSynapses 
 *  classes. Compiler requires dynamic casting because vtable mechanism is involed in solving the 
 *  casting. But the dynamic casting cannot be used for the pointers to device (CUDA) memories. 
 *  Considering these issues, I decided that making the AllDynamicSTDPSynapses class the subclass
 *  of the AllSTDPSynapses class and adding properties of the AllDSSynapses class to it (fumik).
 *   
 */
#pragma once

#include "AllSTDPSynapses.h"

class AllDynamicSTDPSynapses : public AllSTDPSynapses
{
    public:
        AllDynamicSTDPSynapses();
        AllDynamicSTDPSynapses(const int num_neurons, const int max_synapses);
        virtual ~AllDynamicSTDPSynapses();
 
        virtual void setupSynapses(SimulationInfo *sim_info);
        virtual void setupSynapses(const int num_neurons, const int max_synapses);
        virtual void cleanupSynapses();
        virtual void resetSynapse(const uint32_t iSyn, const BGFLOAT deltaT);

    protected:
        virtual void readSynapse(istream &input, const uint32_t iSyn);
        virtual void writeSynapse(ostream& output, const uint32_t iSyn) const;

    public:
#if defined(USE_GPU)
        virtual void allocSynapseDeviceStruct( void** allSynapsesDevice, const SimulationInfo *sim_info );
        virtual void allocSynapseDeviceStruct( void** allSynapsesDevice, int num_neurons, int maxSynapsesPerNeuron );
        virtual void deleteSynapseDeviceStruct( void* allSynapsesDevice );
        virtual void copySynapseHostToDevice( void* allSynapsesDevice, const SimulationInfo *sim_info );
        virtual void copySynapseHostToDevice( void* allSynapsesDevice, int num_neurons, int maxSynapsesPerNeuron );
        virtual void copySynapseDeviceToHost( void* allSynapsesDevice, const SimulationInfo *sim_info );
        virtual void copyDeviceSynapseCountsToHost(void* allSynapsesDevice, const SimulationInfo *sim_info);
        virtual void copyDeviceSynapseSumIdxToHost(void* allSynapsesDevice, const SimulationInfo *sim_info);
        // Update the state of all synapses for a time step
        virtual void getFpCreateSynapse(unsigned long long& fpCreateSynapse_h);
        virtual void getFpChangePSR(unsigned long long& fpChangePSR_h);

    protected:
        virtual void allocDeviceStruct( AllDynamicSTDPSynapses &allSynapses, int num_neurons, int maxSynapsesPerNeuron );
        virtual void deleteDeviceStruct( AllDynamicSTDPSynapses& allSynapses );
        virtual void copyHostToDevice( void* allSynapsesDevice, AllDynamicSTDPSynapses& allSynapses, int num_neurons, int maxSynapsesPerNeuron );
        virtual void copyDeviceToHost( AllDynamicSTDPSynapses& allSynapses, const SimulationInfo *sim_info );

    public:
#else
        virtual void createSynapse(const uint32_t iSyn, int source_index, int dest_index, BGFLOAT* sp, const BGFLOAT deltaT, synapseType type);

    protected:
        virtual void changePSR(const uint32_t iSyn, const BGFLOAT deltaT);

    private:

#endif
    public:

        /*! The time of the last spike.
         *  
         *  Usage: LOCAL VARIABLE
         *  - LIFModel::readSynapse() --- Modified
         *  - LIFModel::writeSynapse() --- Accessed
         *  - LIFModel::resetSynapse() --- Initialized
         *  - SingleThreadedSpikingModel::advanceSynapse() --- Accessed & Modified 
         *  - GpuSim_Struct::createSynapse() --- Initialized
         *  - GpuSim_Struct::advanceSynapseDevice() --- Accessed & Modified  
         */
        uint64_t *lastSpike;

        // dynamic synapse vars...........
        /*! The time varying state variable \f$r\f$ for depression.
         *  
         *  Usage: LOCAL VARIABLE
         *  - LIFModel::readSynapse() --- Modified
         *  - LIFModel::writeSynapse() --- Accessed
         *  - LIFModel::resetSynapse() --- Initialized
         *  - SingleThreadedSpikingModel::advanceSynapse() --- Modified 
         *  - GpuSim_Struct::createSynapse() --- Initialized
         *  - GpuSim_Struct::advanceSynapsesDevice() --- Modified
         */
        BGFLOAT *r;

        /*! The time varying state variable \f$u\f$ for facilitation.
         *  
         *  Usage: LOCAL VARIABL
         *  - LIFModel::readSynapse() --- Modified
         *  - LIFModel::writeSynapse() --- Accessed
         *  - LIFModel::resetSynapse() --- Initialized
         *  - SingleThreadedSpikingModel::advanceSynapse() --- Modified 
         *  - GpuSim_Struct::createSynapse() --- Initialized
         *  - GpuSim_Struct::advanceSynapsesDevice() --- Modified
         */
        BGFLOAT *u;

        /*! The time constant of the depression of the dynamic synapse [range=(0,10); units=sec].
         *  
         *  Usage: LOCAL CONSTANT depending on synapse type
         *  - LIFModel::readSynapse() --- Modified
         *  - LIFModel::writeSynapse() --- Accessed
         *  - SingleThreadedSpikingModel::createSynapse() --- Initialized
         *  - SingleThreadedSpikingModel::advanceSynapse() --- Accessed
         *  - GpuSim_Struct::createSynapse() --- Initialized
         *  - GpuSim_Struct::advanceSynapsesDevice() --- Modified
         */
        BGFLOAT *D;

        /*! The use parameter of the dynamic synapse [range=(1e-5,1)].
         *  
         *  Usage: LOCAL CONSTANT depending on synapse type
         *  - LIFModel::readSynapse() --- Modified
         *  - LIFModel::writeSynapse() --- Accessed
         *  - SingleThreadedSpikingModel::createSynapse() --- Initialized
         *  - SingleThreadedSpikingModel::advanceSynapse() --- Accessed
         *  - GpuSim_Struct::createSynapse() --- Initialized
         *  - GpuSim_Struct::advanceSynapsesDevice() --- Modified
         */
        BGFLOAT *U;

        /*! The time constant of the facilitation of the dynamic synapse [range=(0,10); units=sec].
         *  
         *  Usage: LOCAL CONSTANT depending on synapse type
         *  - LIFModel::readSynapse() --- Modified
         *  - LIFModel::writeSynapse() --- Accessed
         *  - SingleThreadedSpikingModel::createSynapse() --- Initialized
         *  - SingleThreadedSpikingModel::advanceSynapse() --- Accessed
         *  - GpuSim_Struct::createSynapse() --- Initialized
         *  - GpuSim_Struct::advanceSynapsesDevice() --- Modified
         */
        BGFLOAT *F;
};

#if defined(__CUDACC__)
extern __global__ void getFpCreateSynapseDevice(void (**fpCreateSynapse_d)(AllDynamicSTDPSynapses*, const int, const int, int, int, BGFLOAT*, const BGFLOAT, synapseType));
        
extern __global__ void getFpChangePSRDevice(void (**fpChangePSR_d)(AllDynamicSTDPSynapses*, const uint32_t, const uint64_t, const BGFLOAT));

extern __device__ void createSynapse(AllDynamicSTDPSynapses* allSynapsesDevice, const int neuron_index, const int synapse_index, int source_index, int dest_index, BGFLOAT *sum_point, const BGFLOAT deltaT, synapseType type);

extern __device__ void changePSR(AllDynamicSTDPSynapses* allSynapsesDevice, const uint32_t, const uint64_t, const BGFLOAT deltaT);
#endif
