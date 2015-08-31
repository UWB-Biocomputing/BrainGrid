#pragma once

using namespace std;

#include "Global.h"
#include "SimulationInfo.h"
#include "SynapseIndexMap.h"
#include "Layout.h"

class AllSynapses;

class AllNeurons
{
    public:
        /*! List of summation points for each neuron
         *  
         *  Usage: LOCAL CONSTANT
         *  - AllIFNeurons::AllIFNeurons() --- Initialized
         *  - LIFModel::loadMemory() --- Accessed
         *  - SingleThreadedSpikingModel::advanceNeuron() --- Accessed
         *  - GPUSpikingModel::setupSim() --- Accessed
         *  - GpuSim_struct.cu::advanceNeuronsDevice() --- Accessed
         *  - GpuSim_struct.cu::setSynapseSummationPointDevice() --- Accessed
         *  - GpuSim_struct.cu::updateNetworkDevice() --- Accessed
         *  - Network::Network() --- Accessed
         */
        BGFLOAT *summation_map;

        AllNeurons();
        virtual ~AllNeurons();

        virtual void setupNeurons(SimulationInfo *sim_info);
        virtual void cleanupNeurons();
        virtual int numParameters() = 0;
        virtual int readParameters(const TiXmlElement& element) = 0;
        virtual void printParameters(ostream &output) const = 0;
        virtual void createAllNeurons(SimulationInfo *sim_info, Layout *layout) = 0;
        virtual string toString(const int i) const = 0;
        virtual void readNeurons(istream &input, const SimulationInfo *sim_info) = 0;
        virtual void writeNeurons(ostream& output, const SimulationInfo *sim_info) const = 0;

#if defined(USE_GPU)
        virtual void allocNeuronDeviceStruct( void** allNeuronsDevice, SimulationInfo *sim_info ) = 0;
        virtual void deleteNeuronDeviceStruct( void* allNeuronsDevice, const SimulationInfo *sim_info ) = 0;
        virtual void copyNeuronHostToDevice( void* allNeuronsDevice, const SimulationInfo *sim_info ) = 0;
        virtual void copyNeuronDeviceToHost( void* allNeuronsDevice, const SimulationInfo *sim_info ) = 0;
        virtual void copyNeuronDeviceSpikeHistoryToHost( void* allNeuronsDevice, const SimulationInfo *sim_info ) = 0;
        virtual void copyNeuronDeviceSpikeCountsToHost( void* allNeuronsDevice, const SimulationInfo *sim_info ) = 0;
        virtual void clearNeuronSpikeCounts( void* allNeuronsDevice, const SimulationInfo *sim_info ) = 0;
        // Update the state of all neurons for a time step
        virtual void advanceNeurons(AllSynapses &synapses, AllNeurons* allNeuronsDevice, AllSynapses* allSynapsesDevice, const SimulationInfo *sim_info, float* randNoise, SynapseIndexMap* synapseIndexMapDevice) = 0;
#else
        // Update the state of all neurons for a time step
        virtual void advanceNeurons(AllSynapses &synapses, const SimulationInfo *sim_info, const SynapseIndexMap *synapseIndexMap) = 0;
#endif

    protected:
        int size;
 
    private:
        void freeResources();
};
