#pragma once

using namespace std;

#include "Global.h"
#include "SimulationInfo.h"
#include "SynapseIndexMap.h"

class AllSynapses;

class AllNeurons
{
    public:
        /*! The neuron type map (INH, EXC).
         *  
         *  Usage: LOCAL CONSTANT
         *  - LIFModel::generateNeuronTypeMap --- Initialized
         *  - LIFModel::logSimStep() --- Accessed
         *  - SingleThreadedSpikingModel::synType() --- Accessed
         *  - GpuSim_struct.cu::synType() --- Accessed
         *  - Hdf5Recorder::saveSimState() --- Accessed
         */
        neuronType *neuron_type_map;

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

        /*! The starter existence map (T/F).
         *  
         *  Usage: LOCAL CONSTANT
         *  - LIFModel::initStarterMap() --- Initialized
         *  - LIFModel::createAllNeurons() --- Accessed
         *  - LIFModel::logSimStep() --- Accessed
         *  - LIFModel::getStarterNeuronMatrix() --- Accessed
         *  - Hdf5Recorder::saveSimState() --- Accessed
         *  - XmlRecorder::saveSimState() --- Accessed
         */
        bool *starter_map;

        AllNeurons();
        virtual ~AllNeurons();

        virtual void setupNeurons(SimulationInfo *sim_info);
        virtual void cleanupNeurons();
        virtual int numParameters() = 0;
        virtual int readParameters(const TiXmlElement& element) = 0;
        virtual void printParameters(ostream &output) const = 0;
        virtual void createAllNeurons(SimulationInfo *sim_info) = 0;
        virtual string toString(const int i) const = 0;
        virtual void readNeurons(istream &input, const SimulationInfo *sim_info) = 0;
        virtual void writeNeurons(ostream& output, const SimulationInfo *sim_info) const = 0;

        // Determines the type of synapse for a synapse between two neurons.
        synapseType synType(const int src_neuron, const int dest_neuron);
#if defined(USE_GPU)
        virtual void allocNeuronDeviceStruct( void** allNeuronsDevice, SimulationInfo *sim_info ) = 0;
        virtual void deleteNeuronDeviceStruct( void* allNeuronsDevice, const SimulationInfo *sim_info ) = 0;
        virtual void copyNeuronHostToDevice( void* allNeuronsDevice, const SimulationInfo *sim_info ) = 0;
        virtual void copyNeuronDeviceToHost( void* allNeuronsDevice, const SimulationInfo *sim_info ) = 0;
        virtual void copyNeuronDeviceSpikeHistoryToHost( void* allNeuronsDevice, const SimulationInfo *sim_info ) = 0;
        virtual void copyNeuronDeviceSpikeCountsToHost( void* allNeuronsDevice, const SimulationInfo *sim_info ) = 0;
        virtual void clearNeuronSpikeCounts( void* allNeuronsDevice, const SimulationInfo *sim_info ) = 0;
        // Update the state of all neurons for a time step
        virtual void advanceNeurons(AllNeurons* allNeuronsDevice, AllSynapses* allSynapsesDevice, const SimulationInfo *sim_info, float* randNoise, SynapseIndexMap* synapseIndexMapDevice) = 0;
#else
        // Update the state of all neurons for a time step
        virtual void advanceNeurons(AllSynapses &synapses, const SimulationInfo *sim_info, const SynapseIndexMap *synapseIndexMap) = 0;
#endif

    protected:
        int size;
 
    private:
        void freeResources();
};
