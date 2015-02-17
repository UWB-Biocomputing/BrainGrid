#pragma once

using namespace std;

#include "Global.h"
#include "SimulationInfo.h"

class AllNeurons
{
    public:
        /*! A boolean which tracks whether the neuron has fired
         *  
         *  Usage: LOCAL VARIABLE
         *  - AllLIFNeurons::AllLIFNeurons() --- Initialized
         *  - LIFModel::readNeuron() --- Modified
         *  - LIFModel::writeNeuron() --- Accessed
         *  - LIFSingleThreadedModel::advanceNeurons() --- Accessed & Modified
         *  - LIFSingleThreadedModel::fire() --- Modified
         *  - GpuSim_struct.cu::advanceNeuronsDevice() --- Modified
         */
        bool *hasFired;

        /*! The number of spikes since the last growth cycle
         *  
         *  Usage: LOCAL VARIABLE
         *  - AllLIFNeurons::AllLIFNeurons() --- Initialized
         *  - LIFModel::updateHistory() --- Accessed
         *  - LIFModel::clearSpikeCounts() --- Modified
         *  - LIFSingleThreadedModel::fire() --- Modified
         *  - GpuSim_struct.cu::clearSpikeCounts() --- Modified
         *  - LIFGPUModel::copyDeviceSpikeCountsToHost() --- Accessed
         *  - GpuSim_struct.cu::advanceNeuronsDevice() --- Modified
         *  - Hdf5Recorder::compileHistories() --- Accessed
         *  - XmlRecorder::compileHistories() --- Accessed
         */
        int *spikeCount;

        /*! Step count for each spike fired by each neuron
         *
         *  Usage: LOCAL VARIABLE
         *  - LIFModel::createAllNeurons() --- Initialized
         *  - LIFSingleThreadedModel::fire() --- Modified
         *  - GpuSim_struct.cu::advanceNeuronsDevice() --- Modified
         *  - LIFGPUModel::copyDeviceSpikeHistoryToHost() --- Accessed
         *  - Hdf5Recorder::compileHistories() --- Accessed
         *  - XmlRecorder::compileHistories() --- Accessed
         */
        uint64_t **spike_history;

        /*! The neuron type map (INH, EXC).
         *  
         *  Usage: LOCAL CONSTANT
         *  - LIFModel::generateNeuronTypeMap --- Initialized
         *  - LIFModel::logSimStep() --- Accessed
         *  - LIFSingleThreadedModel::synType() --- Accessed
         *  - GpuSim_struct.cu::synType() --- Accessed
         *  - Hdf5Recorder::saveSimState() --- Accessed
         */
        neuronType *neuron_type_map;

        /*! List of summation points for each neuron
         *  
         *  Usage: LOCAL CONSTANT
         *  - AllLIFNeurons::AllLIFNeurons() --- Initialized
         *  - LIFModel::loadMemory() --- Accessed
         *  - LIFSingleThreadedModel::advanceNeuron() --- Accessed
         *  - LIFGPUModel::setupSim() --- Accessed
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
        void clearSpikeCounts(const SimulationInfo *sim_info);

    protected:
        int size;
};
