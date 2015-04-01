#pragma once

using namespace std;

#include "Global.h"
#include "SimulationInfo.h"
#include "AllNeurons.h"

class AllSpikingNeurons : public AllNeurons
{
    public:
        /*! A boolean which tracks whether the neuron has fired
         *  
         *  Usage: LOCAL VARIABLE
         *  - AllIFNeurons::AllIFNeurons() --- Initialized
         *  - LIFModel::readNeuron() --- Modified
         *  - LIFModel::writeNeuron() --- Accessed
         *  - SingleThreadedSpikingModel::advanceNeurons() --- Accessed & Modified
         *  - SingleThreadedSpikingModel::fire() --- Modified
         *  - GpuSim_struct.cu::advanceNeuronsDevice() --- Modified
         */
        bool *hasFired;

        /*! The number of spikes since the last growth cycle
         *  
         *  Usage: LOCAL VARIABLE
         *  - AllIFNeurons::AllIFNeurons() --- Initialized
         *  - LIFModel::updateHistory() --- Accessed
         *  - LIFModel::clearSpikeCounts() --- Modified
         *  - SingleThreadedSpikingModel::fire() --- Modified
         *  - GpuSim_struct.cu::clearSpikeCounts() --- Modified
         *  - GPUSpikingModel::copyDeviceSpikeCountsToHost() --- Accessed
         *  - GpuSim_struct.cu::advanceNeuronsDevice() --- Modified
         *  - Hdf5Recorder::compileHistories() --- Accessed
         *  - XmlRecorder::compileHistories() --- Accessed
         */
        int *spikeCount;

        /*! Step count for each spike fired by each neuron
         *
         *  Usage: LOCAL VARIABLE
         *  - LIFModel::createAllNeurons() --- Initialized
         *  - SingleThreadedSpikingModel::fire() --- Modified
         *  - GpuSim_struct.cu::advanceNeuronsDevice() --- Modified
         *  - GPUSpikingModel::copyDeviceSpikeHistoryToHost() --- Accessed
         *  - Hdf5Recorder::compileHistories() --- Accessed
         *  - XmlRecorder::compileHistories() --- Accessed
         */
        uint64_t **spike_history;

        AllSpikingNeurons();
        virtual ~AllSpikingNeurons();

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

#if defined(USE_GPU)
        virtual void allocNeuronDeviceStruct( void** allNeuronsDevice, SimulationInfo *sim_info ) = 0;
        virtual void deleteNeuronDeviceStruct( void* allNeuronsDevice, const SimulationInfo *sim_info ) = 0;
        virtual void copyNeuronHostToDevice( void* allNeuronsDevice, const SimulationInfo *sim_info ) = 0;
        virtual void copyNeuronDeviceToHost( void* allNeuronsDevice, const SimulationInfo *sim_info ) = 0;
#endif

    private:
        void freeResources();
};
