#ifndef _MODEL_H_
#define _MODEL_H_

#include <iostream>

using namespace std;

#include "include/tinyxml.h"

#include "global.h"
#include "AllNeurons.h"
#include "AllSynapses.h"
#include "SimulationInfo.h"

class Model {

    public:
		virtual ~Model() { }
        
        virtual bool readParameters(TiXmlElement *source) =0;
        
        virtual void printParameters(ostream &output) const;
        
        virtual void loadMemory(istream& input, AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo &sim_info) =0;

        virtual void saveMemory(ostream& output, AllNeurons &neurons, AllSynapses &synapses, FLOAT simulation_step) =0;

        virtual void saveState(ostream& output, const AllNeurons &neurons, const SimulationInfo &sim_info) =0;

        virtual void createAllNeurons(AllNeurons &neurons, const SimulationInfo &sim_info) =0;
        
        virtual void setupSim(const int num_neurons, const SimulationInfo &sim_info) =0;

        virtual void advance(AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo &sim_info) =0;
        
        virtual void updateConnections(const int currentStep, AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo &sim_info) =0;

        virtual void cleanupSim(AllNeurons &neurons, SimulationInfo &sim_info) =0;

        virtual void logSimStep(const AllNeurons &neurons, const AllSynapses &synapses, const SimulationInfo &sim_info) const =0;
};

#endif
