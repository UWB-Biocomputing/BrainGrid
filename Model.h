#ifndef _MODEL_H_
#define _MODEL_H_

#include <iostream>

using namespace std;

#include "include/tinyxml.h"

#include "global.h"
#include "AllNeurons.h"
#include "AllSynapses.h"

class Model {

    public:
		virtual ~Model() { }
        
        virtual bool readParameters(TiXmlElement *source) =0;
        
        virtual void printParameters(ostream &output) const;
        
        virtual void loadState(istream& input, AllNeurons &neurons, AllSynapses &synapses) =0;

        virtual void saveState(ostream& output, const AllNeurons &neurons, const SimulationInfo &sim_info) =0;

        virtual void createAllNeurons(const int num_neurons, AllNeurons &neurons, const SimulationInfo &sim_info) =0;
        
        virtual void setupSim(const int num_neurons, const SimulationInfo &sim_info) =0;

        virtual void advance(const int num_neurons, AllNeurons &neurons, AllSynapses &synapses) =0;
        
        virtual void updateConnections(const int currentStep, const int num_neurons, AllSynapses &synapses) =0;
};

#endif
