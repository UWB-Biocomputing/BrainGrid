#ifndef _MODEL_H_
#define _MODEL_H_

#include "include/tinyxml.h"

#include "global.h"

#include "AllNeurons.h"
#include "AllSynapses.h"

class Model {

    public:
        
        virtual bool readParameters(TiXmlElement *source) =0;
        
        virtual void printParameters(ostream &output) const;
        
        virtual void createAllNeurons(FLOAT neuron_count, AllNeurons &neurons) const =0;
        
        virtual void advance(FLOAT neuron_count, AllNeurons &neurons, AllSynapses &synapses) =0;
        
        virtual void updateConnections(Network &network) const =0;
};

#endif
