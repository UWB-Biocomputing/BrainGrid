/**
 *  A factory class for creating Neuron objects.
 *  Lizzy Presland, October 2019
 */

#include <string>
#include "AllDSSynapses.h"
#include "AllSTDPSynapses.h"
#include "AllDynamicSTDPSynapses.h"

using namespace std;

// constructor
FSynapses::FSynapses() {
    // register synapses classes
    registerSynapses("AllSpikingSynapses", &AllSpikingSynapses::Create);
    registerSynapses("AllDSSynapses", &AllDSSynapses::Create);
    registerSynapses("AllSTDPSynapses", &AllSTDPSynapses::Create);
    registerSynapses("AllDynamicSTDPSynapses", &AllDynamicSTDPSynapses::Create);
}

FSynapses::~FSynapses() {
    createFunctions.clear();
}

/*
 *  Register synapse class and its creation function to the factory.
 *
 *  @param  synapseClassName  synapse class name.
 *  @param  Pointer to the class creation function.
 */
void FSynapses::registerSynapse(const string &synapseClassName, CreateSynapsesFn* function) {
    createFunctions[synapseClassName] = function;
}

/**
 * Creates concrete instance of the desired connections class.
 */
IAllSynapses* FSynapses::createSynapses(const string& synapseClassName) {
    synapsesInstance = invokeSynapseCreateFunction(synapseClassName);
    return synapsesInstance;
}    

/**
 * Returns the static ::Create() method which allocates 
 * a new instance of the desired class.
 *
 * The calling method uses this retrieval mechanism in 
 * value assignment.
 */
IAllSynapses* FSynapses::invokeSynapseCreateFunction(const string& className) {
    synapseClassName = className;
    SynapseFunctionMap::iterator it = createFunctions.find(synapseClassName);
    if (it != createFunctions.end()) return it->second();
    return NULL;
}
