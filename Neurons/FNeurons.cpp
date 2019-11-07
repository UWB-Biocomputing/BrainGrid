/**
 *  A factory class for creating Neuron objects.
 *  Lizzy Presland, October 2019
 */

#include "FNeurons.h"
#include "AllLIFNeurons.h"
#include "AllIZHNeurons.h"

// constructor
FNeurons::FNeurons() {
    // register neurons classes
    registerNeurons("AllLIFNeurons", &AllLIFNeurons::Create);
    registerNeurons("AllIZHNeurons", &AllIZHNeurons::Create);
}

FNeurons::~FNeurons() {
    createFunctions.clear();
}

/*
 *  Register neurons class and its creation function to the factory.
 *
 *  @param  neuronsClassName  neurons class name.
 *  @param  Pointer to the class creation function.
 */
void FNeurons::registerNeurons(const string &neuronsClassName, CreateNeuronsFn* function) {
    createFunctions[neuronsClassName] = function;
}

/**
 * Creates concrete instance of the desired neurons class.
 */
IAllNeurons* FNeurons::createNeurons(const string& className) {
    neuronsInstance = invokeNeuronsCreateFunction(className);
    neuronsInstance->createNeuronsProps();
    return neuronsInstance;
}

/**
 * Returns the static ::Create() method which allocates 
 * a new instance of the desired class.
 *
 * The calling method uses this retrieval mechanism in 
 * value assignment.
 */
IAllNeurons* FNeurons::invokeNeuronsCreateFunction(const string& className) {
    neuronClassName = className;
    NeuronFunctionMap::iterator it = createFunctions.find(neuronClassName);
    if (it != createFunctions.end()) return it->second();
    return NULL;
}

/*
 * Create an instance of the neurons class and copy neurons parameters from the 
 * neurons class object that has been already created.
 *
 * @return Poiner to the neurons object.
 */
IAllNeurons* FNeurons::createNeuronsCopy() {
    IAllNeurons* neurons = createNeuronsWithName(neuronClassName);
    neurons->createNeuronsProps();
    // copy basic parameters
    *neurons = *neuronsInstance;
    return neurons;
}
