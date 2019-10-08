/**
 *  A factory class for creating Neuron objects.
 *  Lizzy Presland, October 2019
 */

#pragma once

#include <map>
#include <string>
#include "Global.h"
#include "IAllNeurons.h"

using namespace std;

class FNeurons {

    public:
        FNeurons();
        ~FNeurons();

        static FNeurons *get() {
            static FNeurons instance;
            return &instance;
        }

        // Invokes constructor for desired concrete class
        IAllNeurons* createNeurons(const string& className);
        // Shortcut for copy constructor for existing concrete object
        IAllNeurons* createNeuronsCopy();
        
    private:
        // Pointer to neurons instance
        IAllNeurons* neuronsInstance;
        string neuronClassName;
        /* Type definitions */
        // Defines function type for usage in internal map
        typedef IAllNeurons* (*CreateNeuronsFn)(void);
        // Defines map between class name and corresponding ::Create() function.
        typedef map<string, CreateNeuronsFn> NeuronFunctionMap;
        // Makes class-to-function map an internal factory member.
        NeuronFunctionMap createFunctions;
        // Retrieves and invokes correct ::Create() function
        IAllNeurons* invokeNeuronsCreateFunction(const string& className);
};
