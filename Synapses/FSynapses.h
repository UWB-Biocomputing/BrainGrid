/**
 *  A factory class for creating Neuron objects.
 *  Lizzy Presland, October 2019
 */

#pragma once

#include <map>
#include <string>
#include "Global.h"
#include "IAllSynapses.h"

using namespace std;

class FSynapses {

    public:
        FSynapses();
        ~FSynapses();

        static FSynapses *get() {
            static FSynapses instance;
            return &instance;
        }

        // Invokes constructor for desired concrete class
        IAllSynapses* createSynapses(const string& className);
        
    private:
        void registerSynapse(const string& className, CreateSynapsesFn* function);
        IAllSynapses* invokeSynapseCreateFunction(const string& className);
        // Pointer to synapses instance
        IAllSynapses* synapsesInstance;
        string synapseClassName;
        /* Type definitions */
        // Defines function type for usage in internal map
        typedef IAllSynapses* (*CreateSynapsesFn)(void);
        // Defines map between class name and corresponding ::Create() function.
        typedef map<string, CreateSynapsesFn> SynapseFunctionMap;
        // Makes class-to-function map an internal factory member.
        SynapseFunctionMap createFunctions;

};
