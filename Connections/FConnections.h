/**
 *  A factory class for creating Neuron objects.
 *  Lizzy Presland, October 2019
 */

#pragma once

#include <map>
#include <string>
#include "Global.h"
#include "Connections.h"

using namespace std;

class FConnections {

    public:
        FConnections();
        ~FConnections();

        static FConnections *get() {
            static FConnections instance;
            return &instance;
        }

        // Invokes constructor for desired concrete class
        Connections* createConnections(const string& className);
        // No copy method required for this class at this time.
        
    private:
        // Pointer to connections instance
        Connections* connectionsInstance;
        string connectionClassName;
        /* Type definitions */
        // Defines function type for usage in internal map
        typedef Connections* (*CreateConnectionsFn)(void);
        // Defines map between class name and corresponding ::Create() function.
        typedef map<string, CreateConnectionsFn> ConnectionFunctionMap;
        // Makes class-to-function map an internal factory member.
        ConnectionFunctionMap createFunctions;
        // Retrieves and invokes correct ::Create() function
        Connections* invokeConnectionsCreateFunction(const string& className);
        void registerConnection(const string& connectionsClassName, CreateConnectionsFn* function);

};
