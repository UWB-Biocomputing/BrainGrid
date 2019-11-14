/**
 *  A factory class for creating Connection objects.
 *  Lizzy Presland, October 2019
 */

#include "FConnections.h"
#include "ConnGrowth.h"
#include "ConnStatic.h"

// constructor
FConnections::FConnections() {
    // register connections classes
    registerConnection("ConnGrowth", &ConnGrowth::Create);
    registerConnection("ConnStatic", &ConnStatic::Create);
}

FConnections::~FConnections() {
    createFunctions.clear();
}

/*
 *  Register connections class and its creation function to the factory.
 *
 *  @param  connectionsClassName  connections class name.
 *  @param  Pointer to the class creation function.
 */
void FConnections::registerConnection(const string &connectionsClassName, CreateConnectionsFn function) {
    createFunctions[connectionsClassName] = function;
}

/**
 * Creates concrete instance of the desired connections class.
 */
Connections* FConnections::createConnections(const string& className) {
    connectionsInstance = invokeConnectionsCreateFunction(className);
    // no createProps() call required
    return connectionsInstance;
}

/**
 * Returns the static ::Create() method which allocates 
 * a new instance of the desired class.
 *
 * The calling method uses this retrieval mechanism in 
 * value assignment.
 */
Connections* FConnections::invokeConnectionsCreateFunction(const string& className) {
    connectionClassName = className;
    ConnectionFunctionMap::iterator it = createFunctions.find(connectionClassName);
    if (it != createFunctions.end()) return it->second();
    return NULL;
}
