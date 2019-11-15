#include "ConnGrowth.h"
#include "ConnStatic.h"
#include "FConnections.h"
#include "Connections.h"
#include "ParameterManager.h"
#include "FSynapses.h"
#include <string>
#include <iostream>
#include <cassert>

using namespace std;

ParameterManager* createPM() {
    return new ParameterManager();
}

bool testConStaticCreate() {
    ParameterManager* pm = createPM();
    if (! pm->loadParameterFile("./configs/ConStatic.xml")) {
        return false;
    }
    string className;
    if (! pm->getStringByXpath("//ConnectionsParams/@class", className)) {
        return false;
    }
    assert(className == "ConnStatic");
    Connections* n = FConnections::get()->createConnections(className);
    assert(n != NULL);
    if (dynamic_cast<ConnStatic*>(n) == NULL) {
        // aPtr is NOT instance of B
        return false;
    }
    delete n;
    return true;
}

bool testConGrowthCreate() {
    ParameterManager* pm = createPM();
    if (! pm->loadParameterFile("./configs/ConGrowth.xml")) {
        return false;
    }
    string className;
    if (! pm->getStringByXpath("//ConnectionsParams/@class", className)) {
        return false;
    }
    assert(className == "ConnGrowth");
    Connections* n = FConnections::get()->createConnections(className);
    assert(n != NULL);
    if (dynamic_cast<ConnGrowth*>(n) == NULL) {
        // aPtr is NOT instance of B
        return false;
    }
    delete n;
    return true;
}

int main() {
    assert(testConGrowthCreate());
    assert(testConStaticCreate());
    return 0;
}
