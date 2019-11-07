#include "AllDSSynapses.h"
#include "AllSTDPSynapses.h"
#include "AllDynamicSTDPSynapses.h"
#include "FSynapses.h"
#include "ParameterManager.h"
#include <string>
#include <iostream>
#include <cassert>

using namespace std;

ParameterManager* createPM() {
    return new ParameterManager();
}

bool testDSSynapsesCreate() {
    ParameterManager* pm = createPM();
    if (! pm->loadParameterFile("./configs/DS-Synapse.xml")) {
        return false;
    }
    string className;
    if (! pm->getStringByXpath("//SynapsesParams/@class", className)) {
        return false;
    }
    assert(className == "AllDSSynapses");
    IAllSynapses* n = FSynapses::get()->createSynapses(className);
    assert(n != NULL);
    if (dynamic_cast<AllDSSynapses*>(n) == NULL) {
        // aPtr is NOT instance of B
        return false;
    }
    delete n;
    return true;
}

bool testSTDPSynapsesCreate() {
    ParameterManager* pm = createPM();
    if (! pm->loadParameterFile("./configs/STDP-Synapse.xml")) {
        return false;
    }
    string className;
    if (! pm->getStringByXpath("//SynapsesParams/@class", className)) {
        return false;
    }
    assert(className == "AllSTDPSynapses");
    IAllSynapses* n = FSynapses::get()->createSynapses(className);
    assert(n != NULL);
    if (dynamic_cast<AllSTDPSynapses*>(n) == NULL) {
        // aPtr is NOT instance of B
        return false;
    }
    delete n;
    return true;
}

bool testDynamicSTDPSynapsesCreate() {
    ParameterManager* pm = createPM();
    if (! pm->loadParameterFile("./configs/Dyn-STDP-Synapse.xml")) {
        return false;
    }
    string className;
    if (! pm->getStringByXpath("//SynapsesParams/@class", className)) {
        return false;
    }
    assert(className == "AllDynamicSTDPSynapses");
    IAllSynapses* n = FSynapses::get()->createSynapses(className);
    assert(n != NULL);
    if (dynamic_cast<AllDynamicSTDPSynapses*>(n) == NULL) {
        // aPtr is NOT instance of B
        return false;
    }
    delete n;
    return true;
}

int main() {
    assert(testDSSynapsesCreate());
    assert(testSTDPSynapsesCreate());
    assert(testDynamicSTDPSynapsesCreate());
    return 0;
}
