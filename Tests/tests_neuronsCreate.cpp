#include "AllLIFNeurons.h"
#include "AllIZHNeurons.h"
#include "FNeurons.h"
#include "ParameterManager.h"
#include <string>
#include <iostream>
#include <cassert>

using namespace std;

ParameterManager* createPM() {
    return new ParameterManager();
}

bool testAllLIFNeuronsCreate() {
    ParameterManager* pm = createPM();
    if (! pm->loadParameterFile("./configs/AllLIF.xml")) {
        return false;
    }
    string className;
    if (! pm->getStringByXpath("//NeuronsParams/@class", className)) {
        return false;
    }
    assert(className == "AllLIFNeurons");
    IAllNeurons* n = FNeurons::get()->createNeurons(className);
    assert(n != NULL);
    if (dynamic_cast<AllLIFNeurons*>(n) == NULL) {
        // aPtr is NOT instance of B
        return false;
    }
    delete n;
    return true;
}

bool testAllIZHNeuronsCreate() {
    ParameterManager* pm = createPM();
    if (! pm->loadParameterFile("./configs/AllIZH.xml")) {
        return false;
    }
    string className;
    if (! pm->getStringByXpath("//NeuronsParams/@class", className)) {
        return false;
    }
    assert(className == "AllIZHNeurons");
    IAllNeurons* n = FNeurons::get()->createNeurons(className);
    assert(n != NULL);
    if (dynamic_cast<AllIZHNeurons*>(n) == NULL) {
        // aPtr is NOT instance of B
        return false;
    }
    delete n;
    return true;
}

int main() {
    assert(testAllLIFNeuronsCreate());
    assert(testAllIZHNeuronsCreate());
    return 0;
}
