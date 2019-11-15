#include "FixedLayout.h"
#include "DynamicLayout.h"
#include "FLayout.h"
#include "ParameterManager.h"
#include <string>
#include <iostream>
#include <cassert>

using namespace std;

ParameterManager* createPM() {
    return new ParameterManager();
}

bool testFixedLayoutCreate() {
    ParameterManager* pm = createPM();
    if (! pm->loadParameterFile("./configs/FixLayout.xml")) {
        return false;
    }
    string className;
    if (! pm->getStringByXpath("//LayoutParams/@class", className)) {
        return false;
    }
    assert(className == "FixedLayout");
    Layout* n = FLayout::get()->createLayout(className);
    assert(n != NULL);
    if (dynamic_cast<FixedLayout*>(n) == NULL) {
        // aPtr is NOT instance of B
        return false;
    }
    delete n;
    return true;
}

bool testDynamicLayoutCreate() {
    ParameterManager* pm = createPM();
    if (! pm->loadParameterFile("./configs/DynLayout.xml")) {
        return false;
    }
    string className;
    if (! pm->getStringByXpath("//LayoutParams/@class", className)) {
        return false;
    }
    assert(className == "DynamicLayout");
    Layout* n = FLayout::get()->createLayout(className);
    assert(n != NULL);
    if (dynamic_cast<DynamicLayout*>(n) == NULL) {
        // aPtr is NOT instance of B
        return false;
    }
    delete n;
    return true;
}

int main() {
    assert(testFixedLayoutCreate());
    assert(testDynamicLayoutCreate());
    return 0;
}
