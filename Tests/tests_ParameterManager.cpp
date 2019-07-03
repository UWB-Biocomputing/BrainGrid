/**
 * Test file for (new) class ParameterManager.
 *
 * Created by Lizzy Presland, 2019
 */

#include <cassert>
#include <iostream>
#include <string>
#include "../Core/ParameterManager.h"

using namespace std;

bool testConstructor() {
    // does the object get created?
    ParameterManager* pm = new ParameterManager();
    // does the object get destroyed?
    delete pm;
    return true;
}

bool testValidXmlFileReading() {
    // TODO
    return true;
}

int main() {
    bool success = testConstructor();
    if (!success) return 1;
    success = testValidXmlFileReading();
    if (!success) return 1;
    return 0;
}
