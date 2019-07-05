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
    ParameterManager* pm = new ParameterManager();
    // does the object get destroyed?
    string valid[] = {"../configfiles/test-medium-100.xml", "../configfiles/test-large-conected.xml", "../configfiles/test-medium.xml", "../configfiles/test-tiny.xml"};
    string invalid[] = {"../Core/BGDriver.cpp", "./this/path/doesnt/exist", "../configfiles/test-large-connected.xml", ""};
    for (int i = 0; i < 4; i++) 
        assert(pm->loadParameterFile(valid[i]));
    for (int i = 0; i < 4; i++) 
        assert(!pm->loadParameterFile(invalid[i]));
    delete pm;
    return true;
}

bool testValidStringTargeting() {
    /*
     * Testing string retrieval for ../configfiles/test-medium-500.xml
     */
    ParameterManager* pm = new ParameterManager();
    if (pm->loadParameterFile("../configfiles/test-medium-500.xml")) {
        string xpaths[] = {"/BGSimParams/SimInfoParams/"};
        string result[] = {};
        string s;
    }
}

int main() {
    bool success = testConstructor();
    if (!success) return 1;
    success = testValidXmlFileReading();
    if (!success) return 1;
    success = testValidStringTargeting();
    if (!success) return 1;
    return 0;
}
