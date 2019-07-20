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
    cout << "\nEntered test method for reading valid + invalid XML files" << endl;
    ParameterManager* pm = new ParameterManager();
    // does the object get destroyed?
    string valid[] = {"../configfiles/test-medium-100.xml", "../configfiles/test-large-conected.xml", "../configfiles/test-medium.xml", "../configfiles/test-tiny.xml"};
    string invalid[] = {"../Core/BGDriver.cpp", "./this/path/doesnt/exist", "../configfiles/test-large-connected.xml", ""};
    for (int i = 0; i < 4; i++) 
        assert(pm->loadParameterFile(valid[i]));
    cout << "Beginning tests for invalid paths (failure expected)" << endl;
    for (int i = 0; i < 4; i++) 
        assert(!pm->loadParameterFile(invalid[i]));
    delete pm;
    return true;        // tests haven't crashed
}

bool testValidStringTargeting() {
    /*
     * Testing string retrieval for ../configfiles/test-medium-500.xml
     */
    cout << "\nEntered test method for targeting strings in XML file" << endl;
    ParameterManager* pm = new ParameterManager();
    if (pm->loadParameterFile("../configfiles/test-medium-500.xml")) {
        string valid_xpaths[] = {"/BGSimParams/SimInfoParams/OutputParams/stateOutputFileName/text()", "//stateOutputFileName/text()", "//NeuronsParams/@class"};
        string result[] = {"results/test-medium-500-out.xml", "results/test-medium-500-out.xml", "AllLIFNeurons"};
        string s;
        for (int i = 0; i < 3; i++) {
            cout << "Testing xpath: " << valid_xpaths[i] << endl;
            assert(pm->getStringByXpath(valid_xpaths[i], s));
            cout << "\tretrieved: " << s << endl;
            assert(s == result[i]);
            cout << "\tsucceeded" << endl;
        }
    } else {
        // expecting test filename to be valid
        return false;
    }
    delete pm;
    return true;        // tests haven't crashed
}

bool testValidIntTargeting() {
    cout << "\nEntered test method for targeting integers in XML file" << endl;
    ParameterManager* pm = new ParameterManager();
    cout << "Testing valid integers..." << endl;
    if (pm->loadParameterFile("../configfiles/test-medium-500.xml")) {
        string valid_xpath[] = {"//maxFiringRate/text()", "//PoolSize/x/text()", "//PoolSize/y/text()", "//PoolSize/z/text()", "//Seed/value/text()", "//numSims/text()"};
        int result[] = {200, 30, 30, 1, 1, 500};
        int val;
        for (int i = 0; i < 6; i++) {
            cout << "Testing xpath: " << valid_xpath[i] << endl;
            assert(pm->getIntByXpath(valid_xpath[i], val));
            cout << "\tretrieved: " << val << endl;
            assert(val == result[i]);
            cout << "\tsucceeded" << endl;
        }
    }
    delete pm;
    return true;
}

bool testValidFloatTargeting() {
    return true;
}

bool testValidDoubleTargeting() {
    return true;
}

int main() {
    cout << "\nRunning tests for ParameterManager.cpp functionality..." << endl;
    bool success = testConstructor();
    if (!success) return 1;
    success = testValidXmlFileReading();
    if (!success) return 1;
    success = testValidStringTargeting();
    if (!success) return 1;
    cout << "About to run testValidIntTargeting()..." << endl;
    success = testValidIntTargeting();
    if (!success) return 1;
    success = testValidFloatTargeting();
    if (!success) return 1;
    success = testValidDoubleTargeting();
    if (!success) return 1;
    return 0;
}
