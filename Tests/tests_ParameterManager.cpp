/**
 * Test file for (new) class ParameterManager.
 *
 * Created by Lizzy Presland, 2019
 */

#include <cassert>
#include <iostream>
#include <string>
#include "../Core/ParameterManager.h"
#include <limits>
#include <algorithm>
#include <cmath>
#include "BGTypes.h"

using namespace std;

template<typename T>
static bool AreEqual(T f1, T f2) { 
  return (fabs(f1 - f2) <= numeric_limits<T>::epsilon() * fmax(fabs(f1), fabs(f2)));
}

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
    cout << "\nEntered test method for targeting valid integers in XML file" << endl;
    ParameterManager* pm = new ParameterManager();
    if (pm->loadParameterFile("../configfiles/test-medium-500.xml")) {
        cout << "Testing valid integers..." << endl;
        string valid_xpath[] = {"//maxFiringRate/text()", "//PoolSize/x/text()", "//PoolSize/y/text()", "//PoolSize/z/text()", "//Seed/value/text()", "//numSims/text()"};
        int result[] = {200, 30, 30, 1, 1, 500};
        int val;
        for (int i = 0; i < 6; i++) {
            assert(pm->getIntByXpath(valid_xpath[i], val));
            assert(val == result[i]);
        }
        /*
         * Test the following invalid paths:
         * string result from XMLNode
         * string result from string text() value
         * float/double result
         * float/double result with scientific notation
         * @name value that is a string
         * invalid xpath (node doesn't exist)
         */
        cout << "Testing NON-valid integers..." << endl;
        string invalid_xpath[] = {"//Iinject", "//activeNListFileName/text()", 
                                  "//beta/text()", "//Iinject/min/text()", 
                                  "//LayoutFiles/@name", "//NoSuchPath", ""};
        for (int i = 0; i < 7; i++) {
            assert(!(pm->getIntByXpath(invalid_xpath[i], val)));
        }
    }
    delete pm;
    return true;
}



bool testValidFloatTargeting() {
    cout << "\nEntered test method for targeting valid floats in XML file" << endl;
    ParameterManager* pm = new ParameterManager();
    if (pm->loadParameterFile("../configfiles/test-medium-500.xml")) {
        string valid_xpaths[] = { "//Vthresh/min/text()", "//Vresting/min/text()", "//Tsim/text()", "//z/text()" };
        float vals[] = { 15.0e-03f, 0.0f, 100.0f, 1 };
        float var;
        for (int i = 0; i < 4; i++) {
            assert(pm->getFloatByXpath(valid_xpaths[i], var));
            assert(AreEqual(var, vals[i]));
        }
        string invalid_xpaths[] = { "//starter_vthresh", "//nonexistent", "" };
        for (int i = 0; i < 3; i++) {
            assert(!pm->getFloatByXpath(invalid_xpaths[i], var));
        }
    }
    return true;
}

bool testValidDoubleTargeting() {
    cout << "\nEntered test method for targeting valid doubles in XML file" << endl;
    ParameterManager* pm = new ParameterManager();
    if (pm->loadParameterFile("../configfiles/test-medium-500.xml")) {
        string valid_xpaths[] = { "//Vthresh/min/text()", "//Vresting/min/text()", "//Tsim/text()", "//z/text()" };
        double vals[] = { 15.0e-03, 0.0, 100.0, 1 };
        double var;
        for (int i = 0; i < 4; i++) {
            assert(pm->getDoubleByXpath(valid_xpaths[i], var));
            assert(AreEqual(var, vals[i]));
        }
        string invalid_xpaths[] = { "//starter_vthresh", "//nonexistent", "" };
        for (int i = 0; i < 3; i++) {
            assert(!pm->getDoubleByXpath(invalid_xpaths[i], var));
        }
    }
    return true;
}

bool testValidBGFloatTargeting() {
    cout << "\nEntered test method for targeting valid BGFLOATs in XML file" << endl;
    ParameterManager* pm = new ParameterManager();
    if (pm->loadParameterFile("../configfiles/test-medium-500.xml")) {
        string valid_xpaths[] = { "//Vthresh/min/text()", "//Vresting/min/text()", "//Tsim/text()", "//z/text()" };
        BGFLOAT vals[] = { 15.0e-03, 0.0, 100.0, 1 };
        BGFLOAT var;
        for (int i = 0; i < 4; i++) {
            assert(pm->getBGFloatByXpath(valid_xpaths[i], var));
            assert(AreEqual(var, vals[i]));
        }
        string invalid_xpaths[] = { "//starter_vthresh", "//nonexistent", "" };
        for (int i = 0; i < 3; i++) {
            assert(!pm->getBGFloatByXpath(invalid_xpaths[i], var));
        }
    }
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
    success = testValidIntTargeting();
    if (!success) return 1;
    success = testValidFloatTargeting();
    if (!success) return 1;
    success = testValidDoubleTargeting();
    if (!success) return 1;
    success = testValidBGFloatTargeting();
    if (!success) return 1;
    return 0;
}
