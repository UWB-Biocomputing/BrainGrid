/**
 * A class which contains and manages access to the XML 
 * parameter file used by a simulator instance at runtime.
 *
 * The class provides a simple interface to access 
 * parameters with the following assumptions:
 *   - The class' TODO is correct?: ::PopulateParameters() 
 *     method knows the XML layout of the parameter file.
 *   - The class makes all its own schema calls as needed.
 *   - The class will validate its own parameters unless 
 *     otherwise defined here.
 *
 * This class makes use of TinyXPath, an open-source utility 
 * which enables XPath parsing of a TinyXml document object.
 * See the documentation here: http://tinyxpath.sourceforge.net/doc/index.html
 *
 * Created by Lizzy Presland, 2019
 * Supervised by Dr. Michael Stiber, UW Bothell CSSE Division
 */

#include "ParameterManager.h"
#include "tinyxml.h"
#include "xpath_static.h"
#include <iostream>
#include <string>

// ----------------------------------------------------
// ----------------- UTILITY METHODS ------------------
// ----------------------------------------------------

/**
 * Class constructor
 * Initialize any heap variables to null
 */
ParameterManager::ParameterManager() {
    xmlDoc = nullptr;
    root = nullptr;
}

/**
 * Class destructor
 * Deallocate all heap memory managed by the class
 */
ParameterManager::~ParameterManager() {
    if (xmlDoc != nullptr) delete xmlDoc;
}

/**
 * Loads the XML file into a TinyXML tree.
 * This is the starting point for all XPath retrieval calls 
 * for the client classes to use.
 */
bool ParameterManager::loadParameterFile(string path) {
    // load the XML document
    xmlDoc = new TiXmlDocument(path.c_str());
    if (!xmlDoc->LoadFile()) {
        cerr << "Failed loading simulation parameter file "
             << path << ":" << "\n\t" << xmlDoc->ErrorDesc()
             << endl;
        cerr << " error: " << xmlDoc->ErrorRow() << ", " << xmlDoc->ErrorCol()
             << endl;
        return false;
    }
    // assign the document root object
    root = xmlDoc->RootElement();
    return true;
}

/**
 * A utility method to ensure the XML document objects exist.
 * If they don't exist, an XPath can't be computed and the 
 * interface methods should terminate early.
 */
bool ParameterManager::checkDocumentStatus() {
    return xmlDoc != nullptr && root != nullptr;
}

// ----------------------------------------------------
// ---------------- INTERFACE METHODS -----------------
// ----------------------------------------------------

/**
 * Interface method to pull a string object from the xml 
 * schema. The calling object must know the xpath to retrieve 
 * the value.
 * 
 * @param xpath The xpath for the desired string value in the XML file
 * @param var The variable to store the string result into
 * @return bool A T/F flag indicating whether the retrieval succeeded
 */
bool ParameterManager::getStringByXpath(string xpath, string& var) {
    if (!checkDocumentStatus()) return false;
    if (!TinyXPath::o_xpath_string(root, xpath.c_str(), var)) {
        cerr << "Failed loading simulation parameter for xpath " 
             << xpath << endl;
        // TODO: possibly get better error information?
        return false;
    }
    return true;
}

/**
 * Interface method to pull an integer value from the xml 
 * schema. The calling object must know the xpath to retrieve 
 * the value.
 *
 * @param xpath The xpath for the desired int value in the XML file
 * @param var The variable to store the int result into
 * @return bool A T/F flag indicating whether the retrieval succeeded
 */
bool ParameterManager::getIntByXpath(string xpath, int& var) {
    if (!checkDocumentStatus()) return false;
    if (!TinyXPath::o_xpath_int(root, xpath.c_str(), var)) {
        cerr << "Failed loading simulation parameter for xpath " 
             << xpath << endl;
        // TODO: possibly get better error information?
        return false;
    }
    return true;
}

/**
 * Interface method to pull a double value from the xml 
 * schema. The calling object must know the xpath to retrieve 
 * the value.
 *
 * NOTE: TinyXPath does not support value extraction for 
 * floating-point numbers. As such, if floats are used, 
 *
 * @param xpath The xpath for the desired double value in the XML file
 * @param var The variable to store the double result into
 * @return bool A T/F flag indicating whether the retrieval succeeded
 */
bool ParameterManager::getDoubleByXpath(string xpath, double& var) {
    if (!checkDocumentStatus()) return false;
    if (!TinyXPath::o_xpath_double(root, xpath.c_str(), var)) {
        cerr << "Failed loading simulation parameter for xpath " 
             << xpath << endl;
        // TODO: possibly get better error information?
        return false;
    }
    return true;
}


/**
 * Interface method to pull a float value from the xml 
 * schema. The calling object must know the xpath to retrieve 
 * the value.
 *
 * NOTE: TinyXPath does not natively support value extraction 
 * for floating-point numbers. As such, if floats are required, 
 * understand that precision may be lost in raw casts.
 *
 * TODO: talk to Dr. Stiber about precision requirements for 
 * non-integer values.
 *
 * @param xpath The xpath for the desired float value in the XML file
 * @param var The variable to store the float result into
 * @return bool A T/F flag indicating whether the retrieval succeeded
 */
bool ParameterManager::getFloatByXpath(string xpath, float& var) {
    if (!checkDocumentStatus()) return false;
    double xmlFileVal;
    if (!TinyXPath::o_xpath_double(root, xpath.c_str(), xmlFileVal)) {
        cerr << "Failed loading simulation parameter for xpath " 
             << xpath << endl;
        // TODO: possibly get better error information?
        return false;
    }
    var = (float) xmlFileVal;
    return true;
}
