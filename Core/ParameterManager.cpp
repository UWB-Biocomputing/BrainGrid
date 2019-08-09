/**
 * A class which contains and manages access to the XML 
 * parameter file used by a simulator instance at runtime.
 *
 * The class provides a simple interface to access 
 * parameters with the following assumptions:
 *   - The client's TODO is correct?: ::PopulateParameters() 
 *     method knows the XML layout of the parameter file.
 *   - The client makes all its own schema calls as needed.
 *   - The client will validate its own parameters unless 
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
#include <stdexcept>
#include "BGTypes.h"

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
    string tmp;
    if (!getStringByXpath(xpath, tmp)) {
        cerr << "Failed loading simulation parameter for xpath " 
             << xpath << endl;
        return false;
    }
    // TODO: optimize this. could use <regex>, isdigit(), or others.
    // Workaround for standard value conversion functions.
    // stoi() will cast floats to ints.
    if (tmp.find('e') != string::npos || tmp.find('.') != string::npos) {
        cerr << "Parsed parameter is likely a float/double value. "
             << "Terminating integer cast. Value: " << tmp << endl;
        return false;
    }
    try {
        var = stoi(tmp);
    } catch (invalid_argument arg_exception) {
        cerr << "Parsed parameter could not be parsed as an integer. Value: "
             << tmp << endl;
        return false;
    } catch (out_of_range range_exception) {
        cerr << "Parsed string parameter could not be converted to an integer. Value: "
             << tmp << endl;
        return false;
    }
    return true;
}

/**
 * Interface method to pull a double value from the xml 
 * schema. The calling object must know the xpath to retrieve 
 * the value.
 *
 * @param xpath The xpath for the desired double value in the XML file
 * @param var The variable to store the double result into
 * @return bool A T/F flag indicating whether the retrieval succeeded
 */
bool ParameterManager::getDoubleByXpath(string xpath, double& var) {
    if (!checkDocumentStatus()) return false;
    string tmp;
    if (!getStringByXpath(xpath, tmp)) {
        cerr << "Failed loading simulation parameter for xpath " 
             << xpath << endl;
        return false;
    }
    try {
        var = stod(tmp);
    } catch (invalid_argument arg_exception) {
        cerr << "Parsed parameter could not be parsed as a double. Value: "
             << tmp << endl;
        return false;
    } catch (out_of_range range_exception) {
        cerr << "Parsed string parameter could not be converted to a double. Value: "
             << tmp << endl;
        return false;
    }
    return true;
}


/**
 * Interface method to pull a float value from the xml 
 * schema. The calling object must know the xpath to retrieve 
 * the value.
 *
 * @param xpath The xpath for the desired float value in the XML file
 * @param var The variable to store the float result into
 * @return bool A T/F flag indicating whether the retrieval succeeded
 */
bool ParameterManager::getFloatByXpath(string xpath, float& var) {
    if (!checkDocumentStatus()) return false;
    string tmp;
    if (!getStringByXpath(xpath, tmp)) {
        cerr << "Failed loading simulation parameter for xpath " 
             << xpath << endl;
        return false;
    }
    try {
        var = stof(tmp);
    } catch (invalid_argument arg_exception) {
        cerr << "Parsed parameter could not be parsed as a float. Value: "
             << tmp << endl;
        return false;
    } catch (out_of_range range_exception) {
        cerr << "Parsed string parameter could not be converted to a float. Value: "
             << tmp << endl;
        return false;
    }
    return true;
}

/**
 * Interface method to pull a BGFLOAT value from the xml 
 * schema. The calling object must know the xpath to retrieve 
 * the value.
 *
 * This method is a wrapper to run the correct calls based on 
 * how BGFLOAT is defined for the simulator. (For multi-threaded
 * usage, floats are used due to register size on GPUs, and 
 * double usage is available for single-threaded CPU instances.)
 *
 * @param xpath The xpath for the desired float value in the XML file
 * @param var The variable to store the float result into
 * @return bool A T/F flag indicating whether the retrieval succeeded
 */
bool ParameterManager::getBGFloatByXpath(string xpath, BGFLOAT& var) {
    #ifdef SINGLEPRECISION
        return getFloatByXpath(xpath, var);
    #endif
    #ifdef DOUBLEPRECISION
        return getDoubleByXpath(xpath, var);
    #endif
    cerr << "Could not infer primitive type for BGFLOAT variable."
         << endl;
    return false;
}
