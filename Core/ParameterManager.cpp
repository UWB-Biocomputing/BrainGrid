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
    delete xmlDoc;
    delete root;
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
    // instantiate the document root object
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
 * TODO: return what if not found?
 */
bool ParameterManager::getStringByXpath(string xpath, string& var) {
    if (!checkDocumentStatus()) return false;
    // static implementation usage
    string result;
    bool succeeded = TinyXPath::o_xpath_string(root, xpath.c_str(), var);
    if (!succeeded) {
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
 * TODO: return what if not found?
 */
int ParameterManager::getIntByXpath(string xpath) {
    return -1;
}

/**
 * Interface method to pull a double value from the xml 
 * schema. The calling object must know the xpath to retrieve 
 * the value.
 *
 * TODO: return what if not found?
 */
double ParameterManager::getDoubleByXpath(string xpath) {
    return -1.0;
}

/**
 * Interface method to pull a float value from the xml 
 * schema. The calling object must know the xpath to retrieve 
 * the value.
 *
 * TODO: return what if not found?
 */
float ParameterManager::getFloatByXpath(string xpath) {
    return -1.0;
}
