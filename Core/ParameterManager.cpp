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
// TODO: are these declarations necessary?
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
}

/**
 * Class destructor
 * Deallocate all heap memory managed by the class
 */
ParameterManager::~ParameterManager() {
    delete xmlDoc;
}


bool ParameterManager::loadParameterFile(string path) {
    xmlDoc = new TiXmlDocument(path.c_str());
    if (!xmlDoc->LoadFile()) {
        cerr << "Failed loading simulation parameter file "
             << path << ":" << "\n\t" << xmlDoc->ErrorDesc()
             << endl;
        cerr << " error: " << xmlDoc->ErrorRow() << ", " << xmlDoc->ErrorCol()
             << endl;
        return false;
    }
    return true;
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
string ParameterManager::getStringByXpath(string xpath) {
    return "";
}

/**
 * Interface method to pull an integer value from the xml 
 * schema. The calling object must know the xpath to retrieve 
 * the value.
 *
 * TODO: return what if not found?
 */
int getIntByXpath(string xpath) {
    return -1;
}

/**
 * Interface method to pull a double value from the xml 
 * schema. The calling object must know the xpath to retrieve 
 * the value.
 *
 * TODO: return what if not found?
 */
double getDoubleByXpath(string xpath) {
    return -1.0;
}

/**
 * Interface method to pull a float value from the xml 
 * schema. The calling object must know the xpath to retrieve 
 * the value.
 *
 * TODO: return what if not found?
 */
float getFloatByXpath(string xpath) {
    return -1.0;
}
