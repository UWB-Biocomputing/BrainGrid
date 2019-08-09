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

#include "tinyxml.h"
#include <string>
#include "BGTypes.h"

using namespace std;

#ifndef PARAMETER_MANAGER_H__
#define PARAMETER_MANAGER_H__

class ParameterManager {
    public:
        // Utility methods
        ParameterManager();
        ~ParameterManager();
        bool loadParameterFile(string path);
        // Interface methods for simulator objects
        bool getStringByXpath(string xpath, string& result);
        bool getIntByXpath(string xpath, int& var);
        bool getDoubleByXpath(string xpath, double& var);
        bool getFloatByXpath(string xpath, float& var);
        bool getBGFloatByXpath(string xpath, BGFLOAT& var);
    private:
        TiXmlDocument* xmlDoc;
        TiXmlElement* root;
        bool checkDocumentStatus();
};

#endif          // PARAMETER_MANAGER_H__
