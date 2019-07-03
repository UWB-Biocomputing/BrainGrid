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

// TODO: are these declarations necessary?
#include <iostream>
#include <string>

using namespace std;

class ParameterManager : TinyXmlDoc {
    public:
        // Utility methods
        ParameterManager();
        ~ParameterManager();
        bool loadParameterFile(string path);
        // Interface methods for simulator objects
        string getStringByXpath(string xpath);
        int getIntByXpath(string xpath);
        double getDoubleByXpath(string xpath);
        float getFloatByXpath(string xpath);
    private:

};
