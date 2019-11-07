/**
 *  A factory class for creating Neuron objects.
 *  Lizzy Presland, October 2019
 */

#pragma once

#include <map>
#include <string>
#include "Global.h"

using namespace std;

class FLayout {

    public:
        FLayout();
        ~FLayout();

        static FLayout *get() {
            static FLayout instance;
            return &instance;
        }

        // Invokes constructor for desired concrete class
        Layout* createLayout(const string& className);
        // No copy method required for this class at this time.
        
    private:
        // Pointer to layout instance
        Layout* layoutInstance;
        string layoutClassName;
        /* Type definitions */
        // Defines function type for usage in internal map
        typedef Layout* (*CreateLayoutFn)(void);
        // Defines map between class name and corresponding ::Create() function.
        typedef map<string, CreateLayoutFn> LayoutFunctionMap;
        // Makes class-to-function map an internal factory member.
        LayoutFunctionMap createFunctions;
        // Retrieves and invokes correct ::Create() function
        Layout* invokeLayoutCreateFunction(const string& className);
        void registerLayout(const string& layoutClassName, CreateLayoutFn* function);
};
