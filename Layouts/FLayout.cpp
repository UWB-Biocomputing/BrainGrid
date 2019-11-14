/**
 *  A factory class for creating Layout objects.
 *  Lizzy Presland, October 2019
 */

#include "FLayout.h"
#include "FixedLayout.h"
#include "DynamicLayout.h"

// constructor
FLayout::FLayout() {
    // register layout classes    
    registerLayout("FixedLayout", &FixedLayout::Create);
    registerLayout("DynamicLayout", &DynamicLayout::Create);
}

FLayout::~FLayout() {
    createFunctions.clear();
}

/*
 *  Register layout class and its creation function to the factory.
 *
 *  @param  layoutClassName  layout class name.
 *  @param  Pointer to the class creation function.
 */
void FLayout::registerLayout(const string &layoutClassName, CreateLayoutFn function) {
    createFunctions[layoutClassName] = function;
}

/**
 * Creates concrete instance of the desired layout class.
 */
Layout* FLayout::createLayout(const string& className) {
    layoutInstance = invokeLayoutCreateFunction(className);
    // no createProps() call required
    return layoutInstance;
}

/**
 * Returns the static ::Create() method which allocates 
 * a new instance of the desired class.
 *
 * The calling method uses this retrieval mechanism in 
 * value assignment.
 */
Layout* FLayout::invokeLayoutCreateFunction(const string& className) {
    layoutClassName = className;
    LayoutFunctionMap::iterator it = createFunctions.find(layoutClassName);
    if (it != createFunctions.end()) return it->second();
    return NULL;
}
