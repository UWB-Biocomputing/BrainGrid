#include "AllIFNeurons.h"

// Default constructor
AllIFNeurons::AllIFNeurons()
{
}

AllIFNeurons::~AllIFNeurons()
{
}

/*
 *  Create and setup neurons properties.
 */
void AllIFNeurons::setupNeuronsProps()
{
    m_pNeuronsProperties = new AllIFNeuronsProperties();
}

