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
void AllIFNeurons::createNeuronsProps()
{
    m_pNeuronsProps = new AllIFNeuronsProps();
}

