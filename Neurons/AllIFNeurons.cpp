#include "AllIFNeurons.h"

// Default constructor
CUDA_CALLABLE AllIFNeurons::AllIFNeurons()
{
}

CUDA_CALLABLE AllIFNeurons::~AllIFNeurons()
{
}

/*
 *  Create and setup neurons properties.
 */
void AllIFNeurons::createNeuronsProps()
{
    m_pNeuronsProps = new AllIFNeuronsProps();
}

