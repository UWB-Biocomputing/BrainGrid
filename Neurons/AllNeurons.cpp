#include "AllNeurons.h"
// Default constructor
AllNeurons::AllNeurons() : 
        nParams(0)
{
}

// Copy constructor
AllNeurons::AllNeurons(const AllNeurons &r_neurons) :
        nParams(0)
{
    copyParameters(dynamic_cast<const AllNeurons &>(r_neurons));
}

AllNeurons::~AllNeurons()
{
    cleanupNeurons();
}

/*
 *  Assignment operator: copy neurons parameters.
 *
 *  @param  r_neurons  Neurons class object to copy from.
 */
IAllNeurons &AllNeurons::operator=(const IAllNeurons &r_neurons)
{
    copyParameters(dynamic_cast<const AllNeurons &>(r_neurons));

    return (*this);
}

/*
 *  Copy neurons parameters.
 *
 *  @param  r_neurons  Neurons class object to copy from.
 */
void AllNeurons::copyParameters(const AllNeurons &r_neurons)
{
}

/*
 *  Setup the internal structure of the class (allocate memories).
 *
 *  @param  sim_info  SimulationInfo class to read information from.
 *  @param  clr_info  ClusterInfo class to read information from.
 */
void AllNeurons::setupNeurons(SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    setupNeuronsInternalState(sim_info, clr_info);

    // allocate neurons properties data
    m_pNeuronsProperties = new AllNeuronsProperties();
    m_pNeuronsProperties->setupNeuronsProperties(sim_info, clr_info);
}

/*
 *  Setup the internal structure of the class.
 *
 *  @param  sim_info  SimulationInfo class to read information from.
 *  @param  clr_info  ClusterInfo class to read information from.
 */
void AllNeurons::setupNeuronsInternalState(SimulationInfo *sim_info, ClusterInfo *clr_info)
{
}

/*
 *  Cleanup the class (deallocate memories).
 */
void AllNeurons::cleanupNeurons()
{
    // deallocate neurons properties data
    delete m_pNeuronsProperties;
    m_pNeuronsProperties = NULL;

    cleanupNeuronsInternalState();
}

/*
 *  Deallocate all resources
 */
void AllNeurons::cleanupNeuronsInternalState()
{
}
