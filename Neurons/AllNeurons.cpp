#include "AllNeurons.h"

// Default constructor
AllNeurons::AllNeurons() 
{
}

AllNeurons::~AllNeurons()
{
}

/*
 *  Cleanup ihe class (deallocate memories).
 */
void AllNeurons::cleanupNeurons()
{
    // deallocate neurons properties data
    delete m_pNeuronsProperties;
    m_pNeuronsProperties = NULL;
}

/*
 *  Assignment operator: copy neurons parameters.
 *
 *  @param  r_neurons  Neurons class object to copy from.
 */
IAllNeurons &AllNeurons::operator=(const IAllNeurons &r_neurons)
{
    m_pNeuronsProperties->copyParameters(dynamic_cast<const AllNeurons&>(r_neurons).m_pNeuronsProperties);

    return (*this);
}

/*
 *  Setup the internal structure of the class (allocate memories).
 *
 *  @param  sim_info  SimulationInfo class to read information from.
 *  @param  clr_info  ClusterInfo class to read information from.
 */
void AllNeurons::setupNeurons(SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    // allocate neurons properties data
    m_pNeuronsProperties->setupNeuronsProperties(sim_info, clr_info);
}

/*
 * Checks the number of required parameters.
 *
 * @return true if all required parameters were successfully read, false otherwise.
 */
bool AllNeurons::checkNumParameters()
{
    return (m_pNeuronsProperties->checkNumParameters());
}

/*
 *  Attempts to read parameters from a XML file.
 *
 *  @param  element TiXmlElement to examine.
 *  @return true if successful, false otherwise.
 */
bool AllNeurons::readParameters(const TiXmlElement& element)
{
    return (m_pNeuronsProperties->readParameters(element));
}

/*
 *  Prints out all parameters of the neurons to ostream.
 *
 *  @param  output  ostream to send output to.
 */
void AllNeurons::printParameters(ostream &output) const
{
    m_pNeuronsProperties->printParameters(output);
}

/*
 *  Sets the data for Neurons to input's data.
 *
 *  @param  input       istream to read from.
 *  @param  clr_info    used as a reference to set info for neurons.
 */
void AllNeurons::deserialize(istream &input, const ClusterInfo *clr_info)
{
    for (int i = 0; i < clr_info->totalClusterNeurons; i++) {
        m_pNeuronsProperties->readNeuronProperties(input, i);
    }
}

/*
 *  Writes out the data in Neurons.
 *
 *  @param  output      stream to write out to.
 *  @param  clr_info    used as a reference to set info for neuronss.
 */
void AllNeurons::serialize(ostream& output, const ClusterInfo *clr_info) const
{
    for (int i = 0; i < clr_info->totalClusterNeurons; i++) {
        m_pNeuronsProperties->writeNeuronProperties(output, i);
    }
}

/*
 *  Creates all the Neurons and generates data for them.
 *
 *  @param  sim_info    SimulationInfo class to read information from.
 *  @param  layout      Layout information of the neunal network.
 *  @param  clr_info    ClusterInfo class to read information from.
 */
void AllNeurons::createAllNeurons(SimulationInfo *sim_info, Layout *layout, ClusterInfo *clr_info)
{
    /* set their specific types */
    for (int neuron_index = 0; neuron_index < clr_info->totalClusterNeurons; neuron_index++) {
        m_pNeuronsProperties->setNeuronPropertyDefaults(neuron_index);

        // set the neuron info for neurons
        m_pNeuronsProperties->setNeuronPropertyValues(sim_info, neuron_index, layout, clr_info);
    }
}

