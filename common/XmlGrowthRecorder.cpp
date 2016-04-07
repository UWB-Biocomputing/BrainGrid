/*
 *      @file XmlGrowthRecorder.cpp
 *
 *      @brief An implementation for recording spikes history on xml file
 */
//! An implementation for recording spikes history on xml file

#include "XmlGrowthRecorder.h"
#include "AllIFNeurons.h"      // TODO: remove LIF model specific code
#include "ConnGrowth.h"

//! THe constructor and destructor
XmlGrowthRecorder::XmlGrowthRecorder(const SimulationInfo* sim_info) :
        XmlRecorder(sim_info),
        ratesHistory(MATRIX_TYPE, MATRIX_INIT, static_cast<int>(sim_info->maxSteps + 1), sim_info->totalNeurons),
        radiiHistory(MATRIX_TYPE, MATRIX_INIT, static_cast<int>(sim_info->maxSteps + 1), sim_info->totalNeurons)
{
}

XmlGrowthRecorder::~XmlGrowthRecorder()
{
}

/*
 * Init radii and rates history matrices with default values
 */
void XmlGrowthRecorder::initDefaultValues()
{
    Connections* pConn = m_model->getConnections();
    BGFLOAT startRadius = dynamic_cast<ConnGrowth*>(pConn)->m_growth.startRadius;

    for (int i = 0; i < m_sim_info->totalNeurons; i++)
    {
        radiiHistory(0, i) = startRadius;
        ratesHistory(0, i) = 0;
    }
}

/*
 * Init radii and rates history matrices with current radii and rates
 */
void XmlGrowthRecorder::initValues()
{
    Connections* pConn = m_model->getConnections();

    for (int i = 0; i < m_sim_info->totalNeurons; i++)
    {
        radiiHistory(0, i) = (*dynamic_cast<ConnGrowth*>(pConn)->radii)[i];
        ratesHistory(0, i) = (*dynamic_cast<ConnGrowth*>(pConn)->rates)[i];
    }
}

/*
 * Get the current radii and rates values
 */
void XmlGrowthRecorder::getValues()
{
    Connections* pConn = m_model->getConnections();

    for (int i = 0; i < m_sim_info->totalNeurons; i++)
    {
        (*dynamic_cast<ConnGrowth*>(pConn)->radii)[i] = radiiHistory(m_sim_info->currentStep, i);
        (*dynamic_cast<ConnGrowth*>(pConn)->rates)[i] = ratesHistory(m_sim_info->currentStep, i);
    }
}

/*
 * Compile history information in every epoch
 *
 * @param[in] neurons 	The entire list of neurons.
 */
void XmlGrowthRecorder::compileHistories(IAllNeurons &neurons)
{
    XmlRecorder::compileHistories(neurons);

    Connections* pConn = m_model->getConnections();

    BGFLOAT minRadius = dynamic_cast<ConnGrowth*>(pConn)->m_growth.minRadius;
    VectorMatrix& rates = (*dynamic_cast<ConnGrowth*>(pConn)->rates);
    VectorMatrix& radii = (*dynamic_cast<ConnGrowth*>(pConn)->radii);

    for (int iNeuron = 0; iNeuron < m_sim_info->totalNeurons; iNeuron++)
    {
        // record firing rate to history matrix
        ratesHistory(m_sim_info->currentStep, iNeuron) = rates[iNeuron];

        // Cap minimum radius size and record radii to history matrix
        // TODO: find out why we cap this here.
        if (radii[iNeuron] < minRadius)
            radii[iNeuron] = minRadius;

        // record radius to history matrix
        radiiHistory(m_sim_info->currentStep, iNeuron) = radii[iNeuron];

        DEBUG_MID(cout << "radii[" << iNeuron << ":" << radii[iNeuron] << "]" << endl;)
    }
}

/*
 * Writes simulation results to an output destination.
 *
 * @param  neurons the Neuron list to search from.
 **/
void XmlGrowthRecorder::saveSimData(const IAllNeurons &neurons)
{
    // create Neuron Types matrix
    VectorMatrix neuronTypes(MATRIX_TYPE, MATRIX_INIT, 1, m_sim_info->totalNeurons, EXC);
    for (int i = 0; i < m_sim_info->totalNeurons; i++) {
        neuronTypes[i] = m_model->getLayout()->neuron_type_map[i];
    }

    // create neuron threshold matrix
    VectorMatrix neuronThresh(MATRIX_TYPE, MATRIX_INIT, 1, m_sim_info->totalNeurons, 0);
    for (int i = 0; i < m_sim_info->totalNeurons; i++) {
        neuronThresh[i] = dynamic_cast<const AllIFNeurons&>(neurons).Vthresh[i];
    }

    // Write XML header information:
    stateOut << "<?xml version=\"1.0\" standalone=\"no\"?>\n" << "<!-- State output file for the DCT growth modeling-->\n";
    //stateOut << version; TODO: version

    // Write the core state information:
    stateOut << "<SimState>\n";
    stateOut << "   " << radiiHistory.toXML("radiiHistory") << endl;
    stateOut << "   " << ratesHistory.toXML("ratesHistory") << endl;
    stateOut << "   " << burstinessHist.toXML("burstinessHist") << endl;
    stateOut << "   " << spikesHistory.toXML("spikesHistory") << endl;
    stateOut << "   " << m_model->getLayout()->xloc->toXML("xloc") << endl;
    stateOut << "   " << m_model->getLayout()->yloc->toXML("yloc") << endl;
    stateOut << "   " << neuronTypes.toXML("neuronTypes") << endl;

    // create starter nuerons matrix
    int num_starter_neurons = static_cast<int>(m_model->getLayout()->num_endogenously_active_neurons);
    if (num_starter_neurons > 0)
    {
        VectorMatrix starterNeurons(MATRIX_TYPE, MATRIX_INIT, 1, num_starter_neurons);
        getStarterNeuronMatrix(starterNeurons, m_model->getLayout()->starter_map, m_sim_info);
        stateOut << "   " << starterNeurons.toXML("starterNeurons") << endl;
    }

    // Write neuron thresold
    stateOut << "   " << neuronThresh.toXML("neuronThresh") << endl;

    // write time between growth cycles
    stateOut << "   <Matrix name=\"Tsim\" type=\"complete\" rows=\"1\" columns=\"1\" multiplier=\"1.0\">" << endl;
    stateOut << "   " << m_sim_info->epochDuration << endl;
    stateOut << "</Matrix>" << endl;

    // write simulation end time
    stateOut << "   <Matrix name=\"simulationEndTime\" type=\"complete\" rows=\"1\" columns=\"1\" multiplier=\"1.0\">" << endl;
    stateOut << "   " << g_simulationStep * m_sim_info->deltaT << endl;
    stateOut << "</Matrix>" << endl;
    stateOut << "</SimState>" << endl;
}

