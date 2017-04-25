#include "ConnStatic.h"
#include "ParseParamError.h"
#include "IAllSynapses.h"
#include "XmlRecorder.h"
#ifdef USE_HDF5
#include "Hdf5Recorder.h"
#endif
#include <algorithm>

ConnStatic::ConnStatic() : Connections()
{
    m_threshConnsRadius = 0;
    m_nConnsPerNeuron = 0;
    m_pRewiring = 0;
}

ConnStatic::~ConnStatic()
{
    cleanupConnections();
}

/*
 *  Setup the internal structure of the class (allocate memories and initialize them).
 *  Initialize the small world network characterized by parameters: 
 *  number of maximum connections per neurons, connection radius threshold, and
 *  small-world rewiring probability.
 *
 *  @param  sim_info  SimulationInfo class to read information from.
 *  @param  layout    Layout information of the neunal network.
 *  @param  neurons   The Neuron list to search from.
 *  @param  synapses  The Synapse list to search from.
 */
void ConnStatic::setupConnections(const SimulationInfo *sim_info, Layout *layout, IAllNeurons *neurons, IAllSynapses *synapses)
{
    int num_neurons = sim_info->totalNeurons;
    vector<DistDestNeuron> distDestNeurons[num_neurons];

    int added = 0;

    DEBUG(cout << "Initializing connections" << endl;)

    for (int src_neuron = 0; src_neuron < num_neurons; src_neuron++) {
        distDestNeurons[src_neuron].clear();

        // pick the connections shorter than threshConnsRadius
        for (int dest_neuron = 0; dest_neuron < num_neurons; dest_neuron++) {
            if (src_neuron != dest_neuron) {
                BGFLOAT dist = (*layout->dist)(src_neuron, dest_neuron);
                if (dist <= m_threshConnsRadius) {
                    DistDestNeuron distDestNeuron;
                    distDestNeuron.dist = dist;
                    distDestNeuron.dest_neuron = dest_neuron;
                    distDestNeurons[src_neuron].push_back(distDestNeuron);
                }
            }
        }

        // sort ascendant
        sort(distDestNeurons[src_neuron].begin(), distDestNeurons[src_neuron].end());
        // pick the shortest m_nConnsPerNeuron connections
        for (BGSIZE i = 0; i < distDestNeurons[src_neuron].size() && (int)i < m_nConnsPerNeuron; i++) {
            int dest_neuron = distDestNeurons[src_neuron][i].dest_neuron;
            synapseType type = layout->synType(src_neuron, dest_neuron);
            BGFLOAT* sum_point = &( dynamic_cast<AllNeurons*>(neurons)->summation_map[dest_neuron] );

            DEBUG_MID (cout << "source: " << src_neuron << " dest: " << dest_neuron << " dist: " << distDestNeurons[src_neuron][i].dist << endl;)

            BGSIZE iSyn;
            synapses->addSynapse(iSyn, type, src_neuron, dest_neuron, sum_point, sim_info->deltaT);
            added++;

            // set synapse weight
            // TODO: we need another synaptic weight distibution mode (normal distribution)
            if (synapses->synSign(type) > 0) {
                dynamic_cast<AllSynapses*>(synapses)->W[iSyn] = rng.inRange(m_excWeight[0], m_excWeight[1]);
            }
            else {
                dynamic_cast<AllSynapses*>(synapses)->W[iSyn] = rng.inRange(m_inhWeight[0], m_inhWeight[1]);
            } 
        }
    }

    int nRewiring = added * m_pRewiring;

    DEBUG(cout << "Rewiring connections: " << nRewiring << endl;)

    DEBUG (cout << "added connections: " << added << endl << endl << endl;)
}

/*
 *  Cleanup the class.
 */
void ConnStatic::cleanupConnections()
{
}

/*
 *  Checks the number of required parameters.
 *
 * @return true if all required parameters were successfully read, false otherwise.
 */
bool ConnStatic::checkNumParameters()
{
    return (nParams >= 2);
}

/*
 *  Attempts to read parameters from a XML file.
 *
 *  @param  element TiXmlElement to examine.
 *  @return true if successful, false otherwise.
 */
bool ConnStatic::readParameters(const TiXmlElement& element)
{
    if (element.ValueStr().compare("StaticConnectionsParams") == 0) {
	nParams++;
	return true;
    }
    if (element.ValueStr().compare("StaticConnectionsWeight") == 0) {
	nParams++;
	return true;
    }

    // Connections parameters
    if (element.Parent()->ValueStr().compare("StaticConnectionsParams") == 0) {
/*
        // number of maximum connections per neurons
        if (element.QueryIntAttribute("nConnsPerNeuron", &m_nConnsPerNeuron) != TIXML_SUCCESS) {
                throw ParseParamError("nConnsPerNeuron", "Static Connections param 'nConnsPerNeuron' missing in XML.");
        }
        if (m_nConnsPerNeuron < 0) {
                throw ParseParamError("nConnsPerNeuron", "Invalid negative Growth param 'nConnsPerNeuron' value.");
        }

        // Connection radius threshold
        if (element.QueryFLOATAttribute("threshConnsRadius", &m_threshConnsRadius) != TIXML_SUCCESS) {
                throw ParseParamError("threshConnsRadius", "Static Connections param 'threshConnsRadius' missing in XML.");
        }
        if (m_threshConnsRadius < 0) {
                throw ParseParamError("threshConnsRadius", "Invalid negative Growth param 'threshConnsRadius' value.");
        }

        // Small-world rewiring probability
        if (element.QueryFLOATAttribute("pRewiring", &m_pRewiring) != TIXML_SUCCESS) {
                throw ParseParamError("pRewiring", "Static Connections param 'pRewiring' missing in XML.");
        }
        if (m_pRewiring < 0 || m_pRewiring > 1.0) {
                throw ParseParamError("pRewiring", "Invalid negative Growth param 'pRewiring' value.");
        }
        nParams++;
*/
	if(element.ValueStr().compare("nConnsPerNeuron") == 0){
            m_nConnsPerNeuron = atoi(element.GetText());
        }
        else if(element.ValueStr().compare("threshConnsRadius") == 0){
            m_threshConnsRadius = atof(element.GetText());
        }
	else if(element.ValueStr().compare("pRewiring") == 0){
	    m_pRewiring = atof(element.GetText());
	}

        if (m_pRewiring < 0 || m_pRewiring > 1.0) {
                throw ParseParamError("pRewiring", "Invalid negative Growth param 'pRewiring' value.");
        }

	return true;
    }
		

    // Connections weight parameters
    if (element.Parent()->ValueStr().compare("StaticConnectionsWeight") == 0) {
/*
        if (element.QueryFLOATAttribute("minExc", &m_excWeight[0]) != TIXML_SUCCESS) {
            throw ParseParamError("ConnectionsWeight minExc", "ConnectionsWeight missing minimum values of excitatory neuron's synapse weight in XML.");
        }
        if (element.QueryFLOATAttribute("maxExc", &m_excWeight[1]) != TIXML_SUCCESS) {
            throw ParseParamError("ConnectionsWeight maxExc", "ConnectionsWeight missing maximum values of excitatory neuron's synapse weight in XML.");
        }
        if (element.QueryFLOATAttribute("minInh", &m_inhWeight[0]) != TIXML_SUCCESS) {
            throw ParseParamError("ConnectionsWeight minInh", "ConnectionsWeight missing minimum values of inhibitory neuron's synapse weight in XML.");
        }
        if (element.QueryFLOATAttribute("maxInh", &m_inhWeight[1]) != TIXML_SUCCESS) {
            throw ParseParamError("ConnectionsWeight maxInh", "ConnectionsWeight missing maximum values of inhibitory neuron's synapse weight in XML.");
        }
        if (m_excWeight[0] < 0 || m_excWeight[0] > m_excWeight[1]) {
            throw ParseParamError("ConnectionsWeight maxExc", "Invalid range for ConnectionsWeight excitatory neuron's synapse weight.");
        }
        if (m_inhWeight[1] > 0 || m_inhWeight[0] > m_inhWeight[1]) {
            throw ParseParamError("ConnectionsWeight maxInh", "Invalid range for ConnectionsWeight inhibitory neuron's synapse weight.");
        }
        nParams++;
*/
	if(element.ValueStr().compare("minExc") == 0){
            m_excWeight[0] = atof(element.GetText());
        }
        else if(element.ValueStr().compare("maxExc") == 0){
            m_excWeight[1] = atof(element.GetText());
        }
        else if(element.ValueStr().compare("minInh") == 0){
            m_inhWeight[0] = atof(element.GetText());
        }
        else if(element.ValueStr().compare("maxInh") == 0){
            m_inhWeight[1] = atof(element.GetText());
        }

        return true;
    }

    return false;
}

/*
 *  Prints out all parameters of the connections to ostream.
 *
 *  @param  output  ostream to send output to.
 */
void ConnStatic::printParameters(ostream &output) const
{
}

/*
 *  Reads the intermediate connection status from istream.
 *
 *  @param  input    istream to read status from.
 *  @param  sim_info SimulationInfo class to read information from.
 */
void ConnStatic::deserialize(istream& input, const SimulationInfo *sim_info)
{
}

/*
 *  Writes the intermediate connection status to ostream.
 *
 *  @param  output   ostream to write status to.
 *  @param  sim_info SimulationInfo class to read information from.
 */
void ConnStatic::serialize(ostream& output, const SimulationInfo *sim_info)
{
}

/*
 *  Creates a recorder class object for the connection.
 *  This function tries to create either Xml recorder or
 *  Hdf5 recorder based on the extension of the file name.
 *
 *  @param  simInfo              SimulationInfo to refer from.
 *  @return Pointer to the recorder class object.
 */
IRecorder* ConnStatic::createRecorder(const SimulationInfo *simInfo)
{
    // create & init simulation recorder
    IRecorder* simRecorder = NULL;
    if (simInfo->stateOutputFileName.find(".xml") != string::npos) {
        simRecorder = new XmlRecorder(simInfo);
    }
#ifdef USE_HDF5
    else if (simInfo->stateOutputFileName.find(".h5") != string::npos) {
        simRecorder = new Hdf5Recorder(simInfo);
    }
#endif // USE_HDF5
    else {
        return NULL;
    }
    if (simRecorder != NULL) {
        simRecorder->init(simInfo->stateOutputFileName);
    }

    return simRecorder;
}
