#include "ConnStatic.h"
#include "ParseParamError.h"
#include "AllSynapses.h"
#include "XmlRecorder.h"
#ifdef USE_HDF5
#include "Hdf5Recorder.h"
#endif
#include <algorithm>

ConnStatic::ConnStatic() : Connections()
{
    threshConnsRadius = 0;
    nConnsPerNeuron = 0;
    pRewiring = 0;
}

ConnStatic::~ConnStatic()
{
    cleanupConnections();
}

void ConnStatic::setupConnections(const SimulationInfo *sim_info, Layout *layout, AllNeurons *neurons, AllSynapses *synapses)
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
                if (dist <= threshConnsRadius) {
                    DistDestNeuron distDestNeuron;
                    distDestNeuron.dist = dist;
                    distDestNeuron.dest_neuron = dest_neuron;
                    distDestNeurons[src_neuron].push_back(distDestNeuron);
                }
            }
        }

        // sort ascendant
        sort(distDestNeurons[src_neuron].begin(), distDestNeurons[src_neuron].end());

        // pick the shortest nConnsPerNeuron connections
        for (int i = 0; i < distDestNeurons[src_neuron].size() && i < nConnsPerNeuron; i++) {
            int dest_neuron = distDestNeurons[src_neuron][i].dest_neuron;
            synapseType type = layout->synType(src_neuron, dest_neuron);
            BGFLOAT* sum_point = &( neurons->summation_map[dest_neuron] );

            DEBUG_MID (cout << "source: " << src_neuron << " dest: " << dest_neuron << " dist: " << distDestNeurons[src_neuron][i].dist << endl;)

            uint32_t iSyn;
            synapses->addSynapse(iSyn, type, src_neuron, dest_neuron, sum_point, sim_info->deltaT);
            added++;
        }
    }

    int nRewiring = added * pRewiring;

    DEBUG(cout << "Rewiring connections: " << nRewiring << endl;)

    DEBUG (cout << "added connections: " << added << endl << endl << endl;)
}

void ConnStatic::cleanupConnections()
{
}

/**
 *  Attempts to read parameters from a XML file.
 *  @param  @param  element TiXmlElement to examine.
 *  @return true if successful, false otherwise.
 */
bool ConnStatic::readParameters(const TiXmlElement& element)
{
    if (element.ValueStr().compare("ConnectionsParams") == 0) {
        // number of maximum connections per neurons
        if (element.QueryIntAttribute("nConnsPerNeuron", &nConnsPerNeuron) != TIXML_SUCCESS) {
                throw ParseParamError("nConnsPerNeuron", "Static Connections param 'nConnsPerNeuron' missing in XML.");
        }
        if (nConnsPerNeuron < 0) {
                throw ParseParamError("nConnsPerNeuron", "Invalid negative Growth param 'nConnsPerNeuron' value.");
        }

        // Connection radius threshold
        if (element.QueryFLOATAttribute("threshConnsRadius", &threshConnsRadius) != TIXML_SUCCESS) {
                throw ParseParamError("threshConnsRadius", "Static Connections param 'threshConnsRadius' missing in XML.");
        }
        if (threshConnsRadius < 0) {
                throw ParseParamError("threshConnsRadius", "Invalid negative Growth param 'threshConnsRadius' value.");
        }

        // Small-world rewiring probability
        if (element.QueryFLOATAttribute("pRewiring", &pRewiring) != TIXML_SUCCESS) {
                throw ParseParamError("pRewiring", "Static Connections param 'pRewiring' missing in XML.");
        }
        if (pRewiring < 0 || pRewiring > 1.0) {
                throw ParseParamError("pRewiring", "Invalid negative Growth param 'pRewiring' value.");
        }
    }

    return true;
}

/**
 *  Prints out all parameters of the connections to ostream.
 *  @param  output  ostream to send output to.
 */
void ConnStatic::printParameters(ostream &output) const
{
}

void ConnStatic::readConns(istream& input, const SimulationInfo *sim_info)
{
}

void ConnStatic::writeConns(ostream& output, const SimulationInfo *sim_info)
{
}

IRecorder* ConnStatic::createRecorder(const string &stateOutputFileName, IModel *model, const SimulationInfo *simInfo)
{
    // create & init simulation recorder
    IRecorder* simRecorder = NULL;
    if (stateOutputFileName.find(".xml") != string::npos) {
        simRecorder = new XmlRecorder(model, simInfo);
    }
#ifdef USE_HDF5
    else if (stateOutputFileName.find(".h5") != string::npos) {
        simRecorder = new Hdf5Recorder(model, simInfo);
    }
#endif // USE_HDF5
    else {
        return NULL;
    }
    if (simRecorder != NULL) {
        simRecorder->init(stateOutputFileName);
    }

    return simRecorder;
}
