#include "ConnStatic.h"
#include "ParseParamError.h"
#include "AllSynapses.h"
#include <algorithm>

ConnStatic::ConnStatic() : Connections()
{
    threshConnsRadius = 2.0;
    nConnsPerNeuron = 100;
    pRewiring = 0.03;
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
