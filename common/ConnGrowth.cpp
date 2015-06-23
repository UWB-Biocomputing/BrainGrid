#include "ConnGrowth.h"
#include "ParseParamError.h"
#include "AllSynapses.h"

/* ------------- CONNECTIONS STRUCT ------------ *\
 * Below all of the resources for the various
 * connections are instantiated and initialized.
 * All of the allocation for memory is done in the
 * constructor’s parameters and not in the body of
 * the function. Once all memory has been allocated
 * the constructor fills in known information
 * into “radii” and “rates”.
\* --------------------------------------------- */
// TODO comment
/* ------------------- ERROR ------------------- *\
 * terminate called after throwing an instance of 'std::bad_alloc'
 *      what():  St9bad_alloc
 * ------------------- CAUSE ------------------- *|
 * As simulations expand in size the number of
 * neurons in total increases exponentially. When
 * using a MATRIX_TYPE = “complete” the amount of
 * used memory increases by another order of magnitude.
 * Once enough memory is used no more memory can be
 * allocated and a “bsd_alloc” will be thrown.
 * The following members of the connection constructor
 * consume equally vast amounts of memory as the
 * simulation sizes grow:
 *      - W             - radii
 *      - rates         - dist2
 *      - delta         - dist
 *      - areai
 * ----------------- 1/25/14 ------------------- *|
 * Currently when running a simulation of sizes
 * equal to or greater than 100 * 100 the above
 * error is thrown. After some testing we have
 * determined that this is a hardware dependent
 * issue, not software. We are also looking into
 * switching matrix types from "complete" to
 * "sparce". If successful it is possible the
 * problematic matricies mentioned above will use
 * only 1/250 of their current space.
\* --------------------------------------------- */
ConnGrowth::ConnGrowth() : Connections()
{
    W = NULL;
    radii = NULL;
    rates = NULL;
    delta = NULL;
    area = NULL;
    outgrowth = NULL;
    deltaR = NULL;
}

ConnGrowth::~ConnGrowth()
{
    cleanupConnections();
}

void ConnGrowth::setupConnections(const SimulationInfo *sim_info)
{
    Connections::setupConnections(sim_info);

    int num_neurons = sim_info->totalNeurons;

    W = new CompleteMatrix(MATRIX_TYPE, MATRIX_INIT, num_neurons, num_neurons, 0);
    radii = new VectorMatrix(MATRIX_TYPE, MATRIX_INIT, 1, num_neurons, m_growth.startRadius);
    rates = new VectorMatrix(MATRIX_TYPE, MATRIX_INIT, 1, num_neurons, 0);
    delta = new CompleteMatrix(MATRIX_TYPE, MATRIX_INIT, num_neurons, num_neurons);
    area = new CompleteMatrix(MATRIX_TYPE, MATRIX_INIT, num_neurons, num_neurons, 0);
    outgrowth = new VectorMatrix(MATRIX_TYPE, MATRIX_INIT, 1, num_neurons);
    deltaR = new VectorMatrix(MATRIX_TYPE, MATRIX_INIT, 1, num_neurons);

    // Init connection frontier distance change matrix with the current distances
    (*delta) = (*dist);
}

void ConnGrowth::cleanupConnections()
{
    if (W != NULL) delete W;
    if (radii != NULL) delete radii;
    if (rates != NULL) delete rates;
    if (delta != NULL) delete delta;
    if (area != NULL) delete area;
    if (outgrowth != NULL) delete outgrowth;
    if (deltaR != NULL) delete deltaR;

    W = NULL;
    radii = NULL;
    rates = NULL;
    delta = NULL;
    area = NULL;
    outgrowth = NULL;
    deltaR = NULL;

    Connections::cleanupConnections();
}

/**
 *  Attempts to read parameters from a XML file.
 *  @param  @param  element TiXmlElement to examine.
 *  @return true if successful, false otherwise.
 */
bool ConnGrowth::readParameters(const TiXmlElement& element)
{
    Connections::readParameters(element);

    if (element.ValueStr().compare("GrowthParams") == 0) {
        if (element.QueryFLOATAttribute("epsilon", &m_growth.epsilon) != TIXML_SUCCESS) {
                throw ParseParamError("epsilon", "Growth param 'epsilon' missing in XML.");
        }
        if (m_growth.epsilon < 0) {
                throw ParseParamError("epsilon", "Invalid negative Growth param 'epsilon' value.");
        }

        if (element.QueryFLOATAttribute("beta", &m_growth.beta) != TIXML_SUCCESS) {
                throw ParseParamError("beta", "Growth param 'beta' missing in XML.");
        }
        if (m_growth.beta < 0) {
                throw ParseParamError("beta", "Invalid negative Growth param 'beta' value.");
        }

        if (element.QueryFLOATAttribute("rho", &m_growth.rho) != TIXML_SUCCESS) {
                throw ParseParamError("rho", "Growth param 'rho' missing in XML.");
        }
        if (m_growth.rho < 0) {
                throw ParseParamError("rho", "Invalid negative Growth param 'rho' value.");
        }

        //check if 'beta' is erroneous info
        if (element.QueryFLOATAttribute("targetRate", &m_growth.targetRate) != TIXML_SUCCESS) {
                throw ParseParamError("targetRate", "Growth targetRate 'beta' missing in XML.");
        }
        if (m_growth.targetRate < 0) {
                throw ParseParamError("targetRate", "Invalid negative Growth targetRate.");
        }

        if (element.QueryFLOATAttribute("minRadius", &m_growth.minRadius) != TIXML_SUCCESS) {
                throw ParseParamError("minRadius", "Growth minRadius 'beta' missing in XML.");
        }
        if (m_growth.minRadius < 0) {
                throw ParseParamError("minRadius", "Invalid negative Growth minRadius.");
        }

        if (element.QueryFLOATAttribute("startRadius", &m_growth.startRadius) != TIXML_SUCCESS) {
                throw ParseParamError("startRadius", "Growth startRadius 'beta' missing in XML.");
        }
        if (m_growth.startRadius < 0) {
                throw ParseParamError("startRadius", "Invalid negative Growth startRadius.");
        }
    }

    // initial maximum firing rate
    m_growth.maxRate = m_growth.targetRate / m_growth.epsilon;
        
    return true;
}

/**
 *  Prints out all parameters of the connections to ostream.
 *  @param  output  ostream to send output to.
 */
void ConnGrowth::printParameters(ostream &output) const
{
    Connections::printParameters(output);

    output << "Growth parameters: " << endl
           << "\tepsilon: " << m_growth.epsilon
           << ", beta: " << m_growth.beta
           << ", rho: " << m_growth.rho
           << ", targetRate: " << m_growth.targetRate << "," << endl
           << "\tminRadius: " << m_growth.minRadius
           << ", startRadius: " << m_growth.startRadius
           << endl;

}

void ConnGrowth::readConns(istream& input, const SimulationInfo *sim_info)
{
    Connections::readConns(input, sim_info);

    // read the radii
    for (int i = 0; i < sim_info->totalNeurons; i++) {
            input >> (*radii)[i]; input.ignore();
    }

    // read the rates
    for (int i = 0; i < sim_info->totalNeurons; i++) {
            input >> (*rates)[i]; input.ignore();
    }
}

void ConnGrowth::writeConns(ostream& output, const SimulationInfo *sim_info)
{
    Connections::writeConns(output, sim_info);

    // write the final radii
    for (int i = 0; i < sim_info->totalNeurons; i++) {
        output << (*radii)[i] << ends;
    }

    // write the final rates
    for (int i = 0; i < sim_info->totalNeurons; i++) {
        output << (*rates)[i] << ends;
    }
}

bool ConnGrowth::updateConnections(AllNeurons &neurons, const SimulationInfo *sim_info)
{
    // Update Connections data
    updateConns(neurons, sim_info);
 
    // Update the distance between frontiers of Neurons
    updateFrontiers(sim_info->totalNeurons);

    // Update the areas of overlap in between Neurons
    updateOverlap(sim_info->totalNeurons);

    return true;
}


void ConnGrowth::updateConns(AllNeurons &neurons, const SimulationInfo *sim_info)
{
    AllSpikingNeurons &spNeurons = dynamic_cast<AllSpikingNeurons&>(neurons);

    // Calculate growth cycle firing rate for previous period
    int max_spikes = static_cast<int> (sim_info->epochDuration * sim_info->maxFiringRate);
    for (int i = 0; i < sim_info->totalNeurons; i++) {
        // Calculate firing rate
        assert(spNeurons.spikeCount[i] < max_spikes);
        (*rates)[i] = spNeurons.spikeCount[i] / sim_info->epochDuration;
    }

    // compute neuron radii change and assign new values
    (*outgrowth) = 1.0 - 2.0 / (1.0 + exp((m_growth.epsilon - *rates / m_growth.maxRate) / m_growth.beta));
    (*deltaR) = sim_info->epochDuration * m_growth.rho * *outgrowth;
    (*radii) += (*deltaR);
}

/**
 *  Update the distance between frontiers of Neurons.
 *  @param  num_neurons in the simulation to update.
 */
void ConnGrowth::updateFrontiers(const int num_neurons)
{
    DEBUG(cout << "Updating distance between frontiers..." << endl;)
    // Update distance between frontiers
    for (int unit = 0; unit < num_neurons - 1; unit++) {
        for (int i = unit + 1; i < num_neurons; i++) {
            (*delta)(unit, i) = (*dist)(unit, i) - ((*radii)[unit] + (*radii)[i]);
            (*delta)(i, unit) = (*delta)(unit, i);
        }
    }
}

/**
 *  Update the areas of overlap in between Neurons.
 *  @param  num_neurons number of Neurons to update.
 */
void ConnGrowth::updateOverlap(BGFLOAT num_neurons)
{
    DEBUG(cout << "computing areas of overlap" << endl;)

    // Compute areas of overlap; this is only done for overlapping units
    for (int i = 0; i < num_neurons; i++) {
        for (int j = 0; j < num_neurons; j++) {
                (*area)(i, j) = 0.0;

                if ((*delta)(i, j) < 0) {
                        BGFLOAT lenAB = (*dist)(i, j);
                        BGFLOAT r1 = (*radii)[i];
                        BGFLOAT r2 = (*radii)[j];

                    if (lenAB + min(r1, r2) <= max(r1, r2)) {
                        (*area)(i, j) = pi * min(r1, r2) * min(r1, r2); // Completely overlapping unit
#ifdef LOGFILE
                        logFile << "Completely overlapping (i, j, r1, r2, area): "
                            << i << ", " << j << ", " << r1 << ", " << r2 << ", " << *pAarea(i, j) << endl;
#endif // LOGFILE
                        } else {
                                // Partially overlapping unit
                                BGFLOAT lenAB2 = (*dist2)(i, j);
                                BGFLOAT r12 = r1 * r1;
                                BGFLOAT r22 = r2 * r2;

                                BGFLOAT cosCBA = (r22 + lenAB2 - r12) / (2.0 * r2 * lenAB);
                                BGFLOAT angCBA = acos(cosCBA);
                                BGFLOAT angCBD = 2.0 * angCBA;

                                BGFLOAT cosCAB = (r12 + lenAB2 - r22) / (2.0 * r1 * lenAB);
                                BGFLOAT angCAB = acos(cosCAB);
                                BGFLOAT angCAD = 2.0 * angCAB;

                                (*area)(i, j) = 0.5 * (r22 * (angCBD - sin(angCBD)) + r12 * (angCAD - sin(angCAD)));
                        }
                }
        }
    }
}

#if !defined(USE_GPU)
/**
 *  Update the weight of the Synapses in the simulation.
 *  Note: Platform Dependent.
 *  @param  num_neurons number of neurons to update.
 *  @param  neurons the Neuron list to search from.
 *  @param  synapses    the Synapse list to search from.
 *  @param  sim_info    SimulationInfo to refer from.
 */
void ConnGrowth::updateSynapsesWeights(const int num_neurons, AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo *sim_info)
{

    // For now, we just set the weights to equal the areas. We will later
    // scale it and set its sign (when we index and get its sign).
    (*W) = (*area);

    int adjusted = 0;
    int could_have_been_removed = 0; // TODO: use this value
    int removed = 0;
    int added = 0;

    DEBUG(cout << "adjusting weights" << endl;)

    // Scale and add sign to the areas
    // visit each neuron 'a'
    for (int src_neuron = 0; src_neuron < num_neurons; src_neuron++) {
        int xa = src_neuron % sim_info->width;
        int ya = src_neuron / sim_info->width;
        Coordinate src_coord(xa, ya);

        // and each destination neuron 'b'
        for (int dest_neuron = 0; dest_neuron < num_neurons; dest_neuron++) {
            int xb = dest_neuron % sim_info->width;
            int yb = dest_neuron / sim_info->width;
            Coordinate dest_coord(xb, yb);

            // visit each synapse at (xa,ya)
            bool connected = false;
            synapseType type = neurons.synType(src_neuron, dest_neuron);

            // for each existing synapse
            size_t synapse_counts = synapses.synapse_counts[src_neuron];
            int synapse_adjusted = 0;
            for (size_t synapse_index = 0; synapse_adjusted < synapse_counts; synapse_index++) {
                uint32_t iSyn = synapses.maxSynapsesPerNeuron * src_neuron + synapse_index;
                if (synapses.in_use[iSyn] == true) {
                    // if there is a synapse between a and b
                    if (synapses.summationCoord[iSyn] == dest_coord) {
                        connected = true;
                        adjusted++;
                        // adjust the strength of the synapse or remove
                        // it from the synapse map if it has gone below
                        // zero.
                        if ((*W)(src_neuron, dest_neuron) < 0) {
                            removed++;
                            synapses.eraseSynapse(src_neuron, iSyn);
                        } else {
                            // adjust
                            // g_synapseStrengthAdjustmentConstant is 1.0e-8;
                            synapses.W[iSyn] = (*W)(src_neuron, dest_neuron) *
                                synapses.synSign(type) * AllSynapses::SYNAPSE_STRENGTH_ADJUSTMENT;

                            DEBUG_MID(cout << "weight of rgSynapseMap" <<
                                   coordToString(xa, ya)<<"[" <<synapse_index<<"]: " <<
                                   synapses.W[iSyn] << endl;);
                        }
                    }
                    synapse_adjusted++;
                }
            }

            // if not connected and weight(a,b) > 0, add a new synapse from a to b
            if (!connected && ((*W)(src_neuron, dest_neuron) > 0)) {

                // locate summation point
                BGFLOAT* sum_point = &( neurons.summation_map[dest_neuron] );
                added++;

                BGFLOAT weight = (*W)(src_neuron, dest_neuron) * synapses.synSign(type) * AllSynapses::SYNAPSE_STRENGTH_ADJUSTMENT;
                synapses.addSynapse(weight, type, src_neuron, dest_neuron, src_coord, dest_coord, sum_point, sim_info->deltaT);

            }
        }
    }

    DEBUG (cout << "adjusted: " << adjusted << endl;)
    DEBUG (cout << "could have been removed (TODO: calculate this): " << could_have_been_removed << endl;)
    DEBUG (cout << "removed: " << removed << endl;)
    DEBUG (cout << "added: " << added << endl << endl << endl;)
}
#endif // !USE_GPU
