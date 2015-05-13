#include "SingleThreadedSpikingModel.h"
#include "AllDSSynapses.h"

/*
*  Constructor
*/
SingleThreadedSpikingModel::SingleThreadedSpikingModel(Connections *conns, AllNeurons *neurons, AllSynapses *synapses, Layout *layout) : 
    Model(conns, neurons, synapses, layout)
{
}

/*
* Destructor
*/
SingleThreadedSpikingModel::~SingleThreadedSpikingModel() 
{
	//Let Model base class handle de-allocation
}

/**
 *  Advance everything in the model one time step. In this case, that
 *  means advancing just the Neurons and Synapses.
 *  @param  sim_info    SimulationInfo class to read information from.
 */
void SingleThreadedSpikingModel::advance(const SimulationInfo *sim_info)
{
    m_neurons->advanceNeurons(*m_synapses, sim_info);
    m_synapses->advanceSynapses(sim_info);
}

/**
 *  Update the connection of all the Neurons and Synapses of the simulation.
 *  @param  currentStep the current step of the simulation.
 *  @param  sim_info    SimulationInfo class to read information from.
 *  @param  simRecorder Pointer to the simulation recordig object.
 */
void SingleThreadedSpikingModel::updateConnections(const int currentStep, const SimulationInfo *sim_info, IRecorder* simRecorder)
{
    const int num_neurons = sim_info->totalNeurons;
    updateHistory(currentStep, sim_info->epochDuration, sim_info, simRecorder);
    // Update the distance between frontiers of Neurons
    m_conns->updateFrontiers(num_neurons);
    // Update the areas of overlap in between Neurons
    m_conns->updateOverlap(num_neurons);
    updateWeights(num_neurons, *m_neurons, *m_synapses, sim_info);
}

/* -----------------
* # Helper Functions
* ------------------
*/


/**
 *  Update the weight of the Synapses in the simulation.
 *  Note: Platform Dependent.
 *  @param  num_neurons number of neurons to update.
 *  @param  neurons the Neuron list to search from.
 *  @param  synapses    the Synapse list to search from.
 *  @param  sim_info    SimulationInfo to refer from.
 */
void SingleThreadedSpikingModel::updateWeights(const int num_neurons, AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo *sim_info)
{

    // For now, we just set the weights to equal the areas. We will later
    // scale it and set its sign (when we index and get its sign).
    (*m_conns->W) = (*m_conns->area);

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
                        if ((*m_conns->W)(src_neuron, dest_neuron) < 0) {
                            removed++;
                            synapses.eraseSynapse(src_neuron, iSyn);
                        } else {
                            // adjust
                            // g_synapseStrengthAdjustmentConstant is 1.0e-8;
                            synapses.W[iSyn] = (*m_conns->W)(src_neuron, dest_neuron) *
                                synSign(type) * AllDSSynapses::SYNAPSE_STRENGTH_ADJUSTMENT;

                            DEBUG_MID(cout << "weight of rgSynapseMap" <<
                                   coordToString(xa, ya)<<"[" <<synapse_index<<"]: " <<
                                   synapses.W[iSyn] << endl;);
                        }
                    }
                    synapse_adjusted++;
                }
            }

            // if not connected and weight(a,b) > 0, add a new synapse from a to b
            if (!connected && ((*m_conns->W)(src_neuron, dest_neuron) > 0)) {

                // locate summation point
                BGFLOAT* sum_point = &( neurons.summation_map[dest_neuron] );
                added++;

                BGFLOAT weight = (*m_conns->W)(src_neuron, dest_neuron) * synSign(type) * AllDSSynapses::SYNAPSE_STRENGTH_ADJUSTMENT;
                synapses.addSynapse(weight, type, src_neuron, dest_neuron, src_coord, dest_coord, sum_point, sim_info->deltaT);

            }
        }
    }

    DEBUG (cout << "adjusted: " << adjusted << endl;)
    DEBUG (cout << "could have been removed (TODO: calculate this): " << could_have_been_removed << endl;)
    DEBUG (cout << "removed: " << removed << endl;)
    DEBUG (cout << "added: " << added << endl << endl << endl;)
}

/**
 *  Get the sign of the synapseType.
 *  @param    type    synapseType I to I, I to E, E to I, or E to E
 *  @return   1 or -1, or 0 if error
 */
int SingleThreadedSpikingModel::synSign(const synapseType type)
{
    switch ( type ) {
        case II:
        case IE:
            return -1;
        case EI:
        case EE:
            return 1;
        case STYPE_UNDEF:
            // TODO error.
            return 0;
    }

    return 0;
}
