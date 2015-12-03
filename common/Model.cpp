#include "Model.h"

#include "../tinyxml/tinyxml.h"

#include "ParseParamError.h"
#include "Util.h"
#include "ConnGrowth.h"


/**
 *  Constructor
 */
Model::Model(Connections *conns, AllNeurons *neurons, AllSynapses *synapses, Layout *layout) :
    m_read_params(0),
    m_conns(conns),
    m_neurons(neurons),
    m_synapses(synapses),
    m_layout(layout),
    m_synapseIndexMap(NULL)
{
}

/**
 *  Destructor
 */
Model::~Model()
{
    if (m_conns != NULL) {
        delete m_conns;
        m_conns = NULL;
    }

    if (m_neurons != NULL) {
        delete m_neurons;
        m_neurons = NULL;
    }

    if (m_synapses != NULL) {
        delete m_synapses;
        m_synapses = NULL;
    }

    if (m_layout != NULL) {
        delete m_layout;
        m_layout = NULL;
    }

    if (m_synapseIndexMap != NULL) {
        delete m_synapseIndexMap;
        m_synapseIndexMap = NULL;
    }
}

/**
 *  Loads the simulation based on istream input.
 *
 *  @param  input       istream to read from.
 *  @param  sim_info    used as a reference to set info for neurons and synapses.
 */
void Model::loadMemory(istream& input, const SimulationInfo *sim_info)
{
    // read the neurons data & create neurons
    m_neurons->readNeurons(input, sim_info);

    // read the synapse data & create synapses
    m_synapses->readSynapses(input, *m_neurons, sim_info);

    // read the connections data
    m_conns->readConns(input, sim_info);

    // create a synapse index map 
    createSynapseImap(*m_synapses, sim_info);
}

/**
 *  Write the simulation's memory image.
 *
 *  @param  output          The filestream to write.
 *  @param  simulation_step The step of the simulation at the current time.
 */
void Model::saveMemory(ostream& output, const SimulationInfo *sim_info)
{
    // write the neurons data
    output << sim_info->totalNeurons << ends;
    m_neurons->writeNeurons(output, sim_info);

    // write the synapse data
    m_synapses->writeSynapses(output, sim_info);

    // write the connections data
    m_conns->writeConns(output, sim_info);

    output << flush;
}

/**
 *  Save current simulation state.
 *
 *  @param simRecorder    Pointer to the simulation recordig object.
 *
 */
void Model::saveState(IRecorder* simRecorder)
{
    if (simRecorder != NULL) {
        simRecorder->saveSimState(*m_neurons);
    }
}

/**
 *  Creates all the Neurons and generates data for them.
 *
 *  @param  sim_info    SimulationInfo class to read information from.
 */
void Model::createAllNeurons(SimulationInfo *sim_info)
{
    DEBUG(cout << "\nAllocating neurons..." << endl;)

    // init neuron's map with layout
    m_layout->generateNeuronTypeMap(sim_info->totalNeurons);
    m_layout->initStarterMap(sim_info->totalNeurons);

    // set their specific types
    m_neurons->createAllNeurons(sim_info, m_layout);

    DEBUG(cout << "Done initializing neurons..." << endl;)
}

/**
 *  Sets up the Simulation.
 *
 *  @param  sim_info    SimulationInfo class to read information from.
 *  @param  simRecorder Pointer to the simulation recordig object.
 */
void Model::setupSim(SimulationInfo *sim_info, IRecorder* simRecorder)
{
    m_neurons->setupNeurons(sim_info);
    m_synapses->setupSynapses(sim_info);
    m_layout->setupLayout(sim_info);

    // Init radii and rates history matrices with default values
    if (simRecorder != NULL) {
        simRecorder->initDefaultValues();
    }

    // Creates all the Neurons and generates data for them.
    createAllNeurons(sim_info);

    m_conns->setupConnections(sim_info, m_layout, m_neurons, m_synapses);

    // create a synapse index map 
    createSynapseImap(*m_synapses, sim_info);
}

/**
 *  Clean up the simulation.
 *
 *  @param  sim_info    SimulationInfo to refer.
 */
void Model::cleanupSim(SimulationInfo *sim_info)
{
    m_neurons->cleanupNeurons();
    m_synapses->cleanupSynapses();
    m_conns->cleanupConnections();
}

/**
 *  Log this simulation step.
 *
 *  @param  sim_info    SimulationInfo to reference.
 */
void Model::logSimStep(const SimulationInfo *sim_info) const
{
    ConnGrowth* pConnGrowth = dynamic_cast<ConnGrowth*>(m_conns);
    if (pConnGrowth == NULL)
        return;

    cout << "format:\ntype,radius,firing rate" << endl;

    for (int y = 0; y < sim_info->height; y++) {
        stringstream ss;
        ss << fixed;
        ss.precision(1);

        for (int x = 0; x < sim_info->width; x++) {
            switch (m_layout->neuron_type_map[x + y * sim_info->width]) {
            case EXC:
                if (m_layout->starter_map[x + y * sim_info->width])
                    ss << "s";
                else
                    ss << "e";
                break;
            case INH:
                ss << "i";
                break;
            case NTYPE_UNDEF:
                assert(false);
                break;
            }

            ss << " " << (*pConnGrowth->radii)[x + y * sim_info->width];

            if (x + 1 < sim_info->width) {
                ss.width(2);
                ss << "|";
                ss.width(2);
            }
        }

        ss << endl;

        for (int i = ss.str().length() - 1; i >= 0; i--) {
            ss << "_";
        }

        ss << endl;
        cout << ss.str();
    }
}

/**
 *  Update the simulation history of every epoch.
 *
 *  @param  sim_info    SimulationInfo to refer from.
 *  @param  simRecorder	Pointer to the simulation recordig object.
 */
void Model::updateHistory(const SimulationInfo *sim_info, IRecorder* simRecorder)
{
    // Compile history information in every epoch
    if (simRecorder != NULL) {
        simRecorder->compileHistories(*m_neurons);
    }
}

/**
 *  Get the AllNeurons class object.
 *
 *  @return Pointer to the AllNeurons class object.
 */
AllNeurons* Model::getNeurons()
{
    return m_neurons;
}

/**
 *  Get the Connections class object.
 *
 *  @return Pointer to the Connections class object.
 */
Connections* Model::getConnections()
{
    return m_conns;
}

/**
 *  Get the Layout class object.
 *
 *  @return Pointer to the Layout class object.
 */
Layout* Model::getLayout()
{
    return m_layout;
}

/**
 *  Create a synapse index map on device memory.
 *
 *  @param  synapses     Reference to the AllSynapses struct on host memory.
 *  @param  sim_info     Pointer to the simulation information.
 */
void Model::createSynapseImap(AllSynapses &synapses, const SimulationInfo* sim_info )
{
        int neuron_count = sim_info->totalNeurons;
        int width = sim_info->width;
        int total_synapse_counts = 0;

        // count the total synapses
        for ( int i = 0; i < neuron_count; i++ )
        {
                assert( synapses.synapse_counts[i] < sim_info->maxSynapsesPerNeuron );
                total_synapse_counts += synapses.synapse_counts[i];
        }

        DEBUG ( cout << "total_synapse_counts: " << total_synapse_counts << endl; )

        if ( total_synapse_counts == 0 )
        {
                return;
        }

        // allocate memories for inverse map
        vector<uint32_t>* rgSynapseSynapseIndexMap = new vector<uint32_t>[neuron_count];

        uint32_t syn_i = 0;
        int n_inUse = 0;

        if (m_synapseIndexMap != NULL) 
        {
            delete m_synapseIndexMap;
            m_synapseIndexMap = NULL;
        }

        // create synapse inverse map
        m_synapseIndexMap = new SynapseIndexMap(neuron_count, total_synapse_counts);
        for (int i = 0; i < neuron_count; i++)
        {
                for ( int j = 0; j < sim_info->maxSynapsesPerNeuron; j++, syn_i++ )
                {
                        uint32_t iSyn = sim_info->maxSynapsesPerNeuron * i + j;
                        if ( synapses.in_use[iSyn] == true )
                        {
                                int idx = synapses.destNeuronIndex[iSyn];
                                rgSynapseSynapseIndexMap[idx].push_back(syn_i);

                                m_synapseIndexMap->activeSynapseIndex[n_inUse] = syn_i;
                                n_inUse++;
                        }
                }
        }

        assert( total_synapse_counts == n_inUse );
        synapses.total_synapse_counts = total_synapse_counts;

        syn_i = 0;
        for (int i = 0; i < neuron_count; i++)
        {
                m_synapseIndexMap->incomingSynapse_begin[i] = syn_i;
                m_synapseIndexMap->synapseCount[i] = rgSynapseSynapseIndexMap[i].size();

                for ( int j = 0; j < rgSynapseSynapseIndexMap[i].size(); j++, syn_i++)
                {
                        m_synapseIndexMap->inverseIndex[syn_i] = rgSynapseSynapseIndexMap[i][j];
                }
        }

        // delete memories
        delete[] rgSynapseSynapseIndexMap;
}

