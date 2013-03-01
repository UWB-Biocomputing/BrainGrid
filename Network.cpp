/**
 *  @file Network.cpp
 *
 *  @author Allan Ortiz & Cory Mayberry
 *
 *  @brief A grid of LIF Neurons and their interconnecting synapses.
 */
#include "Network.h"

const string Network::MATRIX_TYPE = "complete";
const string Network::MATRIX_INIT = "const";

/** 
 * The constructor for Network.
 * @post The network is setup according to parameters and ready for simulation.
 */
Network::Network(Model *model,
        FLOAT startFrac,
        ostream& new_stateout, istream& new_meminput, bool fReadMemImage,
        SimulationInfo simInfo, ISimulation* sim) :
    
    m_model(model);
    
    m_cStarterNeurons(static_cast<int>(simInfo.cNeurons * startFrac)), 
    m_rgSynapseMap(NULL),
    m_summationMap(NULL),
    m_rgNeuronTypeMap(NULL),
    m_targetRate(new_targetRate),
    
    state_out(new_stateout),
    memory_in(new_meminput),
    m_fReadMemImage(fReadMemImage),
    
    m_si(simInfo),
    m_sim(sim),  // =>ISIMULATION
    
    
    radii(MATRIX_TYPE, MATRIX_INIT, 1, simInfo.cNeurons),
    rates(MATRIX_TYPE, MATRIX_INIT, 1, simInfo.cNeurons),
    xloc(MATRIX_TYPE, MATRIX_INIT, 1, simInfo.cNeurons),
    yloc(MATRIX_TYPE, MATRIX_INIT, 1, simInfo.cNeurons)
{
    cout << "Neuron count: " << simInfo.cNeurons << endl;
 
    // init data structures
    reset();
    
    // Initialize parameters of all neurons.
    //initNeuronTypeMap(); // TODO(derek) : delete
    //initStarterMap(); // TODO(derek) : delete
    
    // init neurons
    // initNeurons(Iinject, Inoise, Vthresh, Vresting, Vreset, Vinit, starter_Vthresh, starter_Vreset); // TODO(derek) : delete
    m_model->createAllNeurons(m_si.cNeurons, neurons);
    
    // Initialize neuron locations
    for (int i = 0; i < m_si.cNeurons; i++) {
        xloc[i] = i % m_si.width;
        yloc[i] = i / m_si.width;
    }
}

/**
* Destructor
*
*/
Network::~Network()
{
    freeResources();
}

/**
 * Initialize and prepare network for simulation.
 *
 * @param growthStepDuration
 *
 * @param maxGrowthSteps
 */
void Network::setup(FLOAT growthStepDuration, FLOAT maxGrowthSteps)
{
    // track radii
    radiiHistory = CompleteMatrix(MATRIX_TYPE, MATRIX_INIT, static_cast<int>(maxGrowthSteps + 1), m_si.cNeurons); // state

    // track firing rate
    ratesHistory = CompleteMatrix(MATRIX_TYPE, MATRIX_INIT, static_cast<int>(maxGrowthSteps + 1), m_si.cNeurons);

    // Init radii and rates history matrices with current radii and rates
    for (int i = 0; i < m_si.cNeurons; i++) {
        radiiHistory(0, i) = m_si.startRadius;
        ratesHistory(0, i) = 0;
    }

    // Read a simulation memory image
    if (m_fReadMemImage) {
        readSimMemory(memory_in, radii, rates);
        for (int i = 0; i < m_si.cNeurons; i++) {
            radiiHistory(0, i) = radii[i]; // NOTE: Radii Used for read.
            ratesHistory(0, i) = rates[i]; // NOTE: Rates Used for read.
        }
    }
}

NetworkUpdater* Network::getUpdater() const
{
    return new NetworkUpdater(m_si.cNeurons, m_si.startRadius, xloc, yloc);
}

void Network::finish(FLOAT growthStepDuration, FLOAT maxGrowthSteps)
{
    saveSimState(state_out, growthStepDuration, FLOAT growthStepDuration, FLOAT maxGrowthSteps);

    // Terminate the simulator
    m_sim->term(&m_si); // Can #term be removed w/ the new model architecture?  // =>ISIMULATION
}

void Network::advance()
{
    m_model->advance(m_si.cNeurons, neurons, synapses);
}

void Network::getSpikeCounts( int neuron_count, int *spikeCounts)
{
    for (int i = 0; i < neuron_count; i++) {
        spikeCounts[i] = m_neuronList[i]->getSpikeCount();
    }
}

//! Clear spike count of each neuron.
void Network::clearSpikeCounts(int neuron_count)
{
    for (int i = 0; i < neuron_count; i++) {
         m_neuronList[i]->clearSpikeCount();
    }
}

/**
 * Print network radii to console
 * @param[in] psi	Pointer to the simulation information.
 * @param[in] networkRadii	Array to store neuron radii.
 */
void Network::printRadii(SimulationInfo* psi) const
{
    cout << "format:\ntype,radius,firing rate" << endl;

    for (int y = 0; y < psi->height; y++) {
        stringstream ss;
        ss << fixed;
        ss.precision(1);

        for (int x = 0; x < psi->width; x++) {
            switch (psi->rgNeuronTypeMap[x + y * psi->width]) {
                case EXC:
                    if (psi->rgEndogenouslyActiveNeuronMap[x + y * psi->width])
                        ss << "s";
                    else
                        ss << "e";
                    break;
                case INH:
                    ss << "i";
                    break;
                case NTYPE_UNDEF:
                    assert(false);
            }

            ss << " " << radii[x + y * psi->width];
            ss << " " << radii[x + y * psi->width];

            if (x + 1 < psi->width) {
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
* Clean up heap objects
*
*/
void Network::freeResources()
{
	// Free neuron and synapse maps
    if (m_rgSynapseMap != NULL) {
		for(int x = 0; x < m_si.cNeurons; x++) {
			delete m_neuronList[x];
			for(unsigned int y = 0; y < m_rgSynapseMap[x].size(); y++)
				delete m_rgSynapseMap[x][y];
		}
		delete[] m_rgSynapseMap;
	}

    if (m_rgNeuronTypeMap != NULL)
		delete[] m_rgNeuronTypeMap;
	
    if (m_summationMap != NULL) 
		delete[] m_summationMap;
}

/**
 * Resets all of the maps.
 * Releases and re-allocates memory for each map, clearing them as necessary.
 */
void Network::reset()
{
    DEBUG(cout << "\nEntering Network::reset()";)

    freeResources();

    // Reset global simulation Step to 0
    g_simulationStep = 0;

    // initial maximum firing rate
    m_si.maxRate = m_targetRate / m_si.epsilon;

    // allocate maps
    m_rgNeuronTypeMap = new neuronType[m_si.cNeurons];

    // Used to assign endogenously active neurons
    m_rgEndogenouslyActiveNeuronMap = new bool[m_si.cNeurons]; // MODEL DEPENDENT

    // NOTE - is an empty network a valid network?
    m_neuronList.clear();
    m_neuronList.resize(m_si.cNeurons);

    m_rgSynapseMap = new vector<ISynapse*>[m_si.cNeurons];

    m_summationMap = new FLOAT[m_si.cNeurons];

    // initialize maps
    for (int i = 0; i < m_si.cNeurons; i++)
    {
        m_rgEndogenouslyActiveNeuronMap[i] = false; // MODEL DEPENDENT
        m_summationMap[i] = 0;
    }

    m_si.pNeuronList = &m_neuronList;
    m_si.rgSynapseMap = m_rgSynapseMap;
    m_si.pSummationMap = m_summationMap;

    DEBUG(cout << "\nExiting Network::reset()";)
}

/**
* Save current simulation state to XML
*
* @param os
* @param radiiHistory
* @param ratesHistory
* @param xloc
* @param yloc
* @param neuronTypes
* @param burstinessHist
* @param spikesHistory
* @param Tsim
*/
void Network::saveSimState(ostream& os, FLOAT Tsim, FLOAT growthStepDuration, FLOAT maxGrowthSteps)
{
    // Write XML header information:
    os << "<?xml version=\"1.0\" standalone=\"no\"?>" << endl
       << "<!-- State output file for the DCT growth modeling-->" << endl;
    //os << version; TODO: version

    // Write the core state information:
    os << "<SimState>" << endl;
    os << "   " << radiiHistory.toXML("radiiHistory") << endl;
    os << "   " << ratesHistory.toXML("ratesHistory") << endl;
    
    // burstiness Histogram goes through the
    VectorMatrix burstinessHist(MATRIX_TYPE, MATRIX_INIT, 1, (int)(growthStepDuration * maxGrowthSteps), 0); // state
    // spikes history - history of accumulated spikes count of all neurons (10 ms bin)
    VectorMatrix spikesHistory(MATRIX_TYPE, MATRIX_INIT, 1, (int)(growthStepDuration * maxGrowthSteps), 0); // state
#ifdef STORE_SPIKEHISTORY
    get_spike_history(burstinessHist, spikesHistory);
#endif // STORE_SPIKEHISTORY
    os << "   " << burstinessHist.toXML("burstinessHist") << endl;
    os << "   " << spikesHistory.toXML("spikesHistory") << endl;
    
    os << "   " << xloc.toXML("xloc") << endl;
    os << "   " << yloc.toXML("yloc") << endl;
    
    //Write Neuron Types
    VectorMatrix neuronTypes(MATRIX_TYPE, MATRIX_INIT, 1, simInfo.cNeurons, EXC);
    getNeuronTypes(neuronTypes);
    os << "   " << neuronTypes.toXML("neuronTypes") << endl;

    if (m_cStarterNeurons > 0) {
        VectorMatrix starterNeuronsM("complete", "const", 1, m_cStarterNeurons);

        getStarterNeuronMatrix(starterNeuronsM);

        os << "   " << starterNeuronsM.toXML("starterNeurons") << endl;
    }

    // Write neuron thresold
    // neuron threshold
    VectorMatrix neuronThresh(MATRIX_TYPE, MATRIX_INIT, 1, m_si.cNeurons, 0);
    for (int i = 0; i < m_si.cNeurons; i++) {
        neuronThresh[i] = m_neuronList[i]->Vthresh;
    }
    os << "   " << neuronThresh.toXML("neuronThresh") << endl;

    // write time between growth cycles
    os << "   <Matrix name=\"Tsim\" type=\"complete\" rows=\"1\" columns=\"1\" multiplier=\"1.0\">" << endl;
    os << "   " << Tsim << endl;
    os << "</Matrix>" << endl;

    // write simulation end time
    os << "   <Matrix name=\"simulationEndTime\" type=\"complete\" rows=\"1\" columns=\"1\" multiplier=\"1.0\">" << endl;
    os << "   " << g_simulationStep * m_si.deltaT << endl;
    os << "</Matrix>" << endl;
    os << "</SimState>" << endl;
}

void Network::get_spike_history(VectorMatrix& burstinessHist, VectorMatrix& spikesHistory)
{
    // output spikes
    for (int i = 0; i < m_si.width; i++) {
        for (int j = 0; j < m_si.height; j++) {
            vector<uint64_t>* pSpikes = m_neuronList[i + j * m_si.width]->getSpikes();

            DEBUG2 (cout << endl << coordToString(i, j) << endl);

            for (unsigned int i = 0; i < (*pSpikes).size(); i++) {
                DEBUG2 (cout << i << " ");
                int idx1 = (*pSpikes)[i] * m_si.deltaT;
                burstinessHist[idx1] = burstinessHist[idx1] + 1.0;
                int idx2 = (*pSpikes)[i] * m_si.deltaT * 100;
                spikesHistory[idx2] = spikesHistory[idx2] + 1.0;
            }
        }
    }
}

/**
* Write the simulation memory image
*
* @param os	The filestream to write
*/
void Network::writeSimMemory(FLOAT simulation_step, ostream& os)
{
    // write the neurons data
    os.write(reinterpret_cast<const char*>(&m_si.cNeurons), sizeof(m_si.cNeurons));
    for (int i = 0; i < m_si.cNeurons; i++) {
        m_neuronList[i]->write(os);
    }

    // write the synapse data
    int synapse_count = 0;
    for (int i = 0; i < m_si.cNeurons; i++) {
        synapse_count += m_rgSynapseMap[i].size();
    }
    
    os.write(reinterpret_cast<const char*>(&synapse_count), sizeof(synapse_count));
    for (int i = 0; i < m_si.cNeurons; i++) {
        for (unsigned int j = 0; j < m_rgSynapseMap[i].size(); j++) {
            m_rgSynapseMap[i][j]->write(os);
        }
    }

    // write the final radii
    for (int i = 0; i < m_si.cNeurons; i++) {
        os.write(reinterpret_cast<const char*>(&radiiHistory(simulation_step, i)), sizeof(FLOAT));
    }

    // write the final rates
    for (int i = 0; i < m_si.cNeurons; i++) {
        os.write(reinterpret_cast<const char*>(&ratesHistory(simulation_step, i)), sizeof(FLOAT)); 
    }

    os.flush();
}

/**
* Read the simulation memory image
*
* @param is	The filestream to read
*/
void Network::readSimMemory(istream& is, VectorMatrix& radii, VectorMatrix& rates)
{
    // read the neuron data
    int cNeurons;
    is.read(reinterpret_cast<char*>(&cNeurons), sizeof(cNeurons));
    assert( cNeurons == m_si.cNeurons );

    for (int i = 0; i < m_si.cNeurons; i++)    
        m_neuronList[i]->read(is);    

    // read the synapse data & create synapses
    int synapse_count;
    is.read(reinterpret_cast<char*>(&synapse_count), sizeof(synapse_count));
    for (int i = 0; i < synapse_count; i++)
    {
	// read the synapse data and add it to the list
		// create synapse
		ISynapse* syn = new DynamicSpikingSynapse(is, m_summationMap, m_si.width);
		m_rgSynapseMap[syn->summationCoord.x + syn->summationCoord.y * m_si.width].push_back(syn);
    }

    // read the radii
    for (int i = 0; i < m_si.cNeurons; i++)    
        is.read(reinterpret_cast<char*>(&radii[i]), sizeof(FLOAT));    

    // read the rates
    for (int i = 0; i < m_si.cNeurons; i++)    
        is.read(reinterpret_cast<char*>(&rates[i]), sizeof(FLOAT));    
}

/**
* Copy neuron type array into VectorMatrix
*
* @param neuronTypes [out] Neuron type VectorMatrix
*/
void Network::getNeuronTypes(VectorMatrix& neuronTypes)
{
    for (int i = 0; i < m_si.cNeurons; i++) {
        switch (m_rgNeuronTypeMap[i]) {
            case INH:
                neuronTypes[i] = INH;
                break;

            case EXC:
                neuronTypes[i] = EXC;
                break;

            default:
                assert(false);
        }
    }
}

/**
* Get starter neuron matrix
*
* @param matrix [out] Starter neuron matrix
*/
void Network::getStarterNeuronMatrix(VectorMatrix& matrix)
{
    int cur = 0;
    for (int x = 0; x < m_si.width; x++) {
        for (int y = 0; y < m_si.height; y++) {
            if (m_rgEndogenouslyActiveNeuronMap[x + y * m_si.width]) {
                matrix[cur] = x + y * m_si.height;
                cur++;
            }
        }
    }

    assert (cur == m_cStarterNeurons);
}
