/**
 *	@file Network.cpp
 *
 *	@author Allan Ortiz & Cory Mayberry
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
Network::Network(FLOAT inhFrac, FLOAT excFrac, FLOAT startFrac, FLOAT Iinject[2],
        FLOAT Inoise[2], FLOAT Vthresh[2], FLOAT Vresting[2], FLOAT Vreset[2], FLOAT Vinit[2],
        FLOAT starter_Vthresh[2], FLOAT starter_Vreset[2], FLOAT new_targetRate,
        ostream& new_stateout, istream& new_meminput, bool fReadMemImage, 
        bool fFixedLayout, vector<int>* pEndogenouslyActiveNeuronLayout, vector<int>* pInhibitoryNeuronLayout,
        SimulationInfo simInfo, ISimulation* sim) :
    m_cExcitoryNeurons(static_cast<int>(simInfo.cNeurons * excFrac)), 
    m_cInhibitoryNeurons(static_cast<int>(simInfo.cNeurons * inhFrac)), 
    m_cStarterNeurons(static_cast<int>(simInfo.cNeurons * startFrac)), 
    m_rgSynapseMap(NULL),
    m_summationMap(NULL),
    m_rgNeuronTypeMap(NULL),
    m_rgEndogenouslyActiveNeuronMap(NULL),
    m_targetRate(new_targetRate),
    state_out(new_stateout),
    memory_in(new_meminput),
    m_fReadMemImage(fReadMemImage),
    m_fFixedLayout(fFixedLayout),
    m_pEndogenouslyActiveNeuronLayout(pEndogenouslyActiveNeuronLayout),
    m_pInhibitoryNeuronLayout(pInhibitoryNeuronLayout),
    m_si(simInfo),
    m_sim(sim),
    
    
    
    radii(MATRIX_TYPE, MATRIX_INIT, 1, simInfo.cNeurons),
    rates(MATRIX_TYPE, MATRIX_INIT, 1, simInfo.cNeurons),
    neuronTypes(MATRIX_TYPE, MATRIX_INIT, 1, simInfo.cNeurons, EXC),
    neuronThresh(MATRIX_TYPE, MATRIX_INIT, 1, simInfo.cNeurons, 0),
    xloc(MATRIX_TYPE, MATRIX_INIT, 1, simInfo.cNeurons),
    yloc(MATRIX_TYPE, MATRIX_INIT, 1, simInfo.cNeurons)
{
    cout << "Neuron count: " << simInfo.cNeurons << endl;
 
    // init data structures
    reset();

    // init neurons
    initNeurons(Iinject, Inoise, Vthresh, Vresting, Vreset, Vinit, starter_Vthresh, starter_Vreset);
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
    // burstiness Histogram goes through the
    burstinessHist = VectorMatrix(MATRIX_TYPE, MATRIX_INIT, 1, (int)(growthStepDuration * maxGrowthSteps), 0); // state

    // spikes history - history of accumulated spikes count of all neurons (10 ms bin)
    spikesHistory = VectorMatrix(MATRIX_TYPE, MATRIX_INIT, 1, (int)(growthStepDuration * maxGrowthSteps * 100), 0); // state

    // track radii
    radiiHistory = CompleteMatrix(MATRIX_TYPE, MATRIX_INIT, static_cast<int>(maxGrowthSteps + 1), m_si.cNeurons); // state

    // track firing rate
    ratesHistory = CompleteMatrix(MATRIX_TYPE, MATRIX_INIT, static_cast<int>(maxGrowthSteps + 1), m_si.cNeurons);
    
    for (int i = 0; i < m_si.cNeurons; i++)
        neuronThresh[i] = m_neuronList[i]->Vthresh;


    // Initialize neurons
    for (int i = 0; i < m_si.cNeurons; i++) {
        xloc[i] = i % m_si.width;
        yloc[i] = i / m_si.width;
    }

    // Populate neuron types with current values
    getNeuronTypes(neuronTypes);

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

    // Start the timer
    // TODO: stop the timer at some point and use its output
    // m_timer.start();

    // Initialize and prepare simulator
    m_sim->init(&m_si, xloc, yloc);

    // Set the previous saved radii
    if (m_fReadMemImage) {
        m_sim->initRadii(radii);
    }
}

void Network::finish(FLOAT growthStepDuration, FLOAT maxGrowthSteps)
{
#ifdef STORE_SPIKEHISTORY
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
#endif // STORE_SPIKEHISTORY

    saveSimState(state_out, growthStepDuration);

    // Terminate the simulator
    m_sim->term(&m_si); // Can #term be removed w/ the new model architecture?
}

/**
 * Notify outgoing synapses if neuron has fired.
 * @param[in] psi	Pointer to the simulation information.
 */
void Network::advanceNeurons(SimulationInfo* psi)
{
    // TODO: move this code into a helper class - it's being used in multiple places.
    // For each neuron in the network
    for (int i = psi->cNeurons - 1; i >= 0; --i) {
        // advance neurons
        (*(m_neuronList))[i]->advance(psi->pSummationMap[i]);

        DEBUG2(cout << i << " " << (*(m_neuronList))[i]->Vm << endl;)

        // notify outgoing synapses if neuron has fired
        if ((*(m_neuronList)[i]->hasFired) {
            DEBUG2(cout << " !! Neuron" << i << "has Fired @ t: " << g_simulationStep * psi->deltaT << endl;)

            for (int z = psi->rgSynapseMap[i].size() - 1; z >= 0; --z) {
                psi->rgSynapseMap[i][z]->preSpikeHit();
            }

            (*(m_neuronList))[i]->hasFired = false;
        }
    }

#ifdef DUMP_VOLTAGES
    // ouput a row with every voltage level for each time step
    cout << g_simulationStep * psi->deltaT;

    for (int i = 0; i < psi->cNeurons; i++) {
        cout << "\t i: " << i << " " << (*(m_neuronList))[i].toStringVm();
    }
    
    cout << endl;
#endif /* DUMP_VOLTAGES */
}

/**
 * @param[in] psi	Pointer to the simulation information.
 */
void Network::advanceSynapses(SimulationInfo* psi)
{
    for (int i = psi->cNeurons - 1; i >= 0; --i) {
        for (int z = psi->rgSynapseMap[i].size() - 1; z >= 0; --z) {
            psi->rgSynapseMap[i][z]->advance();
        }
    }
}

// Update the neuron network
void Network::update(SimulationInfo* psi)
{
    m_sim->updateNetwork(psi, radiiHistory, ratesHistory);
}

void Network::getSpikeCounts( int neuron_count, int *spikeCounts)
{
    for (int i = 0; i < neuron_count; i++) {
        spikeCounts[i] = m_NeuronList[i]->getSpikeCount();
    }
}

//! Clear spike count of each neuron.
void Network::clearSpikeCounts(int neuron_count)
{
    for (int i = 0; i < neuron_count; i++) {
         m_NeuronList[i]->->clearSpikeCount();
    }
}

/**
 * Calculate growth cycle firing rate for previous period.
 * Compute neuron radii change, assign new values, and record the radius to histroy matrix.
 * Update distance between frontiers, and compute areas of overlap. 
 * Adjust the strength of the synapse or remove it from the synapse map if it has gone below 
 * zero.
 * @param[in] psi	Pointer to the simulation information.
 * @param[in] radiiHistory	Matrix to save radius history.
 * @param[in] ratesHistory	Matrix to save firing rates history. 
 */
void Network::updateNetwork(SimulationInfo* psi)
{

    // For now, we just set the weights to equal the areas. We will later
    // scale it and set its sign (when we index and get its sign).
    W = area;

    int adjusted = 0;
    int could_have_been_removed = 0; // TODO: use this value
    int removed = 0;
    int added = 0;

    DEBUG(cout << "adjusting weights" << endl;)

    // Scale and add sign to the areas
    // visit each neuron 'a'
    for (int a = 0; a < psi->cNeurons; a++) {
        int xa = a % psi->width;
        int ya = a / psi->width;
        Coordinate aCoord(xa, ya);

        // and each destination neuron 'b'
        for (int b = 0; b < psi->cNeurons; b++) {
            int xb = b % psi->width;
            int yb = b / psi->width;
            Coordinate bCoord(xb, yb);

            // visit each synapse at (xa,ya)
            bool connected = false;

            // for each existing synapse
            for (size_t syn = 0; syn < psi->rgSynapseMap[a].size(); syn++) {
                // if there is a synapse between a and b
                if (psi->rgSynapseMap[a][syn]->summationCoord == bCoord) {
                    connected = true;
                    adjusted++;

                    // adjust the strength of the synapse or remove 
                    // it from the synapse map if it has gone below 
                    // zero.
                    if (W(a, b) < 0) {
                        removed++;
                        psi->rgSynapseMap[a].erase(psi->rgSynapseMap[a].begin() + syn);
                    } else {
                        // adjust
                        // g_synapseStrengthAdjustmentConstant is 1.0e-8;
                        psi->rgSynapseMap[a][syn]->W = W(a, b) * 
                            synSign(synType(psi, aCoord, bCoord)) * g_synapseStrengthAdjustmentConstant;

                        DEBUG2(cout << "weight of rgSynapseMap" << 
                               coordToString(xa, ya)<<"[" <<syn<<"]: " << 
                               psi->rgSynapseMap[a][syn].W << endl;);
                    }
                }
            }

            // if not connected and weight(a,b) > 0, add a new synapse from a to b
            if (!connected && (W(a, b) > 0)) {
                added++;

                ISynapse* newSynapse = addSynapse(psi, xa, ya, xb, yb);
                newSynapse->W = W(a, b) * synSign(synType(psi, aCoord, bCoord)) * g_synapseStrengthAdjustmentConstant;
            }
        }
    }

    DEBUG (cout << "adjusted: " << adjusted << endl;)
    DEBUG (cout << "could have been removed (TODO: calculate this): " << could_have_been_removed << endl;)
    DEBUG (cout << "removed: " << removed << endl;)
    DEBUG (cout << "added: " << added << endl << endl << endl;)
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

            ss << " " << networkRadii[x + y * psi->width];
            ss << " " << networkRadii[x + y * psi->width];

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
    // Empty neuron list
    if (m_rgEndogenouslyActiveNeuronMap != NULL) 
		delete[] m_rgEndogenouslyActiveNeuronMap;

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
 * Randomly populates the network accord to the neuron type counts and other parameters.
 * @post m_neuronList is populated.
 * @post m_rgNeuronTypeMap is populated.
 * @post m_pfStarterMap is populated.
 * @param Iinject
 * @param Inoise
 * @param Vthresh
 * @param Vresting
 * @param Vreset
 * @param Vinit
 * @param starter_Vthresh
 * @param starter_Vreset
 */
void Network::initNeurons(FLOAT Iinject[2], FLOAT Inoise[2], FLOAT Vthresh[2], FLOAT Vresting[2],
        FLOAT Vreset[2], FLOAT Vinit[2], FLOAT starter_Vthresh[2], FLOAT starter_Vreset[2])
{
    DEBUG(cout << "\nAllocating neurons..." << endl;)

    initNeuronTypeMap();
    initStarterMap();

    /* set their specific types */
    for (int i = 0; i < m_si.cNeurons; i++)
    {
        m_neuronList[i] = m_sim->returnNeuron();

        // set common parameters
        m_neuronList[i]->setParams(
            rng.inRange(Iinject[0], Iinject[1]), 
            rng.inRange(Inoise[0], Inoise[1]),
            rng.inRange(Vthresh[0], Vthresh[1]),
            rng.inRange(Vresting[0], Vresting[1]),
            rng.inRange(Vreset[0],Vreset[1]),
            rng.inRange(Vinit[0], Vinit[1]), 
            m_si.deltaT);

        switch (m_rgNeuronTypeMap[i])
        {
        case INH:
            DEBUG2(cout << "setting inhibitory neuron: "<< i << endl;)
            // set inhibitory absolute refractory period
            m_neuronList[i]->Trefract = DEFAULT_InhibTrefract;
            break;

        case EXC:
            DEBUG2(cout << "setting exitory neuron: " << i << endl;)
            // set excitory absolute refractory period
            m_neuronList[i]->Trefract = DEFAULT_ExcitTrefract;
            break;

        default:
            DEBUG2(cout << "ERROR: unknown neuron type: " << m_rgNeuronTypeMap[i] << "@" << i << endl;)
            assert(false);
        }

        if (m_rgEndogenouslyActiveNeuronMap[i])
        {
            DEBUG2(cout << "setting endogenously active neuron properties" << endl;)
            // set endogenously active threshold voltage, reset voltage, and refractory period
            m_neuronList[i]->Vthresh = rng.inRange(starter_Vthresh[0], starter_Vthresh[1]);
            m_neuronList[i]->Vreset = rng.inRange(starter_Vreset[0], starter_Vreset[1]);
            m_neuronList[i]->Trefract = DEFAULT_ExcitTrefract;
        }
        DEBUG2(cout << m_neuronList[i].toStringAll() << endl;)
    }
    DEBUG(cout << "Done initializing neurons..." << endl;)
}

/**
 * Randomly populates the m_rgNeuronTypeMap with the specified number of inhibitory and
 * excitory neurons.
 * @post m_rgNeuronTypeMap is populated.
 */
void Network::initNeuronTypeMap()
{
    DEBUG(cout << "\nInitializing neuron type map"<< endl;);

    // Get random neuron list
    vector<neuronType>* randomDist = getNeuronOrder();

    // Copy the contents of randomDist into m_rgNeuronTypeMap.
    // This is an spatial locality optimization - contiguous arrays usually cause
    // fewer cache misses.
    for (int i = 0; i < m_si.cNeurons; i++)
    {
        m_rgNeuronTypeMap[i] = (*randomDist)[i];
        DEBUG2(cout << "neuron" << i << " as " << neuronTypeToString(m_rgNeuronTypeMap[i]) << endl;);
    }

    delete randomDist;
    DEBUG(cout << "Done initializing neuron type map" << endl;);
}

/**
 * Populates the starter map.
 * Selects \e numStarter excitory neurons and converts them into starter neurons.
 * @pre m_rgNeuronTypeMap must already be properly initialized
 * @post m_pfStarterMap is populated.
 */
void Network::initStarterMap()
{
    if (m_fFixedLayout)
    {
        for (size_t i = 0; i < m_pEndogenouslyActiveNeuronLayout->size(); i++)        
            m_rgEndogenouslyActiveNeuronMap[m_pEndogenouslyActiveNeuronLayout->at(i)] = true;        
    }
    else
    {
        int starters_allocated = 0;

        DEBUG(cout << "\nRandomly initializing starter map\n";);
        DEBUG(cout << "Total neurons: " << m_si.cNeurons << endl;)
        DEBUG(cout << "Starter neurons: " << m_cStarterNeurons << endl;)

        // randomly set neurons as starters until we've created enough
        while (starters_allocated < m_cStarterNeurons)
        {
            // Get a random integer
            int i = static_cast<int>(rng.inRange(0, m_si.cNeurons));

            // If the neuron at that index is excitatory and a starter map
            // entry does not already exist, add an entry.
            if (m_rgNeuronTypeMap[i] == EXC && m_rgEndogenouslyActiveNeuronMap[i] == false)
            {
                m_rgEndogenouslyActiveNeuronMap[i] = true;
                starters_allocated++;
                DEBUG(cout << "allocated EA neuron at random index [" << i << "]" << endl;);
            }
        }

        DEBUG(cout <<"Done randomly initializing starter map\n\n";)
    }
}

/**
 *  Creates a randomly ordered distribution with the specified numbers of neuron types.
 *  @returns A flat vector (to map to 2-d [x,y] = [i % m_width, i / m_width])
 */
vector<neuronType>* Network::getNeuronOrder()
{
    vector<neuronType>* randomlyOrderedNeurons = new vector<neuronType>(0);

    // create a vector of neuron types, defaulting to EXC
    vector<neuronType> orderedNeurons(m_si.cNeurons, EXC);

    if (m_fFixedLayout)
    {
        /* setup neuron types */
        DEBUG(cout << "Total neurons: " << m_si.cNeurons << endl;)
        DEBUG(cout << "Inhibitory Neurons: " << m_pInhibitoryNeuronLayout->size() << endl;)
        DEBUG(cout << "Excitatory Neurons: " << (m_si.cNeurons - m_pInhibitoryNeuronLayout->size()) << endl;)

        randomlyOrderedNeurons->resize(m_si.cNeurons);

        for (int i = 0; i < m_si.cNeurons; i++)        
            (*randomlyOrderedNeurons)[i] = EXC;        

        for (size_t i = 0; i < m_pInhibitoryNeuronLayout->size(); i++)        
            (*randomlyOrderedNeurons)[m_pInhibitoryNeuronLayout->at(i)] = INH;        
    }
    else
    {
        DEBUG(cout << "\nDetermining random ordering...\n";)

        /* setup neuron types */
        DEBUG(cout << "total neurons: " << m_si.cNeurons << endl;)
        DEBUG(cout << "m_cInhibitoryNeurons: " << m_cInhibitoryNeurons << endl;)
        DEBUG(cout << "m_cExcitoryNeurons: " << m_cExcitoryNeurons << endl;)

        // set the correct number to INH
        for (int i = 0; i < m_cInhibitoryNeurons; i++)        
            orderedNeurons[i] = INH;        

        // Shuffle ordered list into an unordered list
        while (!orderedNeurons.empty())
        {
            int i = static_cast<int>(rng() * orderedNeurons.size());

            neuronType t = orderedNeurons[i];
            
            DEBUG2(cout << "ordered neuron [" << i << "], type: " << orderedNeurons[i] << endl;)
            DEBUG2(cout << " allocated to random neuron [" << randomlyOrderedNeurons->size() << "]" << endl;)

            // add random neuron to back
            randomlyOrderedNeurons->push_back(t);

            vector<neuronType>::iterator it = orderedNeurons.begin(); // get iterator to ordered's front

            for (int j = 0; j < i; j++) // move it forward until it is on the pushed neuron
                it++;
            
            orderedNeurons.erase(it); // and remove that neuron from the ordered list
        }

        DEBUG(cout << "Done determining random ordering" << endl;)
    }
    return randomlyOrderedNeurons;
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
void Network::saveSimState(ostream& os, FLOAT Tsim)
{
    // Write XML header information:
    os << "<?xml version=\"1.0\" standalone=\"no\"?>" << endl
       << "<!-- State output file for the DCT growth modeling-->" << endl;
    //os << version; TODO: version

    // Write the core state information:
    os << "<SimState>" << endl;
    os << "   " << radiiHistory.toXML("radiiHistory") << endl;
    os << "   " << ratesHistory.toXML("ratesHistory") << endl;
    os << "   " << burstinessHist.toXML("burstinessHist") << endl;
    os << "   " << spikesHistory.toXML("spikesHistory") << endl;
    os << "   " << xloc.toXML("xloc") << endl;
    os << "   " << yloc.toXML("yloc") << endl;
    os << "   " << neuronTypes.toXML("neuronTypes") << endl;

    if (m_cStarterNeurons > 0) {
        VectorMatrix starterNeuronsM("complete", "const", 1, m_cStarterNeurons);

        getStarterNeuronMatrix(starterNeuronsM);

        os << "   " << starterNeuronsM.toXML("starterNeurons") << endl;
    }

    // Write neuron thresold
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

/**
* Write the simulation memory image
*
* @param os	The filestream to write
*/
void Network::writeSimMemory(simulation_step, ostream& os)
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
    for (int i = 0; i < m_si.cNeurons; i++)
    {
        switch (m_rgNeuronTypeMap[i])
        {
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

    for (int x = 0; x < m_si.width; x++)
    {
        for (int y = 0; y < m_si.height; y++)
        {
            if (m_rgEndogenouslyActiveNeuronMap[x + y * m_si.width])
            {
                matrix[cur] = x + y * m_si.height;
                cur++;
            }
        }
    }

    assert (cur == m_cStarterNeurons);
}
