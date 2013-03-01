#include "LIFModel.h"

#include "ParseParamError.h"
#include "Util.h"

LIFModel::LIFModel() :
     m_read_params(0)
    ,m_fixed_layout(false)
    ,num_starter_neurons(0)
{

}

bool LIFModel::readParameters(TiXmlElement *source)
{
    m_read_params = 0;
    try {
        source->Accept(this);
    } catch (ParseParamError error) {
        error.print(cerr);
        cerr << endl;
        return false;
    }
    return m_read_params == 9;
}

// Visit an element.
bool LIFModel::VisitEnter(const TiXmlElement& element, const TiXmlAttribute* firstAttribute)
{
    if (element.ValueStr().compare("LsmParams") == 0) {
        if (element.QueryFLOATAttribute("frac_EXC", &m_frac_excititory_neurons) != TIXML_SUCCESS) {
            throw ParseParamError("frac_EXC", "Fraction Excitatory missing in XML.");
        }
        if (element.QueryFLOATAttribute("starter_neurons", &m_frac_starter_neurons) != TIXML_SUCCESS) {
            throw ParseParamError("starter_neurons", "Fraction endogenously active missing in XML.");
        }
    }
    
    if (element.ValueStr().compare("Iinject") == 0) {
        if (element.QueryFLOATAttribute("min", &m_Iinject[0]) != TIXML_SUCCESS) {
            throw ParseParamError("Iinject min", "Iinject missing minimum value in XML.");
        }
        if (element.QueryFLOATAttribute("max", &m_Iinject[1]) != TIXML_SUCCESS) {
            throw ParseParamError("Iinject min", "Iinject missing maximum value in XML.");
        }
        m_read_params++;
        return false;
    }
    
    if (element.ValueStr().compare("Inoise") == 0) {
        if (element.QueryFLOATAttribute("min", &m_Inoise[0]) != TIXML_SUCCESS) {
            throw ParseParamError("Inoise min", "Inoise missing minimum value in XML.");
        }
        if (element.QueryFLOATAttribute("max", &m_Inoise[1]) != TIXML_SUCCESS) {
            throw ParseParamError("Inoise max", "Inoise missing maximum value in XML.");
        }
        m_read_params++;
        return false; // TODO
    }

    if (element.ValueStr().compare("Vthresh") == 0) {
        if (element.QueryFLOATAttribute("min", &m_Vthresh[0]) != TIXML_SUCCESS) {
            throw ParseParamError("Vthresh min", "Vthresh missing minimum value in XML.");
        }
        if (element.QueryFLOATAttribute("max", &m_Vthresh[1]) != TIXML_SUCCESS) {
            throw ParseParamError("Vthresh max", "Vthresh missing maximum value in XML.");
        }
        m_read_params++;
    }

    if (element.ValueStr().compare("Vresting") == 0) {
        if (element.QueryFLOATAttribute("min", &m_Vresting[0]) != TIXML_SUCCESS) {
            throw ParseParamError("Vresting min", "Vresting missing minimum value in XML.");
        }
        if (element.QueryFLOATAttribute("max", &m_Vresting[1]) != TIXML_SUCCESS) {
            throw ParseParamError("Vresting max", "Vresting missing maximum value in XML.");
        }
        m_read_params++;
    }

    if (element.ValueStr().compare("Vreset") == 0) {
        if (element.QueryFLOATAttribute("min", &m_Vreset[0]) != TIXML_SUCCESS) {
            throw ParseParamError("Vreset min", "Vreset missing minimum value in XML.");
        }
        if (element.QueryFLOATAttribute("max", &m_Vreset[1]) != TIXML_SUCCESS) {
            throw ParseParamError("Vreset max", "Vreset missing maximum value in XML.");
        }
        m_read_params++;
    }

    if (element.ValueStr().compare("Vinit") == 0) {
        if (element.QueryFLOATAttribute("min", &m_Vinit[0]) != TIXML_SUCCESS) {
            throw ParseParamError("Vinit min", "Vinit missing minimum value in XML.");
        }
        if (element.QueryFLOATAttribute("max", &m_Vinit[1]) != TIXML_SUCCESS) {
            throw ParseParamError("Vinit max", "Vinit missing maximum value in XML.");
        }
        m_read_params++;
    }

    if (element.ValueStr().compare("starter_vthresh") == 0) {
        if (element.QueryFLOATAttribute("min", &m_starter_vthresh[0]) != TIXML_SUCCESS) {
            throw ParseParamError("starter_vthresh min", "starter_vthresh missing minimum value in XML.");
        }
        if (element.QueryFLOATAttribute("max", &m_starter_vthresh[1]) != TIXML_SUCCESS) {
            throw ParseParamError("starter_vthresh max", "starter_vthresh missing maximum value in XML.");
        }
        m_read_params++;
    }

    if (element.ValueStr().compare("starter_vreset") == 0) {
        if (element.QueryFLOATAttribute("min", &m_starter_vreset[0]) != TIXML_SUCCESS) {
            throw ParseParamError("starter_vreset min", "starter_vreset missing minimum value in XML.");
        }
        if (element.QueryFLOATAttribute("max", &m_starter_vreset[1]) != TIXML_SUCCESS) {
            throw ParseParamError("starter_vreset max", "starter_vreset missing maximum value in XML.");
        }
        m_read_params++;
    }
    
    // Parse fixed layout (overrides random layouts)
    if (element.ValueStr().compare("FixedLayout") == 0) {
        m_fixed_layout = true;

        TiXmlNode* pNode = NULL;
        while ((pNode = element->IterateChildren(pNode)) != NULL) {
            if (strcmp(pNode->Value(), "A") == 0)
                getValueList(pNode->ToElement()->GetText(), &m_endogenously_active_neuron_layout);

            else if (strcmp(pNode->Value(), "I") == 0)
                getValueList(pNode->ToElement()->GetText(), &m_inhibitory_neuron_layout);
        }
    }
    
    return true;
}

/*
// Visit an element.
bool VisitExit( const TiXmlElement& element )
{
    return true;
}
*/

// TODO(derek) : comment
void LIFModel::printParameters(ostream &output) const
{
    output << "frac_EXC:" << frac_EXC << " " << "starter_neurons:"
           << starter_neurons << endl;
    output << "Interval of constant injected current: [" << Iinject[0]
           << ", " << Iinject[1] << "]"
           << endl;
    output << "Interval of STD of (gaussian) noise current: [" << Inoise[0]
           << ", " << Inoise[1] << "]\n";
    output << "Interval of firing threshold: [" << Vthresh[0] << ", "
           << Vthresh[1] << "]\n";
    output << "Interval of asymptotic voltage (Vresting): [" << Vresting[0]
           << ", " << Vresting[1] << "]\n";
    output << "Interval of reset voltage: [" << Vreset[0]
           << ", " << Vreset[1] << "]\n";
    output << "Interval of initial membrance voltage: [" << Vinit[0]
           << ", " << Vinit[1] << "]\n";
    output << "Starter firing threshold: [" << starter_vthresh[0]
           << ", " << starter_vthresh[1] << "]\n";
    output << "Starter reset threshold: [" << starter_vreset[0]
           << ", " << starter_vreset[1] << "]\n";
}

/**
 * @return the complete state of the neuron.
 */
void LIFModel::neuron_to_string(AllNeurons &neurons, const int i) const
{
    stringstream ss;
    ss << "Cm: " << neurons.Cm[i] << " "; // membrane capacitance
    ss << "Rm: " << neurons.Rm[i] << " "; // membrane resistance
    ss << "Vthresh: " << Vneurons.thresh[i] << " "; // if Vm exceeds, Vthresh, a spike is emitted
    ss << "Vrest: " << neurons.Vrest[i] << " "; // the resting membrane voltage
    ss << "Vreset: " << neurons.Vreset[i] << " "; // The voltage to reset Vm to after a spike
    ss << "Vinit: " << neurons.Vinit[i] << endl; // The initial condition for V_m at t=0
    ss << "Trefract: " << neurons.Trefract[i] << " "; // the number of steps in the refractory period
    ss << "Inoise: " << neurons.Inoise[i] << " "; // the stdev of the noise to be added each delta_t
    ss << "Iinject: " << neurons.Iinject[i] << " "; // A constant current to be injected into the LIF neuron
    ss << "nStepsInRefr: " << neurons.nStepsInRefr[i] << endl; // the number of steps left in the refractory period
    ss << "Vm: " << neurons.Vm[i] << " "; // the membrane voltage
    ss << "hasFired: " << neurons.hasFired[i] << " "; // it done fired?
    ss << "C1: " << neurons.C1[i] << " ";
    ss << "C2: " << neurons.C2[i] << " ";
    ss << "I0: " << neurons.I0[i] << " ";
    return ss.str( );
}

void LIFModel::createAllNeurons(FLOAT num_neurons, AllNeurons &neurons) const
{
    DEBUG(cout << "\nAllocating neurons..." << endl;)
    
    generate_neuron_type_map(neurons.neuron_type_map, num_neurons);
    init_starter_map(num_neurons, neurons.neuron_type_map);
    
    /* set their specific types */
    for (int i = 0; i < num_neurons; i++) {
        neurons.Iinject[i] = rng.inRange(m_Iinject[0], m_Iinject[1]);
        neurons.Inoise[i] = rng.inRange(m_Inoise[0], m_Inoise[1]);
        neurons.Vthresh[i] = rng.inRange(m_Vthresh[0], m_Vthresh[1]);
        neurons.Vresting[i] = rng.inRange(m_Vresting[0], m_Vresting[1]);
        neurons.Vreset[i] = rng.inRange(m_Vreset[0], m_Vreset[1]);
        neurons.Vinit[i] = rng.inRange(m_Vinit[0], m_Vinit[1]);
        neurons.deltaT[i] = m_si.deltaT);
        
        DEBUG2(cout << "neuron" << i << " as " << neuronTypeToString(neurons.neuron_type_map[i]) << endl;);
        
        switch (neurons.neuron_type_map[i]) {
            case INH:
                DEBUG2(cout << "setting inhibitory neuron: "<< i << endl;)
                // set inhibitory absolute refractory period
                neurons.Trefract[i] = DEFAULT_InhibTrefract; // TODO(derek): move defaults inside model.
                break;

            case EXC:
                DEBUG2(cout << "setting exitory neuron: " << i << endl;)
                // set excitory absolute refractory period
                neurons.Trefract[i] = DEFAULT_ExcitTrefract;
                break;

            default:
                DEBUG2(cout << "ERROR: unknown neuron type: " << m_rgNeuronTypeMap[i] << "@" << i << endl;)
                assert(false);
        }

        // endogenously_active_neuron_map -> Model State
        if (endogenously_active_neuron_map[i]) {
            DEBUG2(cout << "setting endogenously active neuron properties" << endl;)
            // set endogenously active threshold voltage, reset voltage, and refractory period
            neurons.Vthresh[i] = rng.inRange(m_starter_Vthresh[0], m_starter_Vthresh[1]);
            neurons.Vreset[i] = rng.inRange(m_starter_Vreset[0], m_starter_Vreset[1]);
            neurons.Trefract[i] = DEFAULT_ExcitTrefract; // TODO(derek): move defaults inside model.
        }
        DEBUG2(cout << neuron_to_string(neurons, i) << endl;)
    }
    
    DEBUG(cout << "Done initializing neurons..." << endl;)
}

/**
 *  Creates a randomly ordered distribution with the specified numbers of neuron types.
 *  @returns A flat vector (to map to 2-d [x,y] = [i % m_width, i / m_width])
 */
void Network::generate_neuron_type_map(neuronType neuron_types[], num_neurons)
{
    //TODO: m_pInhibitoryNeuronLayout
    
    DEBUG(cout << "\nInitializing neuron type map"<< endl;);
    
    neuronType types[num_neurons];
    for (int i = 0; i < num_neurons; i++) {
        types[i] = EXC;
    }
    
    if (m_fixed_layout) {
        int num_inhibitory_neurons = m_inhibitory_neuron_layout->size();
        int num_excititory_neurons = num_neurons - num_inhibitory_neurons;
        DEBUG(cout << "Total neurons: " << num_neurons << endl;)
        DEBUG(cout << "Inhibitory Neurons: " << num_inhibitory_neurons << endl;)
        DEBUG(cout << "Excitatory Neurons: " << num_inhibitory_neurons << endl;)
        
        for (int i = 0; i < num_inhibitory_neurons; i++) {
            types[m_pInhibitoryNeuronLayout->at(i)] = INH;
        }
    } else {
        int num_excititory_neurons = (int) (frac_EXC * num_neurons + 0.5);
        int num_inhibitory_neurons = numNeurons - nInhNeurons;
        DEBUG(cout << "Total neurons: " << num_neurons << endl;)
        DEBUG(cout << "Inhibitory Neurons: " << num_inhibitory_neurons << endl;)
        DEBUG(cout << "Excitatory Neurons: " << num_inhibitory_neurons << endl;)
        
        DEBUG(cout << endl << "Randomly selecting inhibitory neurons..." << endl;)
        
        int rg_inhibitory_layout[num_inhibitory_neurons];
        
        for (int i = 0; i < num_inhibitory_neurons; i++) {
            rg_inhibitory_layout[i] = i;
        }
        
        for (int i = num_inhibitory_neurons; i < num_neurons; i++) {
            int j = static_cast<int>(rng() * orderedNeurons.size());
            if (j < num_inhibitory_neurons) {
                rg_inhibitory_layout[j] = i;
            }
        }
        
        for (int i = 0; i < num_inhibitory_neurons; i++) {
            types[rg_inhibitory_layout[i]] = INH;
        }
    }
    
    DEBUG(cout << "Done initializing neuron type map" << endl;);
}

/**
 * Populates the starter map.
 * Selects \e numStarter excitory neurons and converts them into starter neurons.
 * @pre m_rgNeuronTypeMap must already be properly initialized
 * @post m_pfStarterMap is populated.
 */
void LIFModel::init_starter_map(const int num_neurons, const neuron_type_map[])
{
    m_endogenously_active_neuron_layout = new bool[num_neurons]
    for (int i = 0; i < num_neurons; i++) {
        m_endogenously_active_neuron_layout[i] = false;
    }
    
    int num_starter_neurons = 0;
    if (!starter_flag) {
        return;
    }
    
    if (m_fixed_layout) {
        int num_endogenously_active_neurons = m_endogenously_active_neuron_list.size();
        for (size_t i = 0; i < num_endogenously_active_neurons; i++) {
            m_endogenously_active_neuron_map[m_endogenously_active_neuron_list->at(i)] = true;
        }
    } else {
        int num_starter_neurons = (int) (m_frac_starter_neurons * numNeurons + 0.5);
        int starters_allocated = 0;

        DEBUG(cout << "\nRandomly initializing starter map\n";);
        DEBUG(cout << "Total neurons: " << num_neurons << endl;)
        DEBUG(cout << "Starter neurons: " << num_starter_neurons << endl;)

        // randomly set neurons as starters until we've created enough
        while (starters_allocated < num_starter_neurons) {
            // Get a random integer
            int i = static_cast<int>(rng.inRange(0, num_neurons));

            // If the neuron at that index is excitatory and a starter map
            // entry does not already exist, add an entry.
            if (neuron_type_map[i] == EXC && m_endogenously_active_neuron_layout[i] == false) {
                m_endogenously_active_neuron_layout[i] = true;
                starters_allocated++;
                DEBUG(cout << "allocated EA neuron at random index [" << i << "]" << endl;);
            }
        }

        DEBUG(cout <<"Done randomly initializing starter map\n\n";)
    }
}

void LIFModel::advance(FLOAT neuron_count, AllNeurons &neurons, AllSynapses &synapses)
{
    advanceNeurons(neuron_count, neurons, synapses);
    advanceSynapses(neuron_count, synapses);
}

/**
 * Notify outgoing synapses if neuron has fired.
 * @param[in] psi - Pointer to the simulation information.
 */
void LIFModel::advanceNeurons(FLOAT num_neurons, AllNeurons &neurons, AllSynapses &synapses, SimulationInfo* psi)
{
    // TODO: move this code into a helper class - it's being used in multiple places.
    // For each neuron in the network
    for (int i = num_neurons - 1; i >= 0; --i) {
        // advance neurons
        advanceNeuron(neurons, i, psi->pSummationMap[i]);

        DEBUG2(cout << i << " " << neurons.Vm[i] << endl;)

        // notify outgoing synapses if neuron has fired
        if (neurons.hasFired[i]) {
            DEBUG2(cout << " !! Neuron" << i << "has Fired @ t: " << g_simulationStep * psi->deltaT << endl;)

            for (int z = synapses[i].size - 1; z >= 0; --z) {
                preSpikeHit(synapses[i], z);
            }

            neurons.hasFired[i] = false;
        }
    }

#ifdef DUMP_VOLTAGES
    // ouput a row with every voltage level for each time step
    cout << g_simulationStep * psi->deltaT;

    for (int i = 0; i < psi->cNeurons; i++) {
        cout << "\t i: " << i << " " << m_neuronList[i].toStringVm();
    }
    
    cout << endl;
#endif /* DUMP_VOLTAGES */
}

void LIFModel::preSpikeHit(synapses[i], z)
{
    
}

void advanceNeuron(AllNeurons &neurons, int neuron_index, FLOAT& summationPoint)
{
    if (neurons.nStepsInRefr[i] > 0) { // is neuron refractory?
        --nStepsInRefr;
    } else if (neurons.Vm >= neurons.Vthresh) { // should it fire?
        fire( );
    } else {
        summationPoint += I0; // add IO
#ifdef USE_OMP
        int tid = OMP(omp_get_thread_num());
        summationPoint += ( (*rgNormrnd[tid])( ) * neurons.Inoise[i] ); // add noise
#else
        summationPoint += ( (*rgNormrnd[0])( ) * neurons.Inoise[i] ); // add noise
#endif
        neurons.Vm[i] = neurons.C1[i] * neurons.Vm[i] + neurons.C2[i] * summationPoint; // decay Vm and add inputs
    }
    // clear synaptic input for next time step
    summationPoint = 0;
}

/**
 * @param[in] psi - Pointer to the simulation information.
 */
void LIFModel::advanceSynapses(FLOAT num_neurons, AllSynapses *synapses)
{
    for (int i = num_neurons - 1; i >= 0; --i) {
        for (int z = synapses.size - 1; z >= 0; --z) {
            // Advance Synapse
            advanceSynapse()
            psi->rgSynapseMap[i][z]->advance();
        }
    }
}

void LIFModel::advanceSynapse(AllSynapses &synapses, int i)
{
    // is an input in the queue?
    if (isSpikeQueue()) {
        // adjust synapse paramaters
        if (lastSpike != ULONG_MAX) {
            FLOAT isi = (g_simulationStep - lastSpike) * deltaT ;
            r = 1 + ( r * ( 1 - u ) - 1 ) * exp( -isi / D );
            u = U + u * ( 1 - U ) * exp( -isi / F );
        }
        psr += ( ( W / decay ) * u * r );// calculate psr
        lastSpike = g_simulationStep; // record the time of the spike
    }

    // decay the post spike response
    psr *= decay;
    // and apply it to the summation point
#ifdef USE_OMP
#pragma omp atomic
#endif
    summationPoint += psr;
#ifdef USE_OMP
    //PAB: atomic above has implied flush (following statement generates error -- can't be member variable)
    //#pragma omp flush (summationPoint)
#endif
}

void LIFModel::updateConnections(Network &network) const
{
    
}
