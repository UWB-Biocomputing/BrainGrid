#include "LIFModel.h"

#include "include/tinyxml.h"

#include "ParseParamError.h"
#include "Util.h"

LIFModel::LIFModel() :
     m_read_params(0)
    ,m_fixed_layout(false)
	,m_conns(NULL)
{

}

bool LIFModel::readParameters(TiXmlElement *source)
{
    m_read_params = 0;
    try {
        source->Accept(this);
    } catch (ParseParamError &error) {
        error.printError(cerr);
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
        if (element.QueryFLOATAttribute("min", &m_starter_Vthresh[0]) != TIXML_SUCCESS) {
            throw ParseParamError("starter_vthresh min", "starter_vthresh missing minimum value in XML.");
        }
        if (element.QueryFLOATAttribute("max", &m_starter_Vthresh[1]) != TIXML_SUCCESS) {
            throw ParseParamError("starter_vthresh max", "starter_vthresh missing maximum value in XML.");
        }
        m_read_params++;
    }

    if (element.ValueStr().compare("starter_vreset") == 0) {
        if (element.QueryFLOATAttribute("min", &m_starter_Vreset[0]) != TIXML_SUCCESS) {
            throw ParseParamError("starter_vreset min", "starter_vreset missing minimum value in XML.");
        }
        if (element.QueryFLOATAttribute("max", &m_starter_Vreset[1]) != TIXML_SUCCESS) {
            throw ParseParamError("starter_vreset max", "starter_vreset missing maximum value in XML.");
        }
        m_read_params++;
    }
    
    if (element.ValueStr().compare("GrowthParams") == 0) {
		if (element.QueryFLOATAttribute("epsilon", &m_growth.epsilon) != TIXML_SUCCESS) {
			throw ParseParamError("epsilon", "Growth param 'epsilon' missing in XML.");
		}
		if (element.QueryFLOATAttribute("beta", &m_growth.beta) != TIXML_SUCCESS) {
			throw ParseParamError("beta", "Growth param 'beta' missing in XML.");
		}
		if (element.QueryFLOATAttribute("rho", &m_growth.rho) != TIXML_SUCCESS) {
			throw ParseParamError("rho", "Growth param 'rho' missing in XML.");
		}
		if (element.QueryFLOATAttribute("targetRate", &m_growth.targetRate) != TIXML_SUCCESS) {
			throw ParseParamError("targetRate", "Growth targetRate 'beta' missing in XML.");
		}
		if (element.QueryFLOATAttribute("minRadius", &m_growth.minRadius) != TIXML_SUCCESS) {
			throw ParseParamError("minRadius", "Growth minRadius 'beta' missing in XML.");
		}
		if (element.QueryFLOATAttribute("startRadius", &m_growth.startRadius) != TIXML_SUCCESS) {
			throw ParseParamError("startRadius", "Growth startRadius 'beta' missing in XML.");
		}
    }

    // Parse fixed layout (overrides random layouts)
    if (element.ValueStr().compare("FixedLayout") == 0) {
        m_fixed_layout = true;

        TiXmlNode* pNode = NULL;
        while ((pNode = element.IterateChildren(pNode)) != NULL) {
            if (strcmp(pNode->Value(), "A") == 0)
                getValueList(pNode->ToElement()->GetText(), &m_endogenously_active_neuron_list);

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
    output << "frac_EXC:" << m_frac_excititory_neurons
    	   << " starter_neurons:" << starter_neurons
    	   << endl;
    output << "Interval of constant injected current: ["
    	   << m_Iinject[0] << ", " << m_Iinject[1] << "]"
           << endl;
    output << "Interval of STD of (gaussian) noise current: ["
           << m_Inoise[0] << ", " << m_Inoise[1] << "]"
           << endl;
    output << "Interval of firing threshold: ["
    	   << m_Vthresh[0] << ", "<< m_Vthresh[1] << "]"
           << endl;
    output << "Interval of asymptotic voltage (Vresting): [" << m_Vresting[0]
           << ", " << m_Vresting[1] << "]"
           << endl;
    output << "Interval of reset voltage: [" << m_Vreset[0]
           << ", " << m_Vreset[1] << "]"
           << endl;
    output << "Interval of initial membrance voltage: [" << m_Vinit[0]
           << ", " << m_Vinit[1] << "]"
           << endl;
    output << "Starter firing threshold: [" << m_starter_Vthresh[0]
           << ", " << m_starter_Vthresh[1] << "]"
           << endl;
    output << "Starter reset threshold: [" << m_starter_Vreset[0]
           << ", " << m_starter_Vreset[1] << "]"
           << endl;
    output << "Growth parameters: " << endl
    	   << "\tepsilon: " << m_growth.epsilon
    	   << ", beta: " << m_growth.beta
    	   << ", rho: " << m_growth.rho
    	   << ", targetRate: " << m_growth.targetRate << "," << endl
    	   << "\tminRadius: " << m_growth.minRadius
    	   << ", startRadius: " << m_growth.startRadius
    	   << endl;
    if (fFixedLayout) {
    	output << "Layout parameters:" << endl;

        cout << "\tEndogenously active neuron positions: ";
        for (size_t i = 0; i < endogenouslyActiveNeuronLayout.size(); i++) {
        	output << endogenouslyActiveNeuronLayout[i] << " ";
        }

        cout << endl;

        cout << "\tInhibitory neuron positions: ";
        for (size_t i = 0; i < inhibitoryNeuronLayout.size(); i++) {
        	output << inhibitoryNeuronLayout[i] << " ";
        }

        output << endl;
    }
}

/**
 * @return the complete state of the neuron.
 */
string LIFModel::neuron_to_string(AllNeurons &neurons, const int i) const
{
    stringstream ss;
    ss << "Cm: " << neurons.Cm[i] << " "; // membrane capacitance
    ss << "Rm: " << neurons.Rm[i] << " "; // membrane resistance
    ss << "Vthresh: " << neurons.Vthresh[i] << " "; // if Vm exceeds, Vthresh, a spike is emitted
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

void LIFModel::createAllNeurons(int num_neurons, AllNeurons &neurons)
{
    DEBUG(cout << "\nAllocating neurons..." << endl;)
    
    generate_neuron_type_map(neurons.neuron_type_map, num_neurons);
    init_starter_map(num_neurons, neurons.neuron_type_map);
    
    /* set their specific types */
    for (int i = 0; i < num_neurons; i++) {
        neurons.Iinject[i] = rng.inRange(m_Iinject[0], m_Iinject[1]);
        neurons.Inoise[i] = rng.inRange(m_Inoise[0], m_Inoise[1]);
        neurons.Vthresh[i] = rng.inRange(m_Vthresh[0], m_Vthresh[1]);
        neurons.Vrest[i] = rng.inRange(m_Vresting[0], m_Vresting[1]);
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
                break;
        }

        // endogenously_active_neuron_map -> Model State
        if (m_endogenously_active_neuron_layout[i]) {
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
void LIFModel::generate_neuron_type_map(neuronType neuron_types[], int num_neurons)
{
    //TODO: m_pInhibitoryNeuronLayout
    
    DEBUG(cout << "\nInitializing neuron type map"<< endl;);
    
    neuronType types[num_neurons];
    for (int i = 0; i < num_neurons; i++) {
        types[i] = EXC;
    }
    
    if (m_fixed_layout) {
        int num_inhibitory_neurons = m_inhibitory_neuron_layout.size();
        int num_excititory_neurons = num_neurons - num_inhibitory_neurons;
        DEBUG(cout << "Total neurons: " << num_neurons << endl;)
        DEBUG(cout << "Inhibitory Neurons: " << num_inhibitory_neurons << endl;)
        DEBUG(cout << "Excitatory Neurons: " << num_inhibitory_neurons << endl;)
        
        for (int i = 0; i < num_inhibitory_neurons; i++) {
            types[m_inhibitory_neuron_layout.at(i)] = INH;
        }
    } else {
        int num_excititory_neurons = (int) (frac_EXC * num_neurons + 0.5);
        int num_inhibitory_neurons = num_neurons - num_excititory_neurons;
        DEBUG(cout << "Total neurons: " << num_neurons << endl;)
        DEBUG(cout << "Inhibitory Neurons: " << num_inhibitory_neurons << endl;)
        DEBUG(cout << "Excitatory Neurons: " << num_inhibitory_neurons << endl;)
        
        DEBUG(cout << endl << "Randomly selecting inhibitory neurons..." << endl;)
        
        int rg_inhibitory_layout[num_inhibitory_neurons];
        
        for (int i = 0; i < num_inhibitory_neurons; i++) {
            rg_inhibitory_layout[i] = i;
        }
        
        for (int i = num_inhibitory_neurons; i < num_neurons; i++) {
            int j = static_cast<int>(rng() * num_neurons);
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
void LIFModel::init_starter_map(const int num_neurons, const neuronType neuron_type_map[])
{
    m_endogenously_active_neuron_layout = new bool[num_neurons];
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
            m_endogenously_active_neuron_layout[m_endogenously_active_neuron_list.at(i)] = true;
        }
    } else {
        int num_starter_neurons = (int) (m_frac_starter_neurons * num_neurons + 0.5);
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

void LIFModel::advance(const int num_neurons, AllNeurons &neurons, AllSynapses &synapses)
{
    advanceNeurons(num_neurons, neurons, synapses);
    advanceSynapses(num_neurons, synapses);
}

/**
 * Notify outgoing synapses if neuron has fired.
 * @param[in] psi - Pointer to the simulation information.
 */
void LIFModel::advanceNeurons(int num_neurons, AllNeurons &neurons, AllSynapses &synapses)
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

void LIFModel::advanceNeuron(AllNeurons &neurons, int idx, FLOAT& summationPoint)
{
    if (neurons.nStepsInRefr[idx] > 0) { // is neuron refractory?
        --neurons.nStepsInRefr[idx];
    } else if (neurons.Vm >= neurons.Vthresh) { // should it fire?
        fire( );
    } else {
        summationPoint += neurons.I0[idx]; // add IO
#ifdef USE_OMP
        int tid = OMP(omp_get_thread_num());
        summationPoint += ( (*rgNormrnd[tid])( ) * neurons.Inoise[idx] ); // add noise
#else
        summationPoint += ( (*rgNormrnd[0])( ) * neurons.Inoise[idx] ); // add noise
#endif
        neurons.Vm[idx] = neurons.C1[idx] * neurons.Vm[idx] + neurons.C2[idx] * summationPoint; // decay Vm and add inputs
    }
    // clear synaptic input for next time step
    summationPoint = 0;
}

/**
 * @param[in] psi - Pointer to the simulation information.
 */
void LIFModel::advanceSynapses(FLOAT num_neurons, AllSynapses &synapses)
{
    for (int i = num_neurons - 1; i >= 0; --i) {
        for (int z = synapses.size - 1; z >= 0; --z) {
            // Advance Synapse
            advanceSynapse(synapses, i, z);
        }
    }
}

void LIFModel::advanceSynapse(AllSynapses &synapses, int i, int z)
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

void LIFModel::updateConnections(const int currentStep, const int num_neurons)
{
	updateHistory(currentStep, 0 /*sim_info->stepDuration*/, num_neurons);
	updateFrontiers(num_neurons);
	updateOverlap(sim_info);
	updateWeights(num_neurons, sim_info);
}

void LIFModel::updateHistory(const int currentStep, FLOAT stepDuration, const int num_neurons)
{
	// Calculate growth cycle firing rate for previous period
	this->getSpikeCounts(num_neurons, m_conns->spikeCounts);

	// Calculate growth cycle firing rate for previous period
	for (int i = 0; i < num_neurons; i++) {
		// Calculate firing rate
		m_conns->rates[i] = m_conns->spikeCounts[i] / stepDuration;
		// record firing rate to history matrix
		m_conns->ratesHistory(currentStep, i) = m_conns->rates[i];
	}

	// clear spike count
	this->clearSpikeCounts(num_neurons);

	// compute neuron radii change and assign new values
	m_conns->outgrowth = 1.0 - 2.0 / (1.0 + exp((m_growth.epsilon - m_conns->rates / m_growth.maxRate) / m_growth.beta));
	m_conns->deltaR = stepDuration * m_growth.rho * m_conns->outgrowth;
	m_conns->radii += m_conns->deltaR;

	// Cap minimum radius size and record radii to history matrix
	for (int i = 0; i < m_conns->radii.Size(); i++) {
		// TODO: find out why we cap this here.
		if (m_conns->radii[i] < m_growth.minRadius) {
			m_conns->radii[i] = m_growth.minRadius;
		}

		// record radius to history matrix
		m_conns->radiiHistory(currentStep, i) = m_conns->radii[i];

		DEBUG2(cout << "radii[" << i << ":" << m_conns->radii[i] << "]" << endl;);
	}
}

void LIFModel::updateFrontiers(const int num_neurons)
{
	DEBUG(cout << "Updating distance between frontiers..." << endl;)
	// Update distance between frontiers
	for (int unit = 0; unit < num_neurons - 1; unit++) {
		for (int i = unit + 1; i < num_neurons; i++) {
			m_conns->delta(unit, i) = m_conns->dist(unit, i) - (m_conns->radii[unit] + m_conns->radii[i]);
			m_conns->delta(i, unit) = m_conns->delta(unit, i);
		}
	}
}

void LIFModel::updateOverlap(FLOAT num_neurons)
{
	DEBUG(cout << "computing areas of overlap" << endl;)

	// Compute areas of overlap; this is only done for overlapping units
	for (int i = 0; i < num_neurons; i++) {
		for (int j = 0; j < num_neurons; j++) {
			m_conns->area(i, j) = 0.0;

			if (m_conns->delta(i, j) < 0) {
				FLOAT lenAB = m_conns->dist(i, j);
				FLOAT r1 = m_conns->radii[i];
				FLOAT r2 = m_conns->radii[j];

				if (lenAB + min(r1, r2) <= max(r1, r2)) {
					m_conns->area(i, j) = pi * min(r1, r2) * min(r1, r2); // Completely overlapping unit
#ifdef LOGFILE
					logFile << "Completely overlapping (i, j, r1, r2, area): "
							<< i << ", " << j << ", " << r1 << ", " << r2 << ", " << *pAarea(i, j) << endl;
#endif // LOGFILE
				} else {
					// Partially overlapping unit
					FLOAT lenAB2 = m_conns->dist2(i, j);
					FLOAT r12 = r1 * r1;
					FLOAT r22 = r2 * r2;

					FLOAT cosCBA = (r22 + lenAB2 - r12) / (2.0 * r2 * lenAB);
					FLOAT angCBA = acos(cosCBA);
					FLOAT angCBD = 2.0 * angCBA;

					FLOAT cosCAB = (r12 + lenAB2 - r22) / (2.0 * r1 * lenAB);
					FLOAT angCAB = acos(cosCAB);
					FLOAT angCAD = 2.0 * angCAB;

					area(i, j) = 0.5 * (r22 * (angCBD - sin(angCBD)) + r12 * (angCAD - sin(angCAD)));
				}
			}
		}
	}
}

/**
 * Platform Dependent
 */
void LIFModel::updateWeights(const int num_neurons, SimulationInfo* sim_info)
{

	// For now, we just set the weights to equal the areas. We will later
	// scale it and set its sign (when we index and get its sign).
	m_conns->W = m_conns->area;

	int adjusted = 0;
	int could_have_been_removed = 0; // TODO: use this value
	int removed = 0;
	int added = 0;

	DEBUG(cout << "adjusting weights" << endl;)

	// Scale and add sign to the areas
	// visit each neuron 'a'
	for (int a = 0; a < num_neurons; a++) {
		int xa = a % sim_info->width;
		int ya = a / sim_info->width;
		Coordinate aCoord(xa, ya);

		// and each destination neuron 'b'
		for (int b = 0; b < num_neurons; b++) {
			int xb = b % sim_info->width;
			int yb = b / sim_info->width;
			Coordinate bCoord(xb, yb);

			// visit each synapse at (xa,ya)
			bool connected = false;

			// for each existing synapse
			for (size_t syn = 0; syn < sim_info->rgSynapseMap[a].size(); syn++) {
				// if there is a synapse between a and b
				if (sim_info->rgSynapseMap[a][syn]->summationCoord == bCoord) {
					connected = true;
					adjusted++;

					// adjust the strength of the synapse or remove
					// it from the synapse map if it has gone below
					// zero.
					if (W(a, b) < 0) {
						removed++;
						sim_info->rgSynapseMap[a].erase(sim_info->rgSynapseMap[a].begin() + syn);
					} else {
						// adjust
						// g_synapseStrengthAdjustmentConstant is 1.0e-8;
						sim_info->rgSynapseMap[a][syn]->W = W(a, b) *
							synSign(synType(sim_info, aCoord, bCoord)) * g_synapseStrengthAdjustmentConstant; // TODO( synSign in HostSim )

						DEBUG2(cout << "weight of rgSynapseMap" <<
							   coordToString(xa, ya)<<"[" <<syn<<"]: " <<
							   sim_info->rgSynapseMap[a][syn].W << endl;);
					}
				}
			}

			// if not connected and weight(a,b) > 0, add a new synapse from a to b
			if (!connected && (W(a, b) > 0)) {
				added++;

				 // TODO( addSynapse,synSign, synType in HostSim )
				ISynapse* newSynapse = addSynapse(sim_info, xa, ya, xb, yb);
				newSynapse->W = W(a, b) * synSign(synType(sim_info, aCoord, bCoord)) * g_synapseStrengthAdjustmentConstant;
			}
		}
	}

	DEBUG (cout << "adjusted: " << adjusted << endl;)
	DEBUG (cout << "could have been removed (TODO: calculate this): " << could_have_been_removed << endl;)
	DEBUG (cout << "removed: " << removed << endl;)
	DEBUG (cout << "added: " << added << endl << endl << endl;)
}

const string LIFModel::Connections::MATRIX_TYPE = "complete";
const string LIFModel::Connections::MATRIX_INIT = "const";

LIFModel::Connections::Connections(const int num_neurons, const double start_radius, const FLOAT maxGrowthSteps) :
	W(MATRIX_TYPE, MATRIX_INIT, num_neurons, num_neurons, 0),
	radii(MATRIX_TYPE, MATRIX_INIT, 1, num_neurons, start_radius),
	rates(MATRIX_TYPE, MATRIX_INIT, 1, num_neurons, 0),
	dist2(MATRIX_TYPE, MATRIX_INIT, num_neurons, num_neurons),
	delta(MATRIX_TYPE, MATRIX_INIT, num_neurons, num_neurons),
	dist(MATRIX_TYPE, MATRIX_INIT, num_neurons, num_neurons),
	area(MATRIX_TYPE, MATRIX_INIT, num_neurons, num_neurons, 0),
	outgrowth(MATRIX_TYPE, MATRIX_INIT, 1, num_neurons),
	deltaR(MATRIX_TYPE, MATRIX_INIT, 1, num_neurons),
    radiiHistory(MATRIX_TYPE, MATRIX_INIT, static_cast<int>(maxGrowthSteps + 1), num_neurons),
    ratesHistory(MATRIX_TYPE, MATRIX_INIT, static_cast<int>(maxGrowthSteps + 1), num_neurons)
{

}

