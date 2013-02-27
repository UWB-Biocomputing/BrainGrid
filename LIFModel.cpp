#include "LIFModel.h"

#include "ParseParamError.h"

LIFModel::LIFModel() :
    ,m_read_params(0)
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
    return m_read_params == 8;
}

// Visit an element.
bool VisitEnter(const TiXmlElement& element, const TiXmlAttribute* firstAttribute)
{
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
    }

    if (element.ValueStr().compare("Vthresh")== 0) {
        if (element.QueryFLOATAttribute("min", &m_Vthresh[0]) != TIXML_SUCCESS) {
            throw ParseParamError("Vthresh min", "Vthresh missing minimum value in XML.");
        }
        if (element.QueryFLOATAttribute("max", &m_Vthresh[1]) != TIXML_SUCCESS) {
            throw ParseParamError("Vthresh max", "Vthresh missing maximum value in XML.");
        }
        m_read_params++;
    }

    if (element.ValueStr().compare("Vresting")== 0) {
        if (element.QueryFLOATAttribute("min", &m_Vresting[0]) != TIXML_SUCCESS) {
            throw ParseParamError("Vresting min", "Vresting missing minimum value in XML.");
        }
        if (element.QueryFLOATAttribute("max", &m_Vresting[1]) != TIXML_SUCCESS) {
            throw ParseParamError("Vresting max", "Vresting missing maximum value in XML.");
        }
        m_read_params++;
    }

    if (element.ValueStr().compare("Vreset")== 0) {
        if (element.QueryFLOATAttribute("min", &m_Vreset[0]) != TIXML_SUCCESS) {
            throw ParseParamError("Vreset min", "Vreset missing minimum value in XML.");
        }
        if (element.QueryFLOATAttribute("max", &m_Vreset[1]) != TIXML_SUCCESS) {
            throw ParseParamError("Vreset max", "Vreset missing maximum value in XML.");
        }
        m_read_params++;
    }

    if (element.ValueStr().compare("Vinit")== 0) {
        if (element.QueryFLOATAttribute("min", &m_Vinit[0]) != TIXML_SUCCESS) {
            throw ParseParamError("Vinit min", "Vinit missing minimum value in XML.");
        }
        if (element.QueryFLOATAttribute("max", &m_Vinit[1]) != TIXML_SUCCESS) {
            throw ParseParamError("Vinit max", "Vinit missing maximum value in XML.");
        }
        m_read_params++;
    }

    if (element.ValueStr().compare("starter_vthresh")== 0) {
        if (element.QueryFLOATAttribute("min", &m_starter_vthresh[0]) != TIXML_SUCCESS) {
            throw ParseParamError("starter_vthresh min", "starter_vthresh missing minimum value in XML.");
        }
        if (element.QueryFLOATAttribute("max", &m_starter_vthresh[1]) != TIXML_SUCCESS) {
            throw ParseParamError("starter_vthresh max", "starter_vthresh missing maximum value in XML.");
        }
        m_read_params++;
    }

    if (element.ValueStr().compare("starter_vreset")== 0) {
        if (element.QueryFLOATAttribute("min", &m_starter_vreset[0]) != TIXML_SUCCESS) {
            throw ParseParamError("starter_vreset min", "starter_vreset missing minimum value in XML.");
        }
        if (element.QueryFLOATAttribute("max", &m_starter_vreset[1]) != TIXML_SUCCESS) {
            throw ParseParamError("starter_vreset max", "starter_vreset missing maximum value in XML.");
        }
        m_read_params++;
    }
    
    return true;
}

// Visit an element.
bool VisitExit( const TiXmlElement& element )
{
    return true;
}

void LIFModel::createAllNeurons(FLOAT neuron_count, neuronType *neuron_type_map, bool *endogenously_active_neuron_map, AllNeurons &neurons) const
{
    DEBUG(cout << "\nAllocating neurons..." << endl;)
    
    /* set their specific types */
    for (int i = 0; i < neuron_count; i++) {
        neurons.Iinject[i] = rng.inRange(m_Iinject[0], m_Iinject[1]);
        neurons.Inoise[i] = rng.inRange(m_Inoise[0], m_Inoise[1]);
        neurons.Vthresh[i] = rng.inRange(m_Vthresh[0], m_Vthresh[1]);
        neurons.Vresting[i] = rng.inRange(m_Vresting[0], m_Vresting[1]);
        neurons.Vreset[i] = rng.inRange(m_Vreset[0], m_Vreset[1]);
        neurons.Vinit[i] = rng.inRange(m_Vinit[0], m_Vinit[1]);
        neurons.deltaT[i] = m_si.deltaT);

        switch (neuron_type_map[i]) {
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

        if (endogenously_active_neuron_map[i]) {
            DEBUG2(cout << "setting endogenously active neuron properties" << endl;)
            // set endogenously active threshold voltage, reset voltage, and refractory period
            neurons.Vthresh[i] = rng.inRange(m_starter_Vthresh[0], m_starter_Vthresh[1]);
            neurons.Vreset[i] = rng.inRange(m_starter_Vreset[0], m_starter_Vreset[1]);
            neurons.Trefract[i] = DEFAULT_ExcitTrefract; // TODO(derek): move defaults inside model.
        }
        DEBUG2(cout << m_neuronList[i].toStringAll() << endl;)
    }
    
    DEBUG(cout << "Done initializing neurons..." << endl;)
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
void LIFModel::advanceNeurons(FLOAT neuron_count, AllNeurons &neurons, AllSynapses &synapses, SimulationInfo* psi)
{
    // TODO: move this code into a helper class - it's being used in multiple places.
    // For each neuron in the network
    for (int i = neuron_count - 1; i >= 0; --i) {
        // advance neurons
        advanceNeuron(neurons, i, psi->pSummationMap[i]);

        DEBUG2(cout << i << " " << neurons.Vm[i] << endl;)

        // notify outgoing synapses if neuron has fired
        if (neurons.hasFired[i]) {
            DEBUG2(cout << " !! Neuron" << i << "has Fired @ t: " << g_simulationStep * psi->deltaT << endl;)

            for (int z = psi->rgSynapseMap[i].size() - 1; z >= 0; --z) {
                psi->rgSynapseMap[i][z]->preSpikeHit();
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
void LIFModel::advanceSynapses(SimulationInfo* psi)
{
    for (int i = psi->cNeurons - 1; i >= 0; --i) {
        for (int z = psi->rgSynapseMap[i].size() - 1; z >= 0; --z) {
            psi->rgSynapseMap[i][z]->advance();
        }
    }
}
