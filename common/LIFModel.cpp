#include "LIFModel.h"

#include "../tinyxml/tinyxml.h"

#include "ParseParamError.h"
#include "Util.h"

const bool LIFModel::STARTER_FLAG(true);

const BGFLOAT LIFModel::SYNAPSE_STRENGTH_ADJUSTMENT = 1.0e-8;

/**
 *  Constructor
 */
LIFModel::LIFModel() :
     m_read_params(0)
    ,m_fixed_layout(false)
    ,m_conns(NULL)
{

}

/**
 *  Destructor
 */
LIFModel::~LIFModel()
{
    if (m_conns != NULL) {
        delete m_conns;
        m_conns = NULL;
    }
}

/**
 *  Attempts to read parameters from a XML file.
 *  @param  source  the TiXmlElement to read from.
 *  @return true if successful, false otherwise.
 */
bool LIFModel::readParameters(TiXmlElement *source)
{
    m_read_params = 0;
    try {
        source->Accept(this);
    } catch (ParseParamError &error) {
        error.print(cerr);
        cerr << endl;
        return false;
    }

    // initial maximum firing rate
    m_growth.maxRate = m_growth.targetRate / m_growth.epsilon;

    cout << "GROWTH PARAMS :: " << m_growth << endl;

    return m_read_params == 9;
}

/**
 *  Takes an XmlElement and checks for errors. If not, calls getValueList().
 *  @param  element TiXmlElement to examine.
 *  @param  firstAttribute  ***NOT USED***.
 *  @return true if method finishes without errors.
 */
bool LIFModel::VisitEnter(const TiXmlElement& element, const TiXmlAttribute* firstAttribute)
//TODO: firstAttribute does not seem to be used!
{
    //-----------------------------------------------------------------------//
    //                           Begin Error Checking                        //
    //-----------------------------------------------------------------------//
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
    //-----------------------------------------------------------------------//
    //                          End Error Checking                           //
    //-----------------------------------------------------------------------//

    // Parse fixed layout (overrides random layouts)
    if (element.ValueStr().compare("FixedLayout") == 0) {
        m_fixed_layout = true;

        const TiXmlNode* pNode = NULL;
        while ((pNode = element.IterateChildren(pNode)) != NULL) {
            if (strcmp(pNode->Value(), "A") == 0) {
                getValueList(pNode->ToElement()->GetText(), &m_endogenously_active_neuron_list);
            } else if (strcmp(pNode->Value(), "I") == 0) {
                getValueList(pNode->ToElement()->GetText(), &m_inhibitory_neuron_layout);
            }
        }
    }
    
    return true;
}

/**
 *  Prints out all parameters of the model to ostream.
 *  @param  output  ostream to send output to.
 */
void LIFModel::printParameters(ostream &output) const
{
    output << "frac_EXC:" << m_frac_excititory_neurons
           << " starter_neurons:" << m_frac_starter_neurons
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
    if (m_fixed_layout) {
        output << "Layout parameters:" << endl;

        cout << "\tEndogenously active neuron positions: ";
        for (size_t i = 0; i < m_endogenously_active_neuron_list.size(); i++) {
            output << m_endogenously_active_neuron_list[i] << " ";
        }

        cout << endl;

        cout << "\tInhibitory neuron positions: ";
        for (size_t i = 0; i < m_inhibitory_neuron_layout.size(); i++) {
            output << m_inhibitory_neuron_layout[i] << " ";
        }

        output << endl;
    }
}

/**
 *  Outputs state of the neuron chosen as a string.
 *  @param  neurons the entire list of neurons.
 *  @param  i   index of the neuron (in neurons) to output info from.
 *  @return the complete state of the neuron.
 */
string LIFModel::neuronToString(AllNeurons &neurons, const int i) const
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

/**
 *  Loads the simulation based on istream input.
 *  @param  input   istream to read from.
 *  @param  neurons list of neurons to set.
 *  @param  synapses    list of synapses to set.
 *  @param  sim_info    used as a reference to set info for neurons and synapses.
 */
void LIFModel::loadMemory(istream& input, AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo &sim_info)
{
    for (int i = 0; i < neurons.size; i++) {
        readNeuron(input, neurons, i);
    }

    int* read_synapses_counts= new int[neurons.size];
    for (int i = 0; i < neurons.size; i++) {
        read_synapses_counts[i] = 0;
    }

    // read the synapse data & create synapses
    int synapse_count;
    input >> synapse_count;
    for (int i = 0; i < synapse_count; i++) {
    // read the synapse data and add it to the list
        // create synapse
        Coordinate summation_coord;
        input >> summation_coord.x;
        input >> summation_coord.y;

        int neuron_index = summation_coord.x + summation_coord.y * sim_info.width;
        int synapses_index = read_synapses_counts[neuron_index];

        synapses.summationCoord[neuron_index][synapses_index] = summation_coord;

        readSynapse(input, synapses, neuron_index, synapses_index);

        synapses.summationPoint[neuron_index][synapses_index] = &(neurons.summation_map[summation_coord.x + summation_coord.y * sim_info.width]);

        read_synapses_counts[neuron_index]++;

    }
    delete[] read_synapses_counts;

    // read the radii
    for (int i = 0; i < neurons.size; i++)
        input >> m_conns->radii[i];

    // read the rates
    for (int i = 0; i < neurons.size; i++) {
        input >> m_conns->rates[i];
    }

    for (int i = 0; i < neurons.size; i++) {
        m_conns->radiiHistory(0, i) = m_conns->radii[i]; // NOTE: Radii Used for read.
        m_conns->ratesHistory(0, i) = m_conns->rates[i]; // NOTE: Rates Used for read.
    }
}

/**
 *  Sets the data for Neuron #index to input's data.
 *  @param  input   istream to read from.
 *  @param  neurons neuron list to find the indexed neuron from.
 *  @param  index   index of neuron to set.
 */
void LIFModel::readNeuron(istream &input, AllNeurons &neurons, const int index)
{
    // input.ignore() so input skips over end-of-line characters.
    input >> neurons.deltaT[index]; input.ignore();
    input >> neurons.Cm[index]; input.ignore();
    input >> neurons.Rm[index]; input.ignore();
    input >> neurons.Vthresh[index]; input.ignore();
    input >> neurons.Vrest[index]; input.ignore();
    input >> neurons.Vreset[index]; input.ignore();
    input >> neurons.Vinit[index]; input.ignore();
    input >> neurons.Trefract[index]; input.ignore();
    input >> neurons.Inoise[index]; input.ignore();
    input >> neurons.Iinject[index]; input.ignore();
    input >> neurons.Isyn[index]; input.ignore();
    input >> neurons.nStepsInRefr[index]; input.ignore();
    input >> neurons.C1[index]; input.ignore();
    input >> neurons.C2[index]; input.ignore();
    input >> neurons.I0[index]; input.ignore();
    input >> neurons.Vm[index]; input.ignore();
    input >> neurons.hasFired[index]; input.ignore();
    input >> neurons.Tau[index]; input.ignore();
}

/**
 *  Sets the data for Synapse #synapse_index from Neuron #neuron_index.
 *  @param  input   istream to read from.
 *  @param  synapses    synapse list to find the indexed synapse from.
 *  @param  neuron_index    index of the neuron that the synapse belongs to.
 *  @param  synapse_index   index of the synapse to set.
 */
void LIFModel::readSynapse(istream &input, AllSynapses &synapses, const int neuron_index, const int synapse_index)
{
    // initialize spike queue
    initSpikeQueue(synapses, neuron_index, synapse_index);
    resetSynapse(synapses, neuron_index, synapse_index);

    int synapse_type(0);

    // input.ignore() so input skips over end-of-line characters.
    input >> synapses.synapseCoord[neuron_index][synapse_index].x; input.ignore();
    input >> synapses.synapseCoord[neuron_index][synapse_index].y; input.ignore();
    input >> synapses.deltaT[neuron_index][synapse_index]; input.ignore();
    input >> synapses.W[neuron_index][synapse_index]; input.ignore();
    input >> synapses.psr[neuron_index][synapse_index]; input.ignore();
    input >> synapses.decay[neuron_index][synapse_index]; input.ignore();
    input >> synapses.total_delay[neuron_index][synapse_index]; input.ignore();
    input >> synapses.delayQueue[neuron_index][synapse_index][0]; input.ignore();
    input >> synapses.delayIdx[neuron_index][synapse_index]; input.ignore();
    input >> synapses.ldelayQueue[neuron_index][synapse_index]; input.ignore();
    input >> synapse_type; input.ignore();
    input >> synapses.tau[neuron_index][synapse_index]; input.ignore();
    input >> synapses.r[neuron_index][synapse_index]; input.ignore();
    input >> synapses.u[neuron_index][synapse_index]; input.ignore();
    input >> synapses.D[neuron_index][synapse_index]; input.ignore();
    input >> synapses.U[neuron_index][synapse_index]; input.ignore();
    input >> synapses.F[neuron_index][synapse_index]; input.ignore();
    input >> synapses.lastSpike[neuron_index][synapse_index]; input.ignore();

    synapses.type[neuron_index][synapse_index] = synapseOrdinalToType(synapse_type);
}

/**
 *  Initializes the queues for the Synapses.
 *  @param  synapses    synapse list to find the indexed synapse from.
 *  @param  neuron_index    index of the neuron that the synapse belongs to.
 *  @param  synapse_index   index of the synapse to set.
 */
void LIFModel::initSpikeQueue(AllSynapses &synapses, const int neuron_index, const int synapse_index)
{
    int &total_delay = synapses.total_delay[neuron_index][synapse_index];
    uint32_t &delayQueue = synapses.delayQueue[neuron_index][synapse_index][0];
    int &delayIdx = synapses.delayIdx[neuron_index][synapse_index];
    int &ldelayQueue = synapses.ldelayQueue[neuron_index][synapse_index];

    size_t size = total_delay / ( sizeof(uint8_t) * 8 ) + 1;
    assert( size <= BYTES_OF_DELAYQUEUE );
    delayQueue = 0;
    delayIdx = 0;
    ldelayQueue = LENGTH_OF_DELAYQUEUE;
}

/**
 *  Reset time varying state vars and recompute decay.
 *  @param  synapses    synapse list to find the indexed synapse from.
 *  @param  neuron_index    index of the neuron that the synapse belongs to.
 *  @param  synapse_index   index of the synapse to set.
 */
void LIFModel::resetSynapse(AllSynapses &synapses, const int neuron_index, const int synapse_index)
{
    synapses.psr[neuron_index][synapse_index] = 0.0;
    assert( updateDecay(synapses, neuron_index, synapse_index) );
    synapses.u[neuron_index][synapse_index] = DEFAULT_U;
    synapses.r[neuron_index][synapse_index] = 1.0;
    synapses.lastSpike[neuron_index][synapse_index] = ULONG_MAX;
}

/**
 *  Write the simulation's memory image.
 *  @param  output  the filestream to write.
 *  @param  neurons the neuron list to search from.
 *  @param  synapses    the synapse list to search from.
 *  @param  simulation_step the step of the simulation at the current time.
 */
void LIFModel::saveMemory(ostream& output, AllNeurons &neurons, AllSynapses &synapses, BGFLOAT simulation_step)
{
    // write the neurons data
    output << neurons.size;
    for (int i = 0; i < neurons.size; i++) {
        writeNeuron(output, neurons, i);
    }

    // write the synapse data
    int synapse_count = 0;
    for (int i = 0; i < neurons.size; i++) {
        synapse_count += synapses.synapse_counts[i];
    }
    output << synapse_count;

    for (int neuron_index = 0; neuron_index < neurons.size; neuron_index++) {
        for (size_t synapse_index = 0; synapse_index < synapses.synapse_counts[neuron_index]; synapse_index++) {
            writeSynapse(output, synapses, neuron_index, synapse_index);
        }
    }

    // write the final radii
    for (int i = 0; i < neurons.size; i++) {
        output << m_conns->radiiHistory(simulation_step, i);
    }

    // write the final rates
    for (int i = 0; i < neurons.size; i++) {
        output << m_conns->ratesHistory(simulation_step, i);
    }

    output << flush;
}

/**
 *  Writes out the data in the selected Neuron.
 *  @param  output  stream to write out to.
 *  @param  neurons the neuron list to search from.
 *  @param  index   the index of the Neuron to output data from.
 */
void LIFModel::writeNeuron(ostream& output, AllNeurons &neurons, const int index) const {
    output << neurons.deltaT[index] << ends;
    output << neurons.Cm[index] << ends;
    output << neurons.Rm[index] << ends;
    output << neurons.Vthresh[index] << ends;
    output << neurons.Vrest[index] << ends;
    output << neurons.Vreset[index] << ends;
    output << neurons.Vinit[index] << ends;
    output << neurons.Trefract[index] << ends;
    output << neurons.Inoise[index] << ends;
    output << neurons.Iinject[index] << ends;
    output << neurons.Isyn[index] << ends;
    output << neurons.nStepsInRefr[index] << ends;
    output << neurons.C1[index] << ends;
    output << neurons.C2[index] << ends;
    output << neurons.I0[index] << ends;
    output << neurons.Vm[index] << ends;
    output << neurons.hasFired[index] << ends;
    output << neurons.Tau[index] << ends;
}

/**
 *  Write the synapse data to the stream.
 *  @param  output  stream to print out to.
 *  @param  synapses    synapse list to find the indexed synapse from.
 *  @param  neuron_index    index of the neuron that the synapse belongs to.
 *  @param  synapse_index   index of the synapse to print out.
 */
void LIFModel::writeSynapse(ostream& output, AllSynapses &synapses, const int neuron_index, const int synapse_index) const {
    output << synapses.summationCoord[neuron_index][synapse_index].x << ends;
    output << synapses.summationCoord[neuron_index][synapse_index].y << ends;
    output << synapses.synapseCoord[neuron_index][synapse_index].x << ends;
    output << synapses.synapseCoord[neuron_index][synapse_index].y << ends;
    output << synapses.deltaT[neuron_index][synapse_index] << ends;
    output << synapses.W[neuron_index][synapse_index] << ends;
    output << synapses.psr[neuron_index][synapse_index] << ends;
    output << synapses.decay[neuron_index][synapse_index] << ends;
    output << synapses.total_delay[neuron_index][synapse_index] << ends;
    output << synapses.delayQueue[neuron_index][synapse_index][0] << ends;
    output << synapses.delayIdx[neuron_index][synapse_index] << ends;
    output << synapses.ldelayQueue[neuron_index][synapse_index] << ends;
    output << synapses.type[neuron_index][synapse_index] << ends;
    output << synapses.tau[neuron_index][synapse_index] << ends;
    output << synapses.r[neuron_index][synapse_index] << ends;
    output << synapses.u[neuron_index][synapse_index] << ends;
    output << synapses.D[neuron_index][synapse_index] << ends;
    output << synapses.U[neuron_index][synapse_index] << ends;
    output << synapses.F[neuron_index][synapse_index] << ends;
    output << synapses.lastSpike[neuron_index][synapse_index] << ends;
}

/**
 *  Prepares a stream with data from the model and Neurons.
 *  @param  output  stream to print out to.
 *  @param  neurons the Neuron list to search from.
 *  @param  sim_info    SimulationInfo class to read information from.
 */
void LIFModel::saveState(ostream &output, const AllNeurons &neurons, const SimulationInfo &sim_info)
{
    output << "   " << m_conns->radiiHistory.toXML("radiiHistory") << endl;
    output << "   " << m_conns->ratesHistory.toXML("ratesHistory") << endl;
    output << "   " << m_conns->burstinessHist.toXML("burstinessHist") << endl;
    output << "   " << m_conns->spikesHistory.toXML("spikesHistory") << endl;

    output << "   " << m_conns->xloc.toXML("xloc") << endl;
    output << "   " << m_conns->yloc.toXML("yloc") << endl;

    //Write Neuron Types
    VectorMatrix neuronTypes("complete", "const", 1, sim_info.totalNeurons, EXC);
    for (int i = 0; i < sim_info.totalNeurons; i++) {
        neuronTypes[i] = neurons.neuron_type_map[i];
    }
    output << "   " << neuronTypes.toXML("neuronTypes") << endl;

    int num_starter_neurons = m_frac_starter_neurons * sim_info.totalNeurons;
    if (num_starter_neurons > 0) {
        VectorMatrix starterNeuronsM("complete", "const", 1, num_starter_neurons);
        getStarterNeuronMatrix(starterNeuronsM, neurons.starter_map, sim_info);
        output << "   " << starterNeuronsM.toXML("starterNeurons") << endl;
    }

    // Write neuron threshold
    // neuron threshold
    VectorMatrix neuronThresh("complete", "const", 1, sim_info.totalNeurons, 0);
    for (int i = 0; i < sim_info.totalNeurons; i++) {
        neuronThresh[i] = neurons.Vthresh[i];
    }
    output << "   " << neuronThresh.toXML("neuronThresh") << endl;
}

/**
 *  Get starter Neuron matrix.
 *  @param  matrix   Starter Neuron matrix.
 *  @param  starter_map bool map to reference neuron matrix location from.
 *  @param  sim_info    SimulationInfo class to read information from.
 */
void LIFModel::getStarterNeuronMatrix(VectorMatrix& matrix, const bool* starter_map, const SimulationInfo &sim_info)
{
    int cur = 0;
    for (int x = 0; x < sim_info.width; x++) {
        for (int y = 0; y < sim_info.height; y++) {
            if (starter_map[x + y * sim_info.width]) {
                matrix[cur] = x + y * sim_info.height;
                cur++;
            }
        }
    }
}

/**
 *  Creates all the Neurons and generates data for them.
 *  @param  neurons the Neuron list to search from.
 *  @param  sim_info    SimulationInfo class to read information from.
 */
void LIFModel::createAllNeurons(AllNeurons &neurons, const SimulationInfo &sim_info)
{
    DEBUG(cout << "\nAllocating neurons..." << endl;)

    generateNeuronTypeMap(neurons.neuron_type_map, sim_info.totalNeurons);
    initStarterMap(neurons.starter_map, sim_info.totalNeurons, neurons.neuron_type_map);
    
    /* set their specific types */
    for (int neuron_index = 0; neuron_index < sim_info.totalNeurons; neuron_index++) {
        setNeuronDefaults(neurons, neuron_index);
        
        // set the neuron info for neurons
        neurons.Iinject[neuron_index] = rng.inRange(m_Iinject[0], m_Iinject[1]);
        neurons.Inoise[neuron_index] = rng.inRange(m_Inoise[0], m_Inoise[1]);
        neurons.Vthresh[neuron_index] = rng.inRange(m_Vthresh[0], m_Vthresh[1]);
        neurons.Vrest[neuron_index] = rng.inRange(m_Vresting[0], m_Vresting[1]);
        neurons.Vreset[neuron_index] = rng.inRange(m_Vreset[0], m_Vreset[1]);
        neurons.Vinit[neuron_index] = rng.inRange(m_Vinit[0], m_Vinit[1]);
        neurons.Vm[neuron_index] = neurons.Vinit[neuron_index];
        neurons.deltaT[neuron_index] = sim_info.deltaT;

        updateNeuron(neurons, neuron_index);

        int max_spikes = (int) ((sim_info.epochDuration * m_growth.maxRate * sim_info.maxSteps));
        neurons.spike_history[neuron_index] = new uint64_t[max_spikes];
        for (int j = 0; j < max_spikes; ++j) {
            neurons.spike_history[neuron_index][j] = -1;
        }

        switch (neurons.neuron_type_map[neuron_index]) {
            case INH:
                DEBUG_MID(cout << "setting inhibitory neuron: "<< neuron_index << endl;)
                // set inhibitory absolute refractory period
                neurons.Trefract[neuron_index] = DEFAULT_InhibTrefract;// TODO(derek): move defaults inside model.
                break;
                
            case EXC:
                DEBUG_MID(cout << "setting exitory neuron: " << neuron_index << endl;)
                // set excitory absolute refractory period
                neurons.Trefract[neuron_index] = DEFAULT_ExcitTrefract;
                break;
                
            default:
                DEBUG_MID(cout << "ERROR: unknown neuron type: " << neurons.neuron_type_map[neuron_index] << "@" << neuron_index << endl;)
                assert(false);
                break;
        }
        // endogenously_active_neuron_map -> Model State
        if (neurons.starter_map[neuron_index]) {
            // set endogenously active threshold voltage, reset voltage, and refractory period
            neurons.Vthresh[neuron_index] = rng.inRange(m_starter_Vthresh[0], m_starter_Vthresh[1]);
            neurons.Vreset[neuron_index] = rng.inRange(m_starter_Vreset[0], m_starter_Vreset[1]);
            neurons.Trefract[neuron_index] = DEFAULT_ExcitTrefract; // TODO(derek): move defaults inside model.
        }

        DEBUG_HI(cout << "CREATE NEURON[" << neuron_index << "] {" << endl
                << "\tVm = " << neurons.Vm[neuron_index] << endl
                << "\tVthresh = " << neurons.Vthresh[neuron_index] << endl
                << "\tI0 = " << neurons.I0[neuron_index] << endl
                << "\tInoise = " << neurons.Inoise[neuron_index] << "from : (" << m_Inoise[0] << "," << m_Inoise[1] << ")" << endl
                << "\tC1 = " << neurons.C1[neuron_index] << endl
                << "\tC2 = " << neurons.C2[neuron_index] << endl
                << "}" << endl
        ;)
    }
    
    DEBUG(cout << "Done initializing neurons..." << endl;)
}

/**
 *  Creates a randomly ordered distribution with the specified numbers of neuron types.
 *  @param  neuron_types    array of the types of neurons to have in the map.
 *  @param  num_neurons number of the neurons to have in the type map.
 *  @return a flat vector (to map to 2-d [x,y] = [i % m_width, i / m_width])
 */
void LIFModel::generateNeuronTypeMap(neuronType neuron_types[], int num_neurons)
{
    //TODO: m_pInhibitoryNeuronLayout
    int num_inhibitory_neurons = m_inhibitory_neuron_layout.size();
	int num_excititory_neurons = num_neurons - num_inhibitory_neurons;    
    DEBUG(cout << "\nInitializing neuron type map"<< endl;);
    
    for (int i = 0; i < num_neurons; i++) {
        neuron_types[i] = EXC;
    }
    
    if (m_fixed_layout) {
        DEBUG(cout << "Total neurons: " << num_neurons << endl;)
        DEBUG(cout << "Inhibitory Neurons: " << num_inhibitory_neurons << endl;)
        DEBUG(cout << "Excitatory Neurons: " << num_excititory_neurons << endl;)
        
        for (int i = 0; i < num_inhibitory_neurons; i++) {
            neuron_types[m_inhibitory_neuron_layout.at(i)] = INH;
        }
    } else {
        int num_excititory_neurons = (int) (m_frac_excititory_neurons * num_neurons + 0.5);
        int num_inhibitory_neurons = num_neurons - num_excititory_neurons;
        DEBUG(cout << "Total neurons: " << num_neurons << endl;)
        DEBUG(cout << "Inhibitory Neurons: " << num_inhibitory_neurons << endl;)
        DEBUG(cout << "Excitatory Neurons: " << num_inhibitory_neurons << endl;)
        
        DEBUG(cout << endl << "Randomly selecting inhibitory neurons..." << endl;)
        
        int* rg_inhibitory_layout = new int[num_inhibitory_neurons];
        
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
            neuron_types[rg_inhibitory_layout[i]] = INH;
        }
        delete[] rg_inhibitory_layout;
    }
    
    DEBUG(cout << "Done initializing neuron type map" << endl;);
}

/**
 *  Populates the starter map.
 *  Selects \e numStarter excitory neurons and converts them into starter neurons.
 *  @param  starter_map booleam array of neurons to initiate.
 *  @param  num_neurons number of neurons to have in the map.
 *  @param  neuron_type_map array of neuronTypes to set the starter map to.
 */
void LIFModel::initStarterMap(bool *starter_map, const int num_neurons, const neuronType neuron_type_map[])
{
    for (int i = 0; i < num_neurons; i++) {
        starter_map[i] = false;
    }
    
    if (!STARTER_FLAG) {
        for (int i = 0; i < num_neurons; i++) {
            starter_map[i] = false;
        }
        return;
    }

    if (m_fixed_layout) {
        size_t num_endogenously_active_neurons = m_endogenously_active_neuron_list.size();
        for (size_t i = 0; i < num_endogenously_active_neurons; i++) {
            starter_map[m_endogenously_active_neuron_list.at(i)] = true;
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
            if (neuron_type_map[i] == EXC && starter_map[i] == false) {
                starter_map[i] = true;
                starters_allocated++;
                DEBUG(cout << "allocated EA neuron at random index [" << i << "]" << endl;);
            }
        }

        DEBUG(cout <<"Done randomly initializing starter map\n\n";)
    }
}

/**
 *  Set the Neuron at the indexed location to default values.
 *  @param  synapses    synapse list to find the indexed synapse from.
 *  @param  neuron_index    index of the Neuron that the synapse belongs to.
 */
void LIFModel::setNeuronDefaults(AllNeurons &neurons, const int index)
{
    neurons.deltaT[index] = DEFAULT_dt;
    neurons.Cm[index] = DEFAULT_Cm;
    neurons.Rm[index] = DEFAULT_Rm;
    neurons.Vthresh[index] = DEFAULT_Vthresh;
    neurons.Vrest[index] = DEFAULT_Vrest;
    neurons.Vreset[index] = DEFAULT_Vreset;
    neurons.Vinit[index] = DEFAULT_Vreset;
    neurons.Trefract[index] = DEFAULT_Trefract;
    neurons.Inoise[index] = DEFAULT_Inoise;
    neurons.Iinject[index] = DEFAULT_Iinject;
    neurons.Tau[index] = DEFAULT_Cm * DEFAULT_Rm;
}

/**
 *  Sets up the Simulation.
 *  @param  num_neurons number of Neurons to have in the simulation.
 *  @param  sim_info    SimulationInfo class to read information from.
 */
void LIFModel::setupSim(const int num_neurons, const SimulationInfo &sim_info)
{
    if (m_conns != NULL) {
        delete m_conns;
        m_conns = NULL;
    }

    m_conns = new Connections(num_neurons, m_growth.startRadius, sim_info.epochDuration, sim_info.maxSteps);
    // Initialize neuron locations
    for (int i = 0; i < num_neurons; i++) {
        m_conns->xloc[i] = i % sim_info.width;
        m_conns->yloc[i] = i / sim_info.width;
    }

    // calculate the distance between neurons
    for (int n = 0; n < num_neurons - 1; n++)
    {
        for (int n2 = n + 1; n2 < num_neurons; n2++)
        {
            // distance^2 between two points in point-slope form
            m_conns->dist2(n, n2) = (m_conns->xloc[n] - m_conns->xloc[n2]) * (m_conns->xloc[n] - m_conns->xloc[n2]) +
                (m_conns->yloc[n] - m_conns->yloc[n2]) * (m_conns->yloc[n] - m_conns->yloc[n2]);

            // both points are equidistant from each other
            m_conns->dist2(n2, n) = m_conns->dist2(n, n2);
        }
    }

    // take the square root to get actual distance (Pythagoras was right!)
    // (The CompleteMatrix class makes this assignment look so easy...)
    m_conns->dist = sqrt(m_conns->dist2);

    // Init connection frontier distance change matrix with the current distances
    m_conns->delta = m_conns->dist;

}

/**
 *  TODO
 *  @param  neurons the Neuron list to search from.
 *  @param  src_coord   the coordinate of the source Neuron.
 *  @param  dest_coord  the coordinate of the destination Neuron.
 *  @param  width   TODO
 */
synapseType LIFModel::synType(AllNeurons &neurons, Coordinate src_coord, Coordinate dest_coord, const int width)
{
    return synType(neurons, src_coord.x + src_coord.y * width, dest_coord.x + dest_coord.y * width);
}

/**
 *  Returns an appropriate synapseType object for the given integer.
 *  @param  type_ordinal    integer that correspond with a synapseType.
 *  @return the SynapseType that corresponds with the given integer.
 */
synapseType LIFModel::synapseOrdinalToType(const int type_ordinal)
{
    switch (type_ordinal) {
        case 0:
            return II;
        case 1:
            return IE;
        case 2:
            return EI;
        case 3:
            return EE;
        default:
            return STYPE_UNDEF;
    }
}

/**
* Returns the type of synapse at the given coordinates
* @param    neurons the list of all Neurons .
* @param    src_neuron  integer that points to a Neuron in the type map as a source.
* @param    dest_neuron integer that points to a Neuron in the type map as a destination.
* @return   type of synapse at the given coordinate or -1 on error
*/
synapseType LIFModel::synType(AllNeurons &neurons, const int src_neuron, const int dest_neuron)
{
    if ( neurons.neuron_type_map[src_neuron] == INH && neurons.neuron_type_map[dest_neuron] == INH )
        return II;
    else if ( neurons.neuron_type_map[src_neuron] == INH && neurons.neuron_type_map[dest_neuron] == EXC )
        return IE;
    else if ( neurons.neuron_type_map[src_neuron] == EXC && neurons.neuron_type_map[dest_neuron] == INH )
        return EI;
    else if ( neurons.neuron_type_map[src_neuron] == EXC && neurons.neuron_type_map[dest_neuron] == EXC )
        return EE;

    return STYPE_UNDEF;
}

/**
 *  Get the sign of the synapseType.
 *  @param    type    synapseType I to I, I to E, E to I, or E to E
 *  @return   1 or -1, or 0 if error
 */
int LIFModel::synSign(const synapseType type)
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

/**
 *  Output operator.
 *  @param  out stream to write to.
 *  @param  params  parameters to output.
 */
ostream& operator<<(ostream &out, const LIFModel::GrowthParams &params) {
    out << "epsilon: " << params.epsilon
        << " beta: " << params.beta
        << " rho: " << params.rho
        << " targetRate: " << params.targetRate
        << " maxRate: " << params.maxRate
        << " minRadius: " << params.minRadius
        << " startRadius" << params.startRadius;
    return out;
}


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
const string LIFModel::Connections::MATRIX_TYPE = "complete";
// TODO comment
const string LIFModel::Connections::MATRIX_INIT = "const";
/* ------------------- ERROR ------------------- *\
 * terminate called after throwing an instance of 'std::bad_alloc'
 *	what():  St9bad_alloc 
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
 *	- W		- radii
 * 	- rates		- dist2
 * 	- delta		- dist
 * 	- areai
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
LIFModel::Connections::Connections(const int num_neurons, const BGFLOAT start_radius, const BGFLOAT growthEpochDuration, const BGFLOAT maxGrowthSteps) :
    xloc(MATRIX_TYPE, MATRIX_INIT, 1, num_neurons),
    yloc(MATRIX_TYPE, MATRIX_INIT, 1, num_neurons),
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
    ratesHistory(MATRIX_TYPE, MATRIX_INIT, static_cast<int>(maxGrowthSteps + 1), num_neurons),
    burstinessHist(MATRIX_TYPE, MATRIX_INIT, 1, (int)(growthEpochDuration * maxGrowthSteps), 0),
    spikesHistory(MATRIX_TYPE, MATRIX_INIT, 1, (int)(growthEpochDuration * maxGrowthSteps * 100), 0)
{
    // Init radii and rates history matrices with current radii and rates
    for (int i = 0; i < num_neurons; i++) {
        radiiHistory(0, i) = start_radius;
        ratesHistory(0, i) = 0;
    }
}




