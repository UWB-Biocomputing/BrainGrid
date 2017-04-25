#include "AllIFNeurons.h"
#include "ParseParamError.h"

// Default constructor
AllIFNeurons::AllIFNeurons() : AllSpikingNeurons()
{
    C1 = NULL;
    C2 = NULL;
    Cm = NULL;
    I0 = NULL;
    Iinject = NULL;
    Inoise = NULL;
    Isyn = NULL;
    Rm = NULL;
    Tau = NULL;
    Trefract = NULL;
    Vinit = NULL;
    Vm = NULL;
    Vreset = NULL;
    Vrest = NULL;
    Vthresh = NULL;
    nStepsInRefr = NULL;
}

AllIFNeurons::~AllIFNeurons()
{
    freeResources();
}

/*
 *  Setup the internal structure of the class (allocate memories).
 *
 *  @param  sim_info  SimulationInfo class to read information from.
 */
void AllIFNeurons::setupNeurons(SimulationInfo *sim_info)
{
    AllSpikingNeurons::setupNeurons(sim_info);

    // TODO: Rename variables for easier identification
    C1 = new BGFLOAT[size];
    C2 = new BGFLOAT[size];
    Cm = new BGFLOAT[size];
    I0 = new BGFLOAT[size];
    Iinject = new BGFLOAT[size];
    Inoise = new BGFLOAT[size];
    Isyn = new BGFLOAT[size];
    Rm = new BGFLOAT[size];
    Tau = new BGFLOAT[size];
    Trefract = new BGFLOAT[size];
    Vinit = new BGFLOAT[size];
    Vm = new BGFLOAT[size];
    Vreset = new BGFLOAT[size];
    Vrest = new BGFLOAT[size];
    Vthresh = new BGFLOAT[size];
    nStepsInRefr = new int[size];

    for (int i = 0; i < size; ++i) {
        nStepsInRefr[i] = 0;
    }
}

/*
 *  Cleanup the class (deallocate memories).
 */
void AllIFNeurons::cleanupNeurons()
{
    freeResources();
    AllSpikingNeurons::cleanupNeurons();
}

/**
 *  Deallocate all resources
 */
void AllIFNeurons::freeResources()
{
    if (size != 0) {
        delete[] C1;
        delete[] C2;
        delete[] Cm;
        delete[] I0;
        delete[] Iinject;
        delete[] Inoise;
        delete[] Isyn;
        delete[] Rm;
        delete[] Tau;
        delete[] Trefract;
        delete[] Vinit;
        delete[] Vm;
        delete[] Vreset;
        delete[] Vrest;
        delete[] Vthresh;
        delete[] nStepsInRefr;
    }

    C1 = NULL;
    C2 = NULL;
    Cm = NULL;
    I0 = NULL;
    Iinject = NULL;
    Inoise = NULL;
    Isyn = NULL;
    Rm = NULL;
    Tau = NULL;
    Trefract = NULL;
    Vinit = NULL;
    Vm = NULL;
    Vreset = NULL;
    Vrest = NULL;
    Vthresh = NULL;
    nStepsInRefr = NULL;
}

/*
 *  Checks the number of required parameters.
 *
 * @return true if all required parameters were successfully read, false otherwise.
 */
bool AllIFNeurons::checkNumParameters()
{
    return (nParams >= 8);
}

/*
 *  Attempts to read parameters from a XML file.
 *
 *  @param  element TiXmlElement to examine.
 *  @return true if successful, false otherwise.
 */
bool AllIFNeurons::readParameters(const TiXmlElement& element)
{
    if (element.ValueStr().compare("Iinject") == 0         ||
	element.ValueStr().compare("Inoise") == 0          ||
    	element.ValueStr().compare("Vthresh") == 0         ||
    	element.ValueStr().compare("Vresting") == 0        ||
    	element.ValueStr().compare("Vreset") == 0          ||
	element.ValueStr().compare("Vinit") == 0           ||
	element.ValueStr().compare("starter_vthresh") == 0 ||
	element.ValueStr().compare("starter_vreset") == 0    )     {
	nParams++;
	return true;
    }

    if (element.Parent()->ValueStr().compare("Iinject") == 0) {
/*
        if (element.QueryFLOATAttribute("min", &m_Iinject[0]) != TIXML_SUCCESS) {
            throw ParseParamError("Iinject min", "Iinject missing minimum value in XML.");
        }
        if (m_Iinject[0] < 0) {
            throw ParseParamError("Iinject min", "Invalid negative Iinject value.");
        }
        if (element.QueryFLOATAttribute("max", &m_Iinject[1]) != TIXML_SUCCESS) {
            throw ParseParamError("Iinject max", "Iinject missing maximum value in XML.");
        }
        if (m_Iinject[0] > m_Iinject[1]) {
            throw ParseParamError("Iinject max", "Invalid range for Iinject value.");
        }
	
        nParams++;
*/
	if(element.ValueStr().compare("min") == 0){
            m_Iinject[0] = atof(element.GetText());
        }
	else if(element.ValueStr().compare("max") == 0){
            m_Iinject[1] = atof(element.GetText());
        }

        return true;
    }

    if (element.Parent()->ValueStr().compare("Inoise") == 0) {
/*
        if (element.QueryFLOATAttribute("min", &m_Inoise[0]) != TIXML_SUCCESS) {
            throw ParseParamError("Inoise min", "Inoise missing minimum value in XML.");
        }
        if (m_Inoise[0] < 0) {
            throw ParseParamError("Inoise min", "Invalid negative Inoise value.");
        }
        if (element.QueryFLOATAttribute("max", &m_Inoise[1]) != TIXML_SUCCESS) {
            throw ParseParamError("Inoise max", "Inoise missing maximum value in XML.");
        }
        if (m_Inoise[0] > m_Inoise[1]) {
            throw ParseParamError("Inoise max", "Invalid range for Inoise value.");
        }
        nParams++;
*/
	if(element.ValueStr().compare("min") == 0){
            m_Inoise[0] = atof(element.GetText());
        }
	else if(element.ValueStr().compare("max") == 0){
            m_Inoise[1] = atof(element.GetText());
        }

        return true;
    }

    if (element.Parent()->ValueStr().compare("Vthresh") == 0) {
/*
        if (element.QueryFLOATAttribute("min", &m_Vthresh[0]) != TIXML_SUCCESS) {
            throw ParseParamError("Vthresh min", "Vthresh missing minimum value in XML.");
        }
        if (m_Vthresh[0] < 0) {
            throw ParseParamError("Vthresh min", "Invalid negative Vthresh value.");
        }
        if (element.QueryFLOATAttribute("max", &m_Vthresh[1]) != TIXML_SUCCESS) {
            throw ParseParamError("Vthresh max", "Vthresh missing maximum value in XML.");
        }
        if (m_Vthresh[0] > m_Vthresh[1]) {
            throw ParseParamError("Vthresh max", "Invalid range for Vthresh value.");
        }
        nParams++;
*/	
	if(element.ValueStr().compare("min") == 0){
            m_Vthresh[0] = atof(element.GetText());
        }
	else if(element.ValueStr().compare("max") == 0){
            m_Vthresh[1] = atof(element.GetText());
        }

        return true;
    }

    if (element.Parent()->ValueStr().compare("Vresting") == 0) {
/*
        if (element.QueryFLOATAttribute("min", &m_Vresting[0]) != TIXML_SUCCESS) {
            throw ParseParamError("Vresting min", "Vresting missing minimum value in XML.");
        }
        if (element.QueryFLOATAttribute("max", &m_Vresting[1]) != TIXML_SUCCESS) {
            throw ParseParamError("Vresting max", "Vresting missing maximum value in XML.");
        }
        if (m_Vresting[0] > m_Vresting[1]) {
            throw ParseParamError("Vresting max", "Invalid range for Vresting value.");
        }
        nParams++;
*/
	if(element.ValueStr().compare("min") == 0){
            m_Vresting[0] = atof(element.GetText());
        }
	else if(element.ValueStr().compare("max") == 0){
            m_Vresting[1] = atof(element.GetText());
        }
        return true;
    }

    if (element.Parent()->ValueStr().compare("Vreset") == 0) {
/*
        if (element.QueryFLOATAttribute("min", &m_Vreset[0]) != TIXML_SUCCESS) {
            throw ParseParamError("Vreset min", "Vreset missing minimum value in XML.");
        }
        if (element.QueryFLOATAttribute("max", &m_Vreset[1]) != TIXML_SUCCESS) {
            throw ParseParamError("Vreset max", "Vreset missing maximum value in XML.");
        }
        if (m_Vreset[0] > m_Vreset[1]) {
            throw ParseParamError("Vreset max", "Invalid range for Vreset value.");
        }
        nParams++;
*/
	if(element.ValueStr().compare("min") == 0){
            m_Vreset[0] = atof(element.GetText());
        }
	else if(element.ValueStr().compare("max") == 0){
            m_Vreset[1] = atof(element.GetText());
        }
        return true;
    }

    if (element.Parent()->ValueStr().compare("Vinit") == 0) {
/*
        if (element.QueryFLOATAttribute("min", &m_Vinit[0]) != TIXML_SUCCESS) {
            throw ParseParamError("Vinit min", "Vinit missing minimum value in XML.");
        }
        if (element.QueryFLOATAttribute("max", &m_Vinit[1]) != TIXML_SUCCESS) {
            throw ParseParamError("Vinit max", "Vinit missing maximum value in XML.");
        }
        if (m_Vinit[0] > m_Vinit[1]) {
            throw ParseParamError("Vinit max", "Invalid range for Vinit value.");
        }
        nParams++;
*/
	if(element.ValueStr().compare("min") == 0){
            m_Vinit[0] = atof(element.GetText());
        }
	else if(element.ValueStr().compare("max") == 0){
            m_Vinit[1] = atof(element.GetText());
        }
        return true;
    }

    if (element.Parent()->ValueStr().compare("starter_vthresh") == 0) {
/*
        if (element.QueryFLOATAttribute("min", &m_starter_Vthresh[0]) != TIXML_SUCCESS) {
            throw ParseParamError("starter_vthresh min", "starter_vthresh missing minimum value in XML.");
        }
        if (m_starter_Vthresh[0] < 0) {
            throw ParseParamError("starter_vthresh min", "Invalid negative starter_vthresh value.");
        }
        if (element.QueryFLOATAttribute("max", &m_starter_Vthresh[1]) != TIXML_SUCCESS) {
            throw ParseParamError("starter_vthresh max", "starter_vthresh missing maximum value in XML.");
        }
        if (m_starter_Vthresh[0] > m_starter_Vthresh[1]) {
            throw ParseParamError("starter_vthresh max", "Invalid range for starter_vthresh value.");
        }
        nParams++;
*/
	if(element.ValueStr().compare("min") == 0){
            m_starter_Vthresh[0] = atof(element.GetText());
        }
	else if(element.ValueStr().compare("max") == 0){
            m_starter_Vthresh[1] = atof(element.GetText());
        }
        return true;
    }

    if (element.Parent()->ValueStr().compare("starter_vreset") == 0) {
/*
        if (element.QueryFLOATAttribute("min", &m_starter_Vreset[0]) != TIXML_SUCCESS) {
            throw ParseParamError("starter_vreset min", "starter_vreset missing minimum value in XML.");
        }
        if (m_starter_Vreset[0] < 0) {
            throw ParseParamError("starter_vreset min", "Invalid negative starter_vreset value.");
        }
        if (element.QueryFLOATAttribute("max", &m_starter_Vreset[1]) != TIXML_SUCCESS) {
            throw ParseParamError("starter_vreset max", "starter_vreset missing maximum value in XML.");
        }
        if (m_starter_Vreset[0] > m_starter_Vreset[1]) {
            throw ParseParamError("starter_vreset max", "Invalid range for starter_vreset value.");
        }
        nParams++;
*/
	if(element.ValueStr().compare("min") == 0){
            m_starter_Vreset[0] = atof(element.GetText());
        }
	else if(element.ValueStr().compare("max") == 0){
            m_starter_Vreset[1] = atof(element.GetText());
        }
        return true;
    }

    return false;
}

/*
 *  Prints out all parameters of the neurons to ostream.
 * 
 *  @param  output  ostream to send output to.
 */
void AllIFNeurons::printParameters(ostream &output) const
{
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
}

/*
 *  Creates all the Neurons and generates data for them.
 *
 *  @param  sim_info    SimulationInfo class to read information from.
 *  @param  layout      Layout information of the neunal network.
 */
void AllIFNeurons::createAllNeurons(SimulationInfo *sim_info, Layout *layout)
{
    /* set their specific types */
    for (int neuron_index = 0; neuron_index < sim_info->totalNeurons; neuron_index++) {
        setNeuronDefaults(neuron_index);

        // set the neuron info for neurons
        createNeuron(sim_info, neuron_index, layout);
    }
}

/*
 *  Creates a single Neuron and generates data for it.
 *
 *  @param  sim_info     SimulationInfo class to read information from.
 *  @param  neuron_index Index of the neuron to create.
 *  @param  layout       Layout information of the neunal network.
 */
void AllIFNeurons::createNeuron(SimulationInfo *sim_info, int neuron_index, Layout *layout)
{
    // set the neuron info for neurons
    Iinject[neuron_index] = rng.inRange(m_Iinject[0], m_Iinject[1]);
    Inoise[neuron_index] = rng.inRange(m_Inoise[0], m_Inoise[1]);
    Vthresh[neuron_index] = rng.inRange(m_Vthresh[0], m_Vthresh[1]);
    Vrest[neuron_index] = rng.inRange(m_Vresting[0], m_Vresting[1]);
    Vreset[neuron_index] = rng.inRange(m_Vreset[0], m_Vreset[1]);
    Vinit[neuron_index] = rng.inRange(m_Vinit[0], m_Vinit[1]);
    Vm[neuron_index] = Vinit[neuron_index];

    initNeuronConstsFromParamValues(neuron_index, sim_info->deltaT);

    int max_spikes = (int) ((sim_info->epochDuration * sim_info->maxFiringRate));
    spike_history[neuron_index] = new uint64_t[max_spikes];
    for (int j = 0; j < max_spikes; ++j) {
        spike_history[neuron_index][j] = ULONG_MAX;
    }

    switch (layout->neuron_type_map[neuron_index]) {
        case INH:
            DEBUG_MID(cout << "setting inhibitory neuron: "<< neuron_index << endl;)
            // set inhibitory absolute refractory period
            Trefract[neuron_index] = DEFAULT_InhibTrefract;// TODO(derek): move defaults inside model.
            break;

        case EXC:
            DEBUG_MID(cout << "setting exitory neuron: " << neuron_index << endl;)
            // set excitory absolute refractory period
            Trefract[neuron_index] = DEFAULT_ExcitTrefract;
            break;

        default:
            DEBUG_MID(cout << "ERROR: unknown neuron type: " << layout->neuron_type_map[neuron_index] << "@" << neuron_index << endl;)
            assert(false);
            break;
    }
    // endogenously_active_neuron_map -> Model State
    if (layout->starter_map[neuron_index]) {
        // set endogenously active threshold voltage, reset voltage, and refractory period
        Vthresh[neuron_index] = rng.inRange(m_starter_Vthresh[0], m_starter_Vthresh[1]);
        Vreset[neuron_index] = rng.inRange(m_starter_Vreset[0], m_starter_Vreset[1]);
        Trefract[neuron_index] = DEFAULT_ExcitTrefract; // TODO(derek): move defaults inside model.
    }

    DEBUG_HI(cout << "CREATE NEURON[" << neuron_index << "] {" << endl
            << "\tVm = " << Vm[neuron_index] << endl
            << "\tVthresh = " << Vthresh[neuron_index] << endl
            << "\tI0 = " << I0[neuron_index] << endl
            << "\tInoise = " << Inoise[neuron_index] << "from : (" << m_Inoise[0] << "," << m_Inoise[1] << ")" << endl
            << "\tC1 = " << C1[neuron_index] << endl
            << "\tC2 = " << C2[neuron_index] << endl
            << "}" << endl
    ;)
}

/*
 *  Set the Neuron at the indexed location to default values.
 *
 *  @param  neuron_index    Index of the Neuron that the synapse belongs to.
 */
void AllIFNeurons::setNeuronDefaults(const int index)
{
    Cm[index] = DEFAULT_Cm;
    Rm[index] = DEFAULT_Rm;
    Vthresh[index] = DEFAULT_Vthresh;
    Vrest[index] = DEFAULT_Vrest;
    Vreset[index] = DEFAULT_Vreset;
    Vinit[index] = DEFAULT_Vreset;
    Trefract[index] = DEFAULT_Trefract;
    Inoise[index] = DEFAULT_Inoise;
    Iinject[index] = DEFAULT_Iinject;
    Tau[index] = DEFAULT_Cm * DEFAULT_Rm;
}

/*
 *  Initializes the Neuron constants at the indexed location.
 *
 *  @param  neuron_index    Index of the Neuron.
 *  @param  deltaT          Inner simulation step duration
 */
void AllIFNeurons::initNeuronConstsFromParamValues(int neuron_index, const BGFLOAT deltaT)
{
        BGFLOAT &Tau = this->Tau[neuron_index];
        BGFLOAT &C1 = this->C1[neuron_index];
        BGFLOAT &C2 = this->C2[neuron_index];
        BGFLOAT &Rm = this->Rm[neuron_index];
        BGFLOAT &I0 = this->I0[neuron_index];
        BGFLOAT &Iinject = this->Iinject[neuron_index];
        BGFLOAT &Vrest = this->Vrest[neuron_index];

        /* init consts C1,C2 for exponential Euler integration */
        if (Tau > 0) {
                C1 = exp( -deltaT / Tau );
                C2 = Rm * ( 1 - C1 );
        } else {
                C1 = 0.0;
                C2 = Rm;
        }
        /* calculate const IO */
        if (Rm > 0) {
                I0 = Iinject + Vrest / Rm;
        }else {
                assert(false);
        }
}

/*
 *  Outputs state of the neuron chosen as a string.
 *
 *  @param  i   index of the neuron (in neurons) to output info from.
 *  @return the complete state of the neuron.
 */
string AllIFNeurons::toString(const int i) const
{
    stringstream ss;
    ss << "Cm: " << Cm[i] << " "; // membrane capacitance
    ss << "Rm: " << Rm[i] << " "; // membrane resistance
    ss << "Vthresh: " << Vthresh[i] << " "; // if Vm exceeds, Vthresh, a spike is emitted
    ss << "Vrest: " << Vrest[i] << " "; // the resting membrane voltage
    ss << "Vreset: " << Vreset[i] << " "; // The voltage to reset Vm to after a spike
    ss << "Vinit: " << Vinit[i] << endl; // The initial condition for V_m at t=0
    ss << "Trefract: " << Trefract[i] << " "; // the number of steps in the refractory period
    ss << "Inoise: " << Inoise[i] << " "; // the stdev of the noise to be added each delta_t
    ss << "Iinject: " << Iinject[i] << " "; // A constant current to be injected into the LIF neuron
    ss << "nStepsInRefr: " << nStepsInRefr[i] << endl; // the number of steps left in the refractory period
    ss << "Vm: " << Vm[i] << " "; // the membrane voltage
    ss << "hasFired: " << hasFired[i] << " "; // it done fired?
    ss << "C1: " << C1[i] << " ";
    ss << "C2: " << C2[i] << " ";
    ss << "I0: " << I0[i] << " ";
    return ss.str( );
}

/*
 *  Sets the data for Neurons to input's data.
 *
 *  @param  input       istream to read from.
 *  @param  sim_info    used as a reference to set info for neuronss.
 */
void AllIFNeurons::deserialize(istream &input, const SimulationInfo *sim_info)
{
    for (int i = 0; i < sim_info->totalNeurons; i++) {
        readNeuron(input, sim_info, i);
    }
}

/*
 *  Sets the data for Neuron #index to input's data.
 *
 *  @param  input       istream to read from.
 *  @param  sim_info    used as a reference to set info for neurons.
 *  @param  i           index of the neuron (in neurons).
 */
void AllIFNeurons::readNeuron(istream &input, const SimulationInfo *sim_info, int i)
{
    // input.ignore() so input skips over end-of-line characters.
    input >> Cm[i]; input.ignore();
    input >> Rm[i]; input.ignore();
    input >> Vthresh[i]; input.ignore();
    input >> Vrest[i]; input.ignore();
    input >> Vreset[i]; input.ignore();
    input >> Vinit[i]; input.ignore();
    input >> Trefract[i]; input.ignore();
    input >> Inoise[i]; input.ignore();
    input >> Iinject[i]; input.ignore();
    input >> Isyn[i]; input.ignore();
    input >> nStepsInRefr[i]; input.ignore();
    input >> C1[i]; input.ignore();
    input >> C2[i]; input.ignore();
    input >> I0[i]; input.ignore();
    input >> Vm[i]; input.ignore();
    input >> hasFired[i]; input.ignore();
    input >> Tau[i]; input.ignore();
}

/*
 *  Writes out the data in Neurons.
 *
 *  @param  output      stream to write out to.
 *  @param  sim_info    used as a reference to set info for neuronss.
 */
void AllIFNeurons::serialize(ostream& output, const SimulationInfo *sim_info) const 
{
    for (int i = 0; i < sim_info->totalNeurons; i++) {
        writeNeuron(output, sim_info, i);
    }
}

/*
 *  Writes out the data in the selected Neuron.
 *
 *  @param  output      stream to write out to.
 *  @param  sim_info    used as a reference to set info for neuronss.
 *  @param  i           index of the neuron (in neurons).
 */
void AllIFNeurons::writeNeuron(ostream& output, const SimulationInfo *sim_info, int i) const
{
    output << Cm[i] << ends;
    output << Rm[i] << ends;
    output << Vthresh[i] << ends;
    output << Vrest[i] << ends;
    output << Vreset[i] << ends;
    output << Vinit[i] << ends;
    output << Trefract[i] << ends;
    output << Inoise[i] << ends;
    output << Iinject[i] << ends;
    output << Isyn[i] << ends;
    output << nStepsInRefr[i] << ends;
    output << C1[i] << ends;
    output << C2[i] << ends;
    output << I0[i] << ends;
    output << Vm[i] << ends;
    output << hasFired[i] << ends;
    output << Tau[i] << ends;
}
