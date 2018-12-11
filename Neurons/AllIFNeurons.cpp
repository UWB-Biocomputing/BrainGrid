#include "AllIFNeurons.h"
#include "ParseParamError.h"

// Default constructor
AllIFNeurons::AllIFNeurons() : AllSpikingNeurons()
{
}

// Copy constructor
AllIFNeurons::AllIFNeurons(const AllIFNeurons &r_neurons) : AllSpikingNeurons(r_neurons)
{
    copyParameters(dynamic_cast<const AllIFNeurons &>(r_neurons));
}

AllIFNeurons::~AllIFNeurons()
{
    cleanupNeurons();
}

/*
 *  Assignment operator: copy neurons parameters.
 *
 *  @param  r_neurons  Neurons class object to copy from.
 */
IAllNeurons &AllIFNeurons::operator=(const IAllNeurons &r_neurons)
{
    copyParameters(dynamic_cast<const AllIFNeurons &>(r_neurons));

    return (*this);
}

/*
 *  Copy neurons parameters.
 *
 *  @param  r_neurons  Neurons class object to copy from.
 */
void AllIFNeurons::copyParameters(const AllIFNeurons &r_neurons)
{
    AllSpikingNeurons::copyParameters(r_neurons);
 
    for (int i = 0; i < 2; i++) {
        m_Iinject[i] = r_neurons.m_Iinject[i];
        m_Inoise[i] = r_neurons.m_Inoise[i];
        m_Vthresh[i] = r_neurons.m_Vthresh[i];
        m_Vresting[i] = r_neurons.m_Vresting[i];
        m_Vreset[i] = r_neurons.m_Vreset[i];
        m_Vinit[i] = r_neurons.m_Vinit[i];
        m_starter_Vthresh[i] = r_neurons.m_starter_Vthresh[i];
        m_starter_Vreset[i] = r_neurons.m_starter_Vreset[i];
    }
}

/*
 *  Setup the internal structure of the class (allocate memories).
 *
 *  @param  sim_info  SimulationInfo class to read information from.
 *  @param  clr_info  ClusterInfo class to read information from.
 */
void AllIFNeurons::setupNeurons(SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    setupNeuronsInternalState(sim_info, clr_info);

    // allocate neurons properties data
    m_pNeuronsProperties = new AllIFNeuronsProperties();
    m_pNeuronsProperties->setupNeuronsProperties(sim_info, clr_info);
}

/*
 *  Setup the internal structure of the class.
 *
 *  @param  sim_info  SimulationInfo class to read information from.
 *  @param  clr_info  ClusterInfo class to read information from.
 */
void AllIFNeurons::setupNeuronsInternalState(SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    AllSpikingNeurons::setupNeuronsInternalState(sim_info, clr_info);
}

/*
 *  Cleanup the class (deallocate memories).
 */
void AllIFNeurons::cleanupNeurons()
{
    // deallocate neurons properties data
    delete m_pNeuronsProperties;
    m_pNeuronsProperties = NULL;

    cleanupNeuronsInternalState();
}

/*
 *  Deallocate all resources
 */
void AllIFNeurons::cleanupNeuronsInternalState()
{
    AllSpikingNeurons::cleanupNeuronsInternalState();
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
 *  @param  clr_info    ClusterInfo class to read information from.
 */
void AllIFNeurons::createAllNeurons(SimulationInfo *sim_info, Layout *layout, ClusterInfo *clr_info)
{
    /* set their specific types */
    for (int neuron_index = 0; neuron_index < clr_info->totalClusterNeurons; neuron_index++) {
        setNeuronDefaults(neuron_index);

        // set the neuron info for neurons
        createNeuron(sim_info, neuron_index, layout, clr_info);
    }
}

/*
 *  Creates a single Neuron and generates data for it.
 *
 *  @param  sim_info     SimulationInfo class to read information from.
 *  @param  neuron_index Index of the neuron to create.
 *  @param  layout       Layout information of the neunal network.
 *  @param  clr_info     ClusterInfo class to read information from.
 */
void AllIFNeurons::createNeuron(SimulationInfo *sim_info, int neuron_index, Layout *layout, ClusterInfo *clr_info)
{
    AllIFNeuronsProperties *pNeuronsProperties = dynamic_cast<AllIFNeuronsProperties*>(m_pNeuronsProperties);
    BGFLOAT &Iinject = pNeuronsProperties->Iinject[neuron_index];
    BGFLOAT &Inoise = pNeuronsProperties->Inoise[neuron_index];
    BGFLOAT &Vthresh = pNeuronsProperties->Vthresh[neuron_index];
    BGFLOAT &Vrest = pNeuronsProperties->Vrest[neuron_index];
    BGFLOAT &Vreset = pNeuronsProperties->Vreset[neuron_index];
    BGFLOAT &Vinit = pNeuronsProperties->Vinit[neuron_index];
    BGFLOAT &Vm = pNeuronsProperties->Vm[neuron_index];
    uint64_t *&spike_history = pNeuronsProperties->spike_history[neuron_index];
    BGFLOAT &Trefract = pNeuronsProperties->Trefract[neuron_index];
    BGFLOAT &I0 = pNeuronsProperties->I0[neuron_index];
    BGFLOAT &C1 = pNeuronsProperties->C1[neuron_index];
    BGFLOAT &C2 = pNeuronsProperties->C2[neuron_index];

    // set the neuron info for neurons
    Iinject = rng.inRange(m_Iinject[0], m_Iinject[1]);
    Inoise = rng.inRange(m_Inoise[0], m_Inoise[1]);
    Vthresh = rng.inRange(m_Vthresh[0], m_Vthresh[1]);
    Vrest = rng.inRange(m_Vresting[0], m_Vresting[1]);
    Vreset = rng.inRange(m_Vreset[0], m_Vreset[1]);
    Vinit = rng.inRange(m_Vinit[0], m_Vinit[1]);
    Vm = Vinit;

    initNeuronConstsFromParamValues(neuron_index, sim_info->deltaT);

    int max_spikes = (int) ((sim_info->epochDuration * sim_info->maxFiringRate));
    spike_history = new uint64_t[max_spikes];
    for (int j = 0; j < max_spikes; ++j) {
        spike_history[j] = ULONG_MAX;
    }

    int neuron_layout_index = clr_info->clusterNeuronsBegin + neuron_index;
    switch (layout->neuron_type_map[neuron_layout_index]) {
        case INH:
            DEBUG_MID(cout << "setting inhibitory neuron: "<< neuron_layout_index << endl;)
            // set inhibitory absolute refractory period
            Trefract = DEFAULT_InhibTrefract;// TODO(derek): move defaults inside model.
            break;

        case EXC:
            DEBUG_MID(cout << "setting exitory neuron: " << neuron_layout_index << endl;)
            // set excitory absolute refractory period
            Trefract = DEFAULT_ExcitTrefract;
            break;

        default:
            DEBUG_MID(cout << "ERROR: unknown neuron type: " << layout->neuron_type_map[neuron_layout_index] << "@" << neuron_layout_index << endl;)
            assert(false);
            break;
    }
    // endogenously_active_neuron_map -> Model State
    if (layout->starter_map[neuron_layout_index]) {
        // set endogenously active threshold voltage, reset voltage, and refractory period
        Vthresh = rng.inRange(m_starter_Vthresh[0], m_starter_Vthresh[1]);
        Vreset = rng.inRange(m_starter_Vreset[0], m_starter_Vreset[1]);
        Trefract= DEFAULT_ExcitTrefract; // TODO(derek): move defaults inside model.
    }

    DEBUG_HI(cout << "CREATE NEURON[" << neuron_layout_index << "] {" << endl
            << "\tVm = " << Vm << endl
            << "\tVthresh = " << Vthresh<< endl
            << "\tI0 = " << I0 << endl
            << "\tInoise = " << Inoise << "from : (" << m_Inoise[0] << "," << m_Inoise[1] << ")" << endl
            << "\tC1 = " << C1 << endl
            << "\tC2 = " << C2 << endl
            << "}" << endl
    ;)
}

/*
 *  Set the Neuron at the indexed location to default values.
 *
 *Inoise  @param  neuron_index    Index of the Neuron that the synapse belongs to.
 */
void AllIFNeurons::setNeuronDefaults(const int index)
{
    AllIFNeuronsProperties *pNeuronsProperties = dynamic_cast<AllIFNeuronsProperties*>(m_pNeuronsProperties);
    BGFLOAT &Cm = pNeuronsProperties->Cm[index];
    BGFLOAT &Rm = pNeuronsProperties->Rm[index];
    BGFLOAT &Vthresh = pNeuronsProperties->Vthresh[index];
    BGFLOAT &Vrest = pNeuronsProperties->Vrest[index];
    BGFLOAT &Vreset = pNeuronsProperties->Vreset[index];
    BGFLOAT &Vinit = pNeuronsProperties->Vinit[index];
    BGFLOAT &Trefract = pNeuronsProperties->Trefract[index];
    BGFLOAT &Inoise = pNeuronsProperties->Inoise[index];
    BGFLOAT &Iinject = pNeuronsProperties->Iinject[index];
    BGFLOAT &Tau = pNeuronsProperties->Tau[index];

    Cm = DEFAULT_Cm;
    Rm = DEFAULT_Rm;
    Vthresh = DEFAULT_Vthresh;
    Vrest = DEFAULT_Vrest;
    Vreset = DEFAULT_Vreset;
    Vinit = DEFAULT_Vreset;
    Trefract = DEFAULT_Trefract;
    Inoise = DEFAULT_Inoise;
    Iinject = DEFAULT_Iinject;
    Tau = DEFAULT_Cm * DEFAULT_Rm;
}

/*
 *  Initializes the Neuron constants at the indexed location.
 *
 *  @param  neuron_index    Index of the Neuron.
 *  @param  deltaT          Inner simulation step duration
 */
void AllIFNeurons::initNeuronConstsFromParamValues(int neuron_index, const BGFLOAT deltaT)
{
        AllIFNeuronsProperties *pNeuronsProperties = dynamic_cast<AllIFNeuronsProperties*>(m_pNeuronsProperties);
        BGFLOAT &Tau = pNeuronsProperties->Tau[neuron_index];
        BGFLOAT &C1 = pNeuronsProperties->C1[neuron_index];
        BGFLOAT &C2 = pNeuronsProperties->C2[neuron_index];
        BGFLOAT &Rm = pNeuronsProperties->Rm[neuron_index];
        BGFLOAT &I0 = pNeuronsProperties->I0[neuron_index];
        BGFLOAT &Iinject = pNeuronsProperties->Iinject[neuron_index];
        BGFLOAT &Vrest = pNeuronsProperties->Vrest[neuron_index];

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
    AllIFNeuronsProperties *pNeuronsProperties = dynamic_cast<AllIFNeuronsProperties*>(m_pNeuronsProperties);
    BGFLOAT &Cm = pNeuronsProperties->Cm[i];
    BGFLOAT &Rm = pNeuronsProperties->Rm[i];
    BGFLOAT &Vthresh = pNeuronsProperties->Vthresh[i];
    BGFLOAT &Vrest = pNeuronsProperties->Vrest[i];
    BGFLOAT &Vreset = pNeuronsProperties->Vreset[i];
    BGFLOAT &Vinit = pNeuronsProperties->Vinit[i];
    BGFLOAT &Trefract = pNeuronsProperties->Trefract[i];
    BGFLOAT &Inoise = pNeuronsProperties->Inoise[i];
    BGFLOAT &Iinject = pNeuronsProperties->Iinject[i];
    int &nStepsInRefr = pNeuronsProperties->nStepsInRefr[i];
    BGFLOAT &Vm = pNeuronsProperties->Vm[i];
    bool &hasFired = pNeuronsProperties->hasFired[i];
    BGFLOAT &C1 = pNeuronsProperties->C1[i];
    BGFLOAT &C2 = pNeuronsProperties->C2[i];
    BGFLOAT &I0 = pNeuronsProperties->I0[i];

    stringstream ss;
    ss << "Cm: " << Cm << " "; // membrane capacitance
    ss << "Rm: " << Rm << " "; // membrane resistance
    ss << "Vthresh: " << Vthresh << " "; // if Vm exceeds, Vthresh, a spike is emitted
    ss << "Vrest: " << Vrest << " "; // the resting membrane voltage
    ss << "Vreset: " << Vreset << " "; // The voltage to reset Vm to after a spike
    ss << "Vinit: " << Vinit << endl; // The initial condition for V_m at t=0
    ss << "Trefract: " << Trefract << " "; // the number of steps in the refractory period
    ss << "Inoise: " << Inoise << " "; // the stdev of the noise to be added each delta_t
    ss << "Iinject: " << Iinject << " "; // A constant current to be injected into the LIF neuron
    ss << "nStepsInRefr: " << nStepsInRefr << endl; // the number of steps left in the refractory period
    ss << "Vm: " << Vm << " "; // the membrane voltage
    ss << "hasFired: " << hasFired << " "; // it done fired?
    ss << "C1: " << C1 << " ";
    ss << "C2: " << C2 << " ";
    ss << "I0: " << I0 << " ";
    return ss.str( );
}

/*
 *  Sets the data for Neurons to input's data.
 *
 *  @param  input       istream to read from.
 *  @param  clr_info    used as a reference to set info for neuronss.
 */
void AllIFNeurons::deserialize(istream &input, const ClusterInfo *clr_info)
{
    for (int i = 0; i < clr_info->totalClusterNeurons; i++) {
        readNeuron(input, i);
    }
}

/*
 *  Sets the data for Neuron #index to input's data.
 *
 *  @param  input       istream to read from.
 *  @param  sim_info    used as a reference to set info for neurons.
 *  @param  i           index of the neuron (in neurons).
 */
void AllIFNeurons::readNeuron(istream &input, int i)
{
    AllIFNeuronsProperties *pNeuronsProperties = dynamic_cast<AllIFNeuronsProperties*>(m_pNeuronsProperties);
    BGFLOAT &Cm = pNeuronsProperties->Cm[i];
    BGFLOAT &Rm = pNeuronsProperties->Rm[i];
    BGFLOAT &Vthresh = pNeuronsProperties->Vthresh[i];
    BGFLOAT &Vrest = pNeuronsProperties->Vrest[i];
    BGFLOAT &Vreset = pNeuronsProperties->Vreset[i];
    BGFLOAT &Vinit = pNeuronsProperties->Vinit[i];
    BGFLOAT &Trefract = pNeuronsProperties->Trefract[i];
    BGFLOAT &Inoise = pNeuronsProperties->Inoise[i];
    BGFLOAT &Iinject = pNeuronsProperties->Iinject[i];
    BGFLOAT &Isyn = pNeuronsProperties->Isyn[i];
    int &nStepsInRefr = pNeuronsProperties->nStepsInRefr[i];
    BGFLOAT &C1 = pNeuronsProperties->C1[i];
    BGFLOAT &C2 = pNeuronsProperties->C2[i];
    BGFLOAT &I0 = pNeuronsProperties->I0[i];
    BGFLOAT &Vm = pNeuronsProperties->Vm[i];
    bool &hasFired = pNeuronsProperties->hasFired[i];
    BGFLOAT &Tau = pNeuronsProperties->Tau[i];

    // input.ignore() so input skips over end-of-line characters.
    input >> Cm; input.ignore();
    input >> Rm; input.ignore();
    input >> Vthresh; input.ignore();
    input >> Vrest; input.ignore();
    input >> Vreset; input.ignore();
    input >> Vinit; input.ignore();
    input >> Trefract; input.ignore();
    input >> Inoise; input.ignore();
    input >> Iinject; input.ignore();
    input >> Isyn; input.ignore();
    input >> nStepsInRefr; input.ignore();
    input >> C1; input.ignore();
    input >> C2; input.ignore();
    input >> I0; input.ignore();
    input >> Vm; input.ignore();
    input >> hasFired; input.ignore();
    input >> Tau; input.ignore();
}

/*
 *  Writes out the data in Neurons.
 *
 *  @param  output      stream to write out to.
 *  @param  clr_info    used as a reference to set info for neuronss.
 */
void AllIFNeurons::serialize(ostream& output, const ClusterInfo *clr_info) const 
{
    for (int i = 0; i < clr_info->totalClusterNeurons; i++) {
        writeNeuron(output, i);
    }
}

/*
 *  Writes out the data in the selected Neuron.
 *
 *  @param  output      stream to write out to.
 *  @param  i           index of the neuron (in neurons).
 */
void AllIFNeurons::writeNeuron(ostream& output, int i) const
{
    AllIFNeuronsProperties *pNeuronsProperties = dynamic_cast<AllIFNeuronsProperties*>(m_pNeuronsProperties);
    BGFLOAT &Cm = pNeuronsProperties->Cm[i];
    BGFLOAT &Rm = pNeuronsProperties->Rm[i];
    BGFLOAT &Vthresh = pNeuronsProperties->Vthresh[i];
    BGFLOAT &Vrest = pNeuronsProperties->Vrest[i];
    BGFLOAT &Vreset = pNeuronsProperties->Vreset[i];
    BGFLOAT &Vinit = pNeuronsProperties->Vinit[i];
    BGFLOAT &Trefract = pNeuronsProperties->Trefract[i];
    BGFLOAT &Inoise = pNeuronsProperties->Inoise[i];
    BGFLOAT &Iinject = pNeuronsProperties->Iinject[i];
    BGFLOAT &Isyn = pNeuronsProperties->Isyn[i];
    int &nStepsInRefr = pNeuronsProperties->nStepsInRefr[i];
    BGFLOAT &C1 = pNeuronsProperties->C1[i];
    BGFLOAT &C2 = pNeuronsProperties->C2[i];
    BGFLOAT &I0 = pNeuronsProperties->I0[i];
    BGFLOAT &Vm = pNeuronsProperties->Vm[i];
    bool &hasFired = pNeuronsProperties->hasFired[i];
    BGFLOAT &Tau = pNeuronsProperties->Tau[i];

    output << Cm << ends;
    output << Rm << ends;
    output << Vthresh << ends;
    output << Vrest << ends;
    output << Vreset << ends;
    output << Vinit << ends;
    output << Trefract << ends;
    output << Inoise << ends;
    output << Iinject << ends;
    output << Isyn << ends;
    output << nStepsInRefr << ends;
    output << C1 << ends;
    output << C2 << ends;
    output << I0 << ends;
    output << Vm << ends;
    output << hasFired << ends;
    output << Tau << ends;
}
