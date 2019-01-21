#include "AllIFNeuronsProperties.h"
#include "ParseParamError.h"

#if !defined(USEGPU)

// Default constructor
AllIFNeuronsProperties::AllIFNeuronsProperties()
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

AllIFNeuronsProperties::~AllIFNeuronsProperties()
{
    cleanupNeuronsProperties();
}

/*
 *  Setup the internal structure of the class (allocate memories).
 *
 *  @param  sim_info  SimulationInfo class to read information from.
 *  @param  clr_info  ClusterInfo class to read information from.
 */
void AllIFNeuronsProperties::setupNeuronsProperties(SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    AllSpikingNeuronsProperties::setupNeuronsProperties(sim_info, clr_info);

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
void AllIFNeuronsProperties::cleanupNeuronsProperties()
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

#else // USE_GPU

// Default constructor
AllIFNeuronsProperties::AllIFNeuronsProperties()
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

AllIFNeuronsProperties::~AllIFNeuronsProperties()
{
}

/*
 *  Setup the internal structure of the class (allocate memories).
 *
 *  @param  pAllNeuronsProperties_d the AllNeuronsProperties on device memory.
 *  @param  sim_info  SimulationInfo class to read information from.
 *  @param  clr_info  ClusterInfo class to read information from.
 */
static void AllIFNeuronsProperties::setupNeuronsProperties(void *pAllNeuronsProperties_d, SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    AllIFNeuronsProperties allNeuronsProperties;

    // allocate GPU memories to store all neuron's states
    allocNeuronsProperties(allNeuronsProperties, sim_info, clr_info);

    // copy the pointer address to structure on device memory
    checkCudaErrors( cudaMemcpy ( pAllNeuronsDeviceProperties_d, &allNeuronsProperties, sizeof( AllIFNeuronsProperties ), cudaMemcpyHostToDevice ) );
}

/*
 *  Allocate GPU memories to store all neurons' states.
 *
 *  @param  allNeuronsProperties   Reference to the AllNeuronsProperties struct.
 *  @param  sim_info  SimulationInfo class to read information from.
 *  @param  clr_info               ClusterInfo to refer from.
 */
static void AllIFNeuronsProperties::allocNeuronsProperties(AllIFNeuronsProperties &allNeuronsProperties, SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    int size = clr_info->totalClusterNeurons;

    AllSpikingNeuronsProperties::allocNeuronsProperties(allNeuronsProperties, sim_info, clr_info);

    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProperties.C1, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProperties.C2, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProperties.Cm, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProperties.I0, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProperties.Iinject, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProperties.Inoise, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProperties.Isyn, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProperties.Rm, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProperties.Tau, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProperties.Trefract, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProperties.Vinit, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProperties.Vm, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProperties.Vreset, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProperties.Vrest, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProperties.Vthresh, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProperties.nStepsInRefr, size * sizeof( int ) ) );
}

/*
 *  Cleanup the class (deallocate memories).
 *
 *  @param  pAllNeuronsProperties_d the AllNeuronsProperties on device memory.
 *  @param  clr_info               ClusterInfo to refer from.
 */
static void AllIFNeuronsProperties::cleanupNeuronsProperties(void *pAllNeuronsProperties_d, ClusterInfo *clr_info)
{
    AllIFNeuronsProperties allNeuronsProperties;

    checkCudaErrors( cudaMemcpy ( &allNeuronsProperties, pAllNeuronsProperties_d, sizeof( AllIFNeuronsProperties ), cudaMemcpyDeviceToHost ) );

    deleteNeuronsProperties(allNeuronsProperties);

    checkCudaErrors( cudaFree( pAllNeuronsProperties_d ) );
}

/*
 *  Delete GPU memories.
 *
 *  @param  allNeuronsProperties   Reference to the AllNeuronsProperties struct.
 *  @param  clr_info               ClusterInfo to refer from.
 */
static void AllIFNeuronsProperties::deleteNeuronsProperties(AllIFNeuronsProperties &allNeuronsProperties, ClusterInfo *clr_info)
{
    int size = clr_info->totalClusterNeurons;

    checkCudaErrors( cudaFree( allNeuronsProperties.C1 ) );
    checkCudaErrors( cudaFree( allNeuronsProperties.C2 ) );
    checkCudaErrors( cudaFree( allNeuronsProperties.Cm ) );
    checkCudaErrors( cudaFree( allNeuronsProperties.I0 ) );
    checkCudaErrors( cudaFree( allNeuronsProperties.Iinject ) );
    checkCudaErrors( cudaFree( allNeuronsProperties.Inoise ) );
    checkCudaErrors( cudaFree( allNeuronsProperties.Isyn ) );
    checkCudaErrors( cudaFree( allNeuronsProperties.Rm ) );
    checkCudaErrors( cudaFree( allNeuronsProperties.Tau ) );
    checkCudaErrors( cudaFree( allNeuronsProperties.Trefract ) );
    checkCudaErrors( cudaFree( allNeuronsProperties.Vinit ) );
    checkCudaErrors( cudaFree( allNeuronsProperties.Vm ) );
    checkCudaErrors( cudaFree( allNeuronsProperties.Vreset ) );
    checkCudaErrors( cudaFree( allNeuronsProperties.Vrest ) );
    checkCudaErrors( cudaFree( allNeuronsProperties.Vthresh ) );
    checkCudaErrors( cudaFree( allNeuronsProperties.nStepsInRefr ) );

    AllSpikingNeuronsProperties::deleteNeuronsProperties(allNeuronsProperties);
}

#endif // USE_GPU

/*
 *  Checks the number of required parameters.
 *
 * @return true if all required parameters were successfully read, false otherwise.
 */
bool AllIFNeuronsProperties::checkNumParameters()
{
    return (nParams >= 8);
}

/*
 *  Attempts to read parameters from a XML file.
 *
 *  @param  element TiXmlElement to examine.
 *  @return true if successful, false otherwise.
 */
bool AllIFNeuronsProperties::readParameters(const TiXmlElement& element)
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
        if(element.ValueStr().compare("min") == 0){
            m_Iinject[0] = atof(element.GetText());
        }
        else if(element.ValueStr().compare("max") == 0){
            m_Iinject[1] = atof(element.GetText());
        }

        return true;
    }

    if (element.Parent()->ValueStr().compare("Inoise") == 0) {
        if(element.ValueStr().compare("min") == 0){
            m_Inoise[0] = atof(element.GetText());
        }
        else if(element.ValueStr().compare("max") == 0){
            m_Inoise[1] = atof(element.GetText());
        }

        return true;
    }

    if (element.Parent()->ValueStr().compare("Vthresh") == 0) {
        if(element.ValueStr().compare("min") == 0){
            m_Vthresh[0] = atof(element.GetText());
        }
        else if(element.ValueStr().compare("max") == 0){
            m_Vthresh[1] = atof(element.GetText());
        }

        return true;
    }

    if (element.Parent()->ValueStr().compare("Vresting") == 0) {
        if(element.ValueStr().compare("min") == 0){
            m_Vresting[0] = atof(element.GetText());
        }
        else if(element.ValueStr().compare("max") == 0){
            m_Vresting[1] = atof(element.GetText());
        }
        return true;
    }

    if (element.Parent()->ValueStr().compare("Vreset") == 0) {
        if(element.ValueStr().compare("min") == 0){
            m_Vreset[0] = atof(element.GetText());
        }
        else if(element.ValueStr().compare("max") == 0){
            m_Vreset[1] = atof(element.GetText());
        }
        return true;
    }

    if (element.Parent()->ValueStr().compare("Vinit") == 0) {
        if(element.ValueStr().compare("min") == 0){
            m_Vinit[0] = atof(element.GetText());
        }
        else if(element.ValueStr().compare("max") == 0){
            m_Vinit[1] = atof(element.GetText());
        }
        return true;
    }

    if (element.Parent()->ValueStr().compare("starter_vthresh") == 0) {
        if(element.ValueStr().compare("min") == 0){
            m_starter_Vthresh[0] = atof(element.GetText());
        }
        else if(element.ValueStr().compare("max") == 0){
            m_starter_Vthresh[1] = atof(element.GetText());
        }
        return true;
    }

    if (element.Parent()->ValueStr().compare("starter_vreset") == 0) {
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
void AllIFNeuronsProperties::printParameters(ostream &output) const
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
 *  Copy neurons parameters.
 *
 *  @param  r_neurons  Neurons class object to copy from.
 */
void AllIFNeuronsProperties::copyParameters(const AllNeuronsProperties *r_neuronsProps)
{
    const AllIFNeuronsProperties *pProps = dynamic_cast<const AllIFNeuronsProperties*>(r_neuronsProps);

    for (int i = 0; i < 2; i++) {
        m_Iinject[i] = pProps->m_Iinject[i];
        m_Inoise[i] = pProps->m_Inoise[i];
        m_Vthresh[i] = pProps->m_Vthresh[i];
        m_Vresting[i] = pProps->m_Vresting[i];
        m_Vreset[i] = pProps->m_Vreset[i];
        m_Vinit[i] = pProps->m_Vinit[i];
        m_starter_Vthresh[i] = pProps->m_starter_Vthresh[i];
        m_starter_Vreset[i] = pProps->m_starter_Vreset[i];
    }
}

/*
 *  Sets the data for Neuron #index to input's data.
 *
 *  @param  input       istream to read from.
 *  @param  i           index of the neuron (in neurons).
 */
void AllIFNeuronsProperties::readNeuronProperties(istream &input, int i)
{
    BGFLOAT &Cm = this->Cm[i];
    BGFLOAT &Rm = this->Rm[i];
    BGFLOAT &Vthresh = this->Vthresh[i];
    BGFLOAT &Vrest = this->Vrest[i];
    BGFLOAT &Vreset = this->Vreset[i];
    BGFLOAT &Vinit = this->Vinit[i];
    BGFLOAT &Trefract = this->Trefract[i];
    BGFLOAT &Inoise = this->Inoise[i];
    BGFLOAT &Iinject = this->Iinject[i];
    BGFLOAT &Isyn = this->Isyn[i];
    int &nStepsInRefr = this->nStepsInRefr[i];
    BGFLOAT &C1 = this->C1[i];
    BGFLOAT &C2 = this->C2[i];
    BGFLOAT &I0 = this->I0[i];
    BGFLOAT &Vm = this->Vm[i];
    bool &hasFired = this->hasFired[i];
    BGFLOAT &Tau = this->Tau[i];

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
 *  Writes out the data in the selected Neuron.
 *
 *  @param  output      stream to write out to.
 *  @param  i           index of the neuron (in neurons).
 */
void AllIFNeuronsProperties::writeNeuronProperties(ostream& output, int i) const
{
    BGFLOAT &Cm = this->Cm[i];
    BGFLOAT &Rm = this->Rm[i];
    BGFLOAT &Vthresh = this->Vthresh[i];
    BGFLOAT &Vrest = this->Vrest[i];
    BGFLOAT &Vreset = this->Vreset[i];
    BGFLOAT &Vinit = this->Vinit[i];
    BGFLOAT &Trefract = this->Trefract[i];
    BGFLOAT &Inoise = this->Inoise[i];
    BGFLOAT &Iinject = this->Iinject[i];
    BGFLOAT &Isyn = this->Isyn[i];
    int &nStepsInRefr = this->nStepsInRefr[i];
    BGFLOAT &C1 = this->C1[i];
    BGFLOAT &C2 = this->C2[i];
    BGFLOAT &I0 = this->I0[i];
    BGFLOAT &Vm = this->Vm[i];
    bool &hasFired = this->hasFired[i];
    BGFLOAT &Tau = this->Tau[i];

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

/*
 *  Creates a single Neuron and generates data for it.
 *
 *  @param  sim_info     SimulationInfo class to read information from.
 *  @param  neuron_index Index of the neuron to create.
 *  @param  layout       Layout information of the neunal network.
 *  @param  clr_info     ClusterInfo class to read information from.
 */
void AllIFNeuronsProperties::setNeuronPropertyValues(SimulationInfo *sim_info, int neuron_index, Layout *layout, ClusterInfo *clr_info)
{
    BGFLOAT &Iinject = this->Iinject[neuron_index];
    BGFLOAT &Inoise = this->Inoise[neuron_index];
    BGFLOAT &Vthresh = this->Vthresh[neuron_index];
    BGFLOAT &Vrest = this->Vrest[neuron_index];
    BGFLOAT &Vreset = this->Vreset[neuron_index];
    BGFLOAT &Vinit = this->Vinit[neuron_index];
    BGFLOAT &Vm = this->Vm[neuron_index];
    uint64_t *&spike_history = this->spike_history[neuron_index];
    BGFLOAT &Trefract = this->Trefract[neuron_index];
    BGFLOAT &I0 = this->I0[neuron_index];
    BGFLOAT &C1 = this->C1[neuron_index];
    BGFLOAT &C2 = this->C2[neuron_index];

    // set the neuron info for neurons
    Iinject = rng.inRange(m_Iinject[0], m_Iinject[1]);
    Inoise = rng.inRange(m_Inoise[0], m_Inoise[1]);
    Vthresh = rng.inRange(m_Vthresh[0], m_Vthresh[1]);
    Vrest = rng.inRange(m_Vresting[0], m_Vresting[1]);
    Vreset = rng.inRange(m_Vreset[0], m_Vreset[1]);
    Vinit = rng.inRange(m_Vinit[0], m_Vinit[1]);
    Vm = Vinit;

    initNeuronPropertyConstsFromParamValues(neuron_index, sim_info->deltaT);

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
void AllIFNeuronsProperties::setNeuronPropertyDefaults(const int index)
{
    BGFLOAT &Cm = this->Cm[index];
    BGFLOAT &Rm = this->Rm[index];
    BGFLOAT &Vthresh = this->Vthresh[index];
    BGFLOAT &Vrest = this->Vrest[index];
    BGFLOAT &Vreset = this->Vreset[index];
    BGFLOAT &Vinit = this->Vinit[index];
    BGFLOAT &Trefract = this->Trefract[index];
    BGFLOAT &Inoise = this->Inoise[index];
    BGFLOAT &Iinject = this->Iinject[index];
    BGFLOAT &Tau = this->Tau[index];

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
void AllIFNeuronsProperties::initNeuronPropertyConstsFromParamValues(int neuron_index, const BGFLOAT deltaT)
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
