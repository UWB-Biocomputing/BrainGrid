#include "AllIFNeuronsProps.h"
#include "ParseParamError.h"
#if defined(USE_GPU)
#include <helper_cuda.h>
#endif

// Default constructor
AllIFNeuronsProps::AllIFNeuronsProps()
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

AllIFNeuronsProps::~AllIFNeuronsProps()
{
    cleanupNeuronsProps();
}

/*
 *  Setup the internal structure of the class (allocate memories).
 *
 *  @param  sim_info  SimulationInfo class to read information from.
 *  @param  clr_info  ClusterInfo class to read information from.
 */
void AllIFNeuronsProps::setupNeuronsProps(SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    AllSpikingNeuronsProps::setupNeuronsProps(sim_info, clr_info);

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
void AllIFNeuronsProps::cleanupNeuronsProps()
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


#if defined(USE_GPU)
/*
 *  Allocate GPU memories to store all neurons' states,
 *  and copy them from host to GPU memory.
 *
 *  @param  allNeuronsDeviceProps   Reference to the AllIZHNeuronsProps class on device memory.
 *  @param  sim_info                SimulationInfo to refer from.
 *  @param  clr_info                ClusterInfo to refer from.
 */
void AllIFNeuronsProps::setupNeuronsDeviceProps(void** allNeuronsDeviceProps, SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    AllIFNeuronsProps allNeuronsProps;

    // allocate GPU memories to store all neuron's states
    allocNeuronsDeviceProps(allNeuronsProps, sim_info, clr_info);

    checkCudaErrors( cudaMalloc( allNeuronsDeviceProps, sizeof( AllIFNeuronsProps ) ) );

    // copy the pointer address to structure on device memory
    checkCudaErrors( cudaMemcpy ( *allNeuronsDeviceProps, &allNeuronsProps, sizeof( AllIFNeuronsProps ), cudaMemcpyHostToDevice ) );
}

/*
 *  Allocate GPU memories to store all neurons' states.
 *
 *  @param  allNeuronsProps   Reference to the AllNeuronsProps class.
 *  @param  sim_info          SimulationInfo class to read information from.
 *  @param  clr_info          ClusterInfo to refer from.
 */
void AllIFNeuronsProps::allocNeuronsDeviceProps(AllIFNeuronsProps &allNeuronsProps, SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    int size = clr_info->totalClusterNeurons;

    AllSpikingNeuronsProps::allocNeuronsDeviceProps(allNeuronsProps, sim_info, clr_info);

    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProps.C1, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProps.C2, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProps.Cm, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProps.I0, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProps.Iinject, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProps.Inoise, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProps.Isyn, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProps.Rm, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProps.Tau, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProps.Trefract, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProps.Vinit, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProps.Vm, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProps.Vreset, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProps.Vrest, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProps.Vthresh, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProps.nStepsInRefr, size * sizeof( int ) ) );
}

/*
 *  Delete GPU memories.
 *
 *  @param  allNeuronsDeviceProps   Reference to the AllIFNeuronsProps class on device memory.
 *  @param  clr_info                ClusterInfo to refer from.
 */
void AllIFNeuronsProps::cleanupNeuronsDeviceProps(void *allNeuronsDeviceProps, ClusterInfo *clr_info)
{
    AllIFNeuronsProps allNeuronsProps;

    checkCudaErrors( cudaMemcpy ( &allNeuronsProps, allNeuronsDeviceProps, sizeof( AllIFNeuronsProps ), cudaMemcpyDeviceToHost ) );
    deleteNeuronsDeviceProps(allNeuronsProps, clr_info);

    checkCudaErrors( cudaFree( allNeuronsDeviceProps ) );

    // Set size to 0 to avoid illegal memory deallocation
    // at AllIFNeuronsProps deconstructor.
    allNeuronsProps.size = 0;
}

/*
 *  Delete GPU memories.
 *
 *  @param  allNeuronsProps   Reference to the AllNeuronsProps class.
 *  @param  clr_info          ClusterInfo to refer from.
 */
void AllIFNeuronsProps::deleteNeuronsDeviceProps(AllIFNeuronsProps &allNeuronsProps, ClusterInfo *clr_info)
{
    checkCudaErrors( cudaFree( allNeuronsProps.C1 ) );
    checkCudaErrors( cudaFree( allNeuronsProps.C2 ) );
    checkCudaErrors( cudaFree( allNeuronsProps.Cm ) );
    checkCudaErrors( cudaFree( allNeuronsProps.I0 ) );
    checkCudaErrors( cudaFree( allNeuronsProps.Iinject ) );
    checkCudaErrors( cudaFree( allNeuronsProps.Inoise ) );
    checkCudaErrors( cudaFree( allNeuronsProps.Isyn ) );
    checkCudaErrors( cudaFree( allNeuronsProps.Rm ) );
    checkCudaErrors( cudaFree( allNeuronsProps.Tau ) );
    checkCudaErrors( cudaFree( allNeuronsProps.Trefract ) );
    checkCudaErrors( cudaFree( allNeuronsProps.Vinit ) );
    checkCudaErrors( cudaFree( allNeuronsProps.Vm ) );
    checkCudaErrors( cudaFree( allNeuronsProps.Vreset ) );
    checkCudaErrors( cudaFree( allNeuronsProps.Vrest ) );
    checkCudaErrors( cudaFree( allNeuronsProps.Vthresh ) );
    checkCudaErrors( cudaFree( allNeuronsProps.nStepsInRefr ) );

    AllSpikingNeuronsProps::deleteNeuronsDeviceProps(allNeuronsProps, clr_info);
}

/*
 *  Copy all neurons' data from host to device.
 *
 *  @param  allNeuronsDeviceProps   Reference to the AllIFNeuronsProps class on device memory.
 *  @param  sim_info                SimulationInfo to refer from.
 *  @param  clr_info                ClusterInfo to refer from.
 */
void AllIFNeuronsProps::copyNeuronHostToDeviceProps( void* allNeuronsDeviceProps, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) 
{
    AllIFNeuronsProps allNeuronsProps;

    checkCudaErrors( cudaMemcpy ( &allNeuronsProps, allNeuronsDeviceProps, sizeof( AllIFNeuronsProps ), cudaMemcpyDeviceToHost ) );
    copyHostToDeviceProps( allNeuronsProps, sim_info, clr_info );

    // Set size to 0 to avoid illegal memory deallocation
    // at AllIFNeuronsProps deconstructor.
    allNeuronsProps.size = 0;
}

/*
 *  Copy all neurons' data from host to device.
 *  (Helper function of copyNeuronHostToDeviceProps)
 *
 *  @param  allNeuronsProps    Reference to the AllIFNeuronsProps class.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllIFNeuronsProps::copyHostToDeviceProps( AllIFNeuronsProps& allNeuronsProps, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) 
{
    int size = clr_info->totalClusterNeurons;

    AllSpikingNeuronsProps::copyHostToDeviceProps(allNeuronsProps, sim_info, clr_info);

    checkCudaErrors( cudaMemcpy ( allNeuronsProps.C1, C1, size * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( allNeuronsProps.C2, C2, size * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( allNeuronsProps.Cm, Cm, size * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( allNeuronsProps.I0, I0, size * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( allNeuronsProps.Iinject, Iinject, size * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( allNeuronsProps.Inoise, Inoise, size * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( allNeuronsProps.Isyn, Isyn, size * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( allNeuronsProps.Rm, Rm, size * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( allNeuronsProps.Tau, Tau, size * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( allNeuronsProps.Trefract, Trefract, size * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( allNeuronsProps.Vinit, Vinit, size * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( allNeuronsProps.Vm, Vm, size * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( allNeuronsProps.Vreset, Vreset, size * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( allNeuronsProps.Vrest, Vrest, size * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( allNeuronsProps.Vthresh, Vthresh, size * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( allNeuronsProps.nStepsInRefr, nStepsInRefr, size * sizeof( int ), cudaMemcpyHostToDevice ) );
}

/*
 *  Copy all neurons' data from device to host.
 *
 *  @param  allNeuronsDeviceProps   Reference to the AllIFNeuronsProps class on device memory.
 *  @param  sim_info                SimulationInfo to refer from.
 *  @param  clr_info                ClusterInfo to refer from.
 */
void AllIFNeuronsProps::copyNeuronDeviceToHostProps( void* allNeuronsDeviceProps, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) 
{
    AllIFNeuronsProps allNeuronsProps;

    checkCudaErrors( cudaMemcpy ( &allNeuronsProps, allNeuronsDeviceProps, sizeof( AllIFNeuronsProps ), cudaMemcpyDeviceToHost ) );
    copyDeviceToHostProps( allNeuronsProps, sim_info, clr_info );

    // Set size to 0 to avoid illegal memory deallocation
    // at AllIFNeuronsProps deconstructor.
    allNeuronsProps.size = 0;
}

/*
 *  Copy all neurons' data from device to host.
 *  (Helper function of copyNeuronDeviceToHostProps)
 *
 *  @param  allNeuronsProps    Reference to the AllIFNeuronsProps class.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllIFNeuronsProps::copyDeviceToHost( AllIFNeuronsProps& allNeuronsProps, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) 
{
    int size = clr_info->totalClusterNeurons;

    AllSpikingNeuronsProps::copyDeviceToHostProps(allNeuronsProps, sim_info, clr_info);

    checkCudaErrors( cudaMemcpy ( C1, allNeuronsProps.C1, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( C2, allNeuronsProps.C2, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( Cm, allNeuronsProps.C1, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( I0, allNeuronsProps.I0, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( Iinject, allNeuronsProps.Iinject, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( Inoise, allNeuronsProps.Inoise, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( Isyn, allNeuronsProps.Isyn, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( Rm, allNeuronsProps.Rm, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( Tau, allNeuronsProps.Tau, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( Trefract, allNeuronsProps.Trefract, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( Vinit, allNeuronsProps.Vinit, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( Vm, allNeuronsProps.Vm, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( Vreset, allNeuronsProps.Vreset, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( Vrest, allNeuronsProps.Vrest, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( Vthresh, allNeuronsProps.Vthresh, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( nStepsInRefr, allNeuronsProps.nStepsInRefr, size * sizeof( int ), cudaMemcpyDeviceToHost ) );
}
#endif // USE_GPU

/*
 *  Checks the number of required parameters.
 *
 * @return true if all required parameters were successfully read, false otherwise.
 */
bool AllIFNeuronsProps::checkNumParameters()
{
    return (nParams >= 8);
}

/*
 *  Attempts to read parameters from a XML file.
 *
 *  @param  element TiXmlElement to examine.
 *  @return true if successful, false otherwise.
 */
bool AllIFNeuronsProps::readParameters(const TiXmlElement& element)
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
void AllIFNeuronsProps::printParameters(ostream &output) const
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
 *  @param  r_neurons  Neurons properties class object to copy from.
 */
void AllIFNeuronsProps::copyParameters(const AllNeuronsProps *r_neuronsProps)
{
    const AllIFNeuronsProps *pProps = dynamic_cast<const AllIFNeuronsProps*>(r_neuronsProps);

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
void AllIFNeuronsProps::readNeuronProps(istream &input, int i)
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
 *  Writes out the data in the selected Neuron.
 *
 *  @param  output      stream to write out to.
 *  @param  i           index of the neuron (in neurons).
 */
void AllIFNeuronsProps::writeNeuronProps(ostream& output, int i) const
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

/*
 *  Creates a single Neuron and generates data for it.
 *
 *  @param  sim_info     SimulationInfo class to read information from.
 *  @param  neuron_index Index of the neuron to create.
 *  @param  layout       Layout information of the neunal network.
 *  @param  clr_info     ClusterInfo class to read information from.
 */
void AllIFNeuronsProps::setNeuronPropValues(SimulationInfo *sim_info, int neuron_index, Layout *layout, ClusterInfo *clr_info)
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

    initNeuronPropConstsFromParamValues(neuron_index, sim_info->deltaT);

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
void AllIFNeuronsProps::setNeuronPropDefaults(const int index)
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
void AllIFNeuronsProps::initNeuronPropConstsFromParamValues(int neuron_index, const BGFLOAT deltaT)
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
