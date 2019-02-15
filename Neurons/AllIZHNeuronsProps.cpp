#include "AllIZHNeuronsProps.h"
#include "ParseParamError.h"
#if defined(USE_GPU)
#include <helper_cuda.h>
#endif

// Default constructor
AllIZHNeuronsProps::AllIZHNeuronsProps() 
{
    Aconst = NULL;
    Bconst = NULL;
    Cconst = NULL;
    Dconst = NULL;
    u = NULL;
    C3 = NULL;
}

AllIZHNeuronsProps::~AllIZHNeuronsProps()
{
    cleanupNeuronsProps();
}

/*
 *  Setup the internal structure of the class (allocate memories).
 *
 *  @param  sim_info  SimulationInfo class to read information from.
 *  @param  clr_info  ClusterInfo class to read information from.
 */
void AllIZHNeuronsProps::setupNeuronsProps(SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    AllIFNeuronsProps::setupNeuronsProps(sim_info, clr_info);

    Aconst = new BGFLOAT[size];
    Bconst = new BGFLOAT[size];
    Cconst = new BGFLOAT[size];
    Dconst = new BGFLOAT[size];
    u = new BGFLOAT[size];
    C3 = new BGFLOAT[size];
}

/*
 *  Cleanup the class (deallocate memories).
 */
void AllIZHNeuronsProps::cleanupNeuronsProps()
{
    if (size != 0) {
        delete[] Aconst;
        delete[] Bconst;
        delete[] Cconst;
        delete[] Dconst;
        delete[] u;
        delete[] C3;
    }

    Aconst = NULL;
    Bconst = NULL;
    Cconst = NULL;
    Dconst = NULL;
    u = NULL;
    C3 = NULL;
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
void AllIZHNeuronsProps::setupNeuronsDeviceProps(void** allNeuronsDeviceProps, SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    AllIZHNeuronsProps allNeuronsProps;

    // allocate GPU memories to store all neuron's states
    allocNeuronsDeviceProps(allNeuronsProps, sim_info, clr_info);

    checkCudaErrors( cudaMalloc( allNeuronsDeviceProps, sizeof( AllIZHNeuronsProps ) ) );

    // copy the pointer address to structure on device memory
    checkCudaErrors( cudaMemcpy (*allNeuronsDeviceProps, &allNeuronsProps, sizeof( AllIZHNeuronsProps ), cudaMemcpyHostToDevice ) );
}

/*
 *  Allocate GPU memories to store all neurons' states.
 *
 *  @param  allNeuronsProps   Reference to the AllIZHNeuronsProps class.
 *  @param  sim_info          SimulationInfo class to read information from.
 *  @param  clr_info          ClusterInfo to refer from.
 */
void AllIZHNeuronsProps::allocNeuronsDeviceProps(AllIZHNeuronsProps &allNeuronsProps, SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    int size = clr_info->totalClusterNeurons;

    AllIFNeuronsProps::allocNeuronsDeviceProps(allNeuronsProps, sim_info, clr_info);

    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProps.Aconst, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProps.Bconst, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProps.Cconst, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProps.Dconst, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProps.u, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProps.C3, size * sizeof( BGFLOAT ) ) );
}

/*
 *  Delete GPU memories.
 *
 *  @param  allNeuronsDeviceProps   Reference to the AllIZHNeuronsProps class on device memory.
 *  @param  clr_info                ClusterInfo to refer from.
 */
void AllIZHNeuronsProps::cleanupNeuronsDeviceProps(void *allNeuronsDeviceProps, ClusterInfo *clr_info)
{
    AllIZHNeuronsProps allNeuronsProps;

    checkCudaErrors( cudaMemcpy ( &allNeuronsProps, allNeuronsDeviceProps, sizeof( AllIZHNeuronsProps ), cudaMemcpyDeviceToHost ) );
    deleteNeuronsDeviceProps(allNeuronsProps, clr_info);

    checkCudaErrors( cudaFree( allNeuronsDeviceProps ) );

    // Set size to 0 to avoid illegal memory deallocation
    // at AllIZHNeuronsProps deconstructor.
    allNeuronsProps.size = 0;
}

/*
 *  Delete GPU memories.
 *
 *  @param  allNeuronsProps   Reference to the AllIZHNeuronsProps class.
 *  @param  clr_info          ClusterInfo to refer from.
 */
void AllIZHNeuronsProps::deleteNeuronsDeviceProps(AllIZHNeuronsProps &allNeuronsProps, ClusterInfo *clr_info)
{
    checkCudaErrors( cudaFree( allNeuronsProps.Aconst ) );
    checkCudaErrors( cudaFree( allNeuronsProps.Bconst ) );
    checkCudaErrors( cudaFree( allNeuronsProps.Cconst ) );
    checkCudaErrors( cudaFree( allNeuronsProps.Dconst ) );
    checkCudaErrors( cudaFree( allNeuronsProps.u ) );
    checkCudaErrors( cudaFree( allNeuronsProps.C3 ) );

    AllIFNeuronsProps::deleteNeuronsDeviceProps(allNeuronsProps, clr_info);
}

/*
 *  Copy all neurons' data from host to device.
 *
 *  @param  allNeuronsDeviceProps   Reference to the AllIZHNeuronsProps class on device memory.
 *  @param  sim_info                SimulationInfo to refer from.
 *  @param  clr_info                ClusterInfo to refer from.
 */
void AllIZHNeuronsProps::copyNeuronHostToDeviceProps( void* allNeuronsDeviceProps, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) {
    AllIZHNeuronsProps allNeuronsProps;

    checkCudaErrors( cudaMemcpy ( &allNeuronsProps, allNeuronsDeviceProps, sizeof( AllIZHNeuronsProps ), cudaMemcpyDeviceToHost ) );
    copyHostToDeviceProps( allNeuronsProps, sim_info, clr_info );

    // Set size to 0 to avoid illegal memory deallocation
    // at AllIZHNeuronsProps deconstructor.
    allNeuronsProps.size = 0;
}

/*
 *  Copy all neurons' data from host to device.
 *  (Helper function of copyNeuronHostToDevicePrps)
 *
 *  @param  allNeuronsProps         Reference to the AllIZHNeuronsProps struct.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllIZHNeuronsProps::copyHostToDeviceProps( AllIZHNeuronsProps& allNeuronsProps, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) 
{
    int size = clr_info->totalClusterNeurons;

    AllIFNeuronsProps::copyHostToDeviceProps( allNeuronsProps, sim_info, clr_info );

    checkCudaErrors( cudaMemcpy ( allNeuronsProps.Aconst, Aconst, size * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( allNeuronsProps.Bconst, Bconst, size * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( allNeuronsProps.Cconst, Cconst, size * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( allNeuronsProps.Dconst, Dconst, size * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( allNeuronsProps.u, u, size * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( allNeuronsProps.C3, C3, size * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
}

/*
 *  Copy all neurons' data from device to host.
 *
 *  @param  allNeuronsDeviceProps   Reference to the AllIZHNeuronsProps class on device memory.
 *  @param  sim_info                SimulationInfo to refer from.
 *  @param  clr_info                ClusterInfo to refer from.
 */
void AllIZHNeuronsProps::copyNeuronDeviceToHostProps( void* allNeuronsDeviceProps, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) 
{
    AllIZHNeuronsProps allNeuronsProps;

    checkCudaErrors( cudaMemcpy ( &allNeuronsProps, allNeuronsDeviceProps, sizeof( AllIZHNeuronsProps ), cudaMemcpyDeviceToHost ) );
    copyDeviceToHostProps( allNeuronsProps, sim_info, clr_info );

    // Set size to 0 to avoid illegal memory deallocation
    // at AllIZHNeuronsProps deconstructor.
    allNeuronsProps.size = 0;
}

/*
 *  Copy all neurons' data from device to host.
 *  (Helper function of copyNeuronDeviceToHostProps)
 *
 *  @param  allNeuronsProps         Reference to the AllIZHNeuronsProps struct.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllIZHNeuronsProps::copyDeviceToHostProps( AllIZHNeuronsProps& allNeuronsProps, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) 
{
    int size = clr_info->totalClusterNeurons;

    AllIFNeuronsProps::copyDeviceToHostProps( allNeuronsProps, sim_info, clr_info );

    checkCudaErrors( cudaMemcpy ( Aconst, allNeuronsProps.Aconst, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( Bconst, allNeuronsProps.Bconst, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( Cconst, allNeuronsProps.Cconst, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( Dconst, allNeuronsProps.Dconst, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( u, allNeuronsProps.u, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( C3, allNeuronsProps.C3, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
}
#endif // USE_GPU

/*
 * Checks the number of required parameters.
 *
 * @return true if all required parameters were successfully read, false otherwise.
 */
bool AllIZHNeuronsProps::checkNumParameters()
{
    return (nParams >= 12);
}

/*
 *  Attempts to read parameters from a XML file.
 *
 *  @param  element TiXmlElement to examine.
 *  @return true if successful, false otherwise.
 */
bool AllIZHNeuronsProps::readParameters(const TiXmlElement& element)
{
    if (AllIFNeuronsProps::readParameters(element)) {
        // this parameter was already handled
        return true;
    }

    if (element.ValueStr().compare("Aconst") == 0 ||
        element.ValueStr().compare("Bconst") == 0 ||
        element.ValueStr().compare("Cconst") == 0 ||
        element.ValueStr().compare("Dconst") == 0    ) {
        nParams++;
        return true;
    }

    if (element.Parent()->ValueStr().compare("Aconst") == 0) {
        // Min/max values of Aconst for excitatory neurons.
        if(element.ValueStr().compare("minExc") == 0){
            m_excAconst[0] = atof(element.GetText());

            if (m_excAconst[0] < 0) {
                throw ParseParamError("Aconst minExc", "Invalid negative Aconst value.");
            }
        }
        else if(element.ValueStr().compare("maxExc") == 0){
            m_excAconst[1] = atof(element.GetText());

            if (m_excAconst[0] > m_excAconst[1]) {
                throw ParseParamError("Aconst maxExc", "Invalid range for Aconst value.");
            }
        }
        else if(element.ValueStr().compare("minInh") == 0){
            m_inhAconst[0] = atof(element.GetText());

            if (m_inhAconst[0] < 0) {
                throw ParseParamError("Aconst minInh", "Invalid negative Aconst value.");
            }
        }
        else if(element.ValueStr().compare("maxInh") == 0){
            m_inhAconst[1] = atof(element.GetText());

            if (m_inhAconst[0] > m_inhAconst[1]) {
                throw ParseParamError("Aconst maxInh", "Invalid range for Aconst value.");
            }
        }

        nParams++;
        return true;
    }

    if (element.Parent()->ValueStr().compare("Bconst") == 0) {
        // Min/max values of Bconst for excitatory neurons.
        if(element.ValueStr().compare("minExc") == 0){
            m_excBconst[0] = atof(element.GetText());

            if (m_excBconst[0] < 0) {
                throw ParseParamError("Bconst minExc", "Invalid negative Bconst value.");
            }
        }
        else if(element.ValueStr().compare("maxExc") == 0){
            m_excBconst[1] = atof(element.GetText());

            if (m_excBconst[0] > m_excBconst[1]) {
                throw ParseParamError("Bconst maxExc", "Invalid range for Bconst value.");
            }
        }

        // Min/max values of Bconst for inhibitory neurons.
        if(element.ValueStr().compare("minInh") == 0){
            m_inhBconst[0] = atof(element.GetText());

            if (m_inhBconst[0] < 0) {
                throw ParseParamError("Bconst minInh", "Invalid negative Bconst value.");
            }
        }
        else if(element.ValueStr().compare("maxInh") == 0){
            m_inhBconst[1] = atof(element.GetText());

            if (m_inhBconst[0] > m_inhBconst[1]) {
                throw ParseParamError("Bconst maxInh", "Invalid range for Bconst value.");
            }
        }

        nParams++;
        return true;
    }

    if (element.Parent()->ValueStr().compare("Cconst") == 0) {
        // Min/max values of Cconst for excitatory neurons.
        if(element.ValueStr().compare("minExc") == 0){
            m_excCconst[0] = atof(element.GetText());
        }
        else if(element.ValueStr().compare("maxExc") == 0){
            m_excCconst[1] = atof(element.GetText());

            if (m_excCconst[0] > m_excCconst[1]) {
                throw ParseParamError("Cconst maxExc", "Invalid range for Cconst value.");
            }
        }

        // Min/max values of Cconst for inhibitory neurons.
        if(element.ValueStr().compare("minInh") == 0){
            m_inhCconst[0] = atof(element.GetText());
        }
        else if(element.ValueStr().compare("maxInh") == 0){
            m_inhCconst[1] = atof(element.GetText());

            if (m_inhCconst[0] > m_inhCconst[1]) {
                throw ParseParamError("Cconst maxInh", "Invalid range for Cconst value.");
            }
        }

        nParams++;
        return true;
    }

    if (element.Parent()->ValueStr().compare("Dconst") == 0) {
        // Min/max values of Dconst for excitatory neurons.
        if(element.ValueStr().compare("minExc") == 0){
            m_excDconst[0] = atof(element.GetText());

            if (m_excDconst[0] < 0) {
                throw ParseParamError("Dconst minExc", "Invalid negative Dconst value.");
            }
        }
        else if(element.ValueStr().compare("maxExc") == 0){
            m_excDconst[1] = atof(element.GetText());

            if (m_excDconst[0] > m_excDconst[1]) {
                throw ParseParamError("Dconst maxExc", "Invalid range for Dconst value.");
            }
        }
        // Min/max values of Dconst for inhibitory neurons.
        if(element.ValueStr().compare("minInh") == 0){
            m_inhDconst[0] = atof(element.GetText());

            if (m_inhDconst[0] < 0) {
                throw ParseParamError("Dconst minInh", "Invalid negative Dconst value.");
            }
        }
        else if(element.ValueStr().compare("maxInh") == 0){
            m_inhDconst[1] = atof(element.GetText());

            if (m_inhDconst[0] > m_inhDconst[1]) {
                throw ParseParamError("Dconst maxInh", "Invalid range for Dconst value.");
            }
        }

        nParams++;
        return true;
    }

    return false;
}

/*
 *  Prints out all parameters of the neurons to ostream.
 *
 *  @param  output  ostream to send output to.
 */
void AllIZHNeuronsProps::printParameters(ostream &output) const
{
    AllIFNeuronsProps::printParameters(output);

    output << "Interval of A constant for excitatory neurons: ["
           << m_excAconst[0] << ", " << m_excAconst[1] << "]"
           << endl;
    output << "Interval of A constant for inhibitory neurons: ["
           << m_inhAconst[0] << ", " << m_inhAconst[1] << "]"
           << endl;
    output << "Interval of B constant for excitatory neurons: ["
           << m_excBconst[0] << ", " << m_excBconst[1] << "]"
           << endl;
    output << "Interval of B constant for inhibitory neurons: ["
           << m_inhBconst[0] << ", " << m_inhBconst[1] << "]"
           << endl;
    output << "Interval of C constant for excitatory neurons: ["
           << m_excCconst[0] << ", "<< m_excCconst[1] << "]"
           << endl;
    output << "Interval of C constant for inhibitory neurons: ["
           << m_inhCconst[0] << ", "<< m_inhCconst[1] << "]"
           << endl;
    output << "Interval of D constant for excitatory neurons: ["
           << m_excDconst[0] << ", "<< m_excDconst[1] << "]"
           << endl;
    output << "Interval of D constant for inhibitory neurons: ["
           << m_inhDconst[0] << ", "<< m_inhDconst[1] << "]"
           << endl;
}

/*
 *  Copy neurons parameters.
 *
 *  @param  r_neurons  Neurons properties class object to copy from.
 */
void AllIZHNeuronsProps::copyParameters(const AllNeuronsProps *r_neuronsProps)
{
    AllIFNeuronsProps::copyParameters(r_neuronsProps);

    const AllIZHNeuronsProps *pProps = dynamic_cast<const AllIZHNeuronsProps*>(r_neuronsProps);

    for (int i = 0; i < 2; i++) {
        m_excAconst[i] = pProps->m_excAconst[i];
        m_inhAconst[i] = pProps->m_inhAconst[i];
        m_excBconst[i] = pProps->m_excBconst[i];
        m_inhBconst[i] = pProps->m_inhBconst[i];
        m_excCconst[i] = pProps->m_excCconst[i];
        m_inhCconst[i] = pProps->m_inhCconst[i];
        m_excDconst[i] = pProps->m_excDconst[i];
        m_inhDconst[i] = pProps->m_inhDconst[i];
    }
}

/*
 *  Sets the data for Neuron #index to input's data.
 *
 *  @param  input       istream to read from.
 *  @param  i           index of the neuron (in neurons).
 */
void AllIZHNeuronsProps::readNeuronProps(istream &input, int i)
{
    AllIFNeuronsProps::readNeuronProps(input, i);

    input >> Aconst[i]; input.ignore();
    input >> Bconst[i]; input.ignore();
    input >> Cconst[i]; input.ignore();
    input >> Dconst[i]; input.ignore();
    input >> u[i]; input.ignore();
    input >> C3[i]; input.ignore();
}

/*
 *  Writes out the data in the selected Neuron.
 *
 *  @param  output      stream to write out to.
 *  @param  i           index of the neuron (in neurons).
 */
void AllIZHNeuronsProps::writeNeuronProps(ostream& output, int i) const
{
    AllIFNeuronsProps::writeNeuronProps(output, i);

    output << Aconst[i] << ends;
    output << Bconst[i] << ends;
    output << Cconst[i] << ends;
    output << Dconst[i] << ends;
    output << u[i] << ends;
    output << C3[i] << ends;
}

/*
 *  Creates a single Neuron and generates data for it.
 *
 *  @param  sim_info     SimulationInfo class to read information from.
 *  @param  neuron_index Index of the neuron to create.
 *  @param  layout       Layout information of the neunal network.
 *  @param  clr_info     ClusterInfo class to read information from.
 */
void AllIZHNeuronsProps::setNeuronPropValues(SimulationInfo *sim_info, int neuron_index, Layout *layout, ClusterInfo *clr_info)
{
    BGFLOAT &Aconst = this->Aconst[neuron_index];
    BGFLOAT &Bconst = this->Bconst[neuron_index];
    BGFLOAT &Cconst = this->Cconst[neuron_index];
    BGFLOAT &Dconst = this->Dconst[neuron_index];
    BGFLOAT &u = this->u[neuron_index];
    BGFLOAT &C3 = this->C3[neuron_index];

    // set the neuron info for neurons
    AllIFNeuronsProps::setNeuronPropValues(sim_info, neuron_index, layout, clr_info);

    // TODO: we may need another distribution mode besides flat distribution
    int neuron_layout_index = clr_info->clusterNeuronsBegin + neuron_index;
    if (layout->neuron_type_map[neuron_layout_index] == EXC) {
        // excitatory neuron
        Aconst = rng.inRange(m_excAconst[0], m_excAconst[1]);
        Bconst = rng.inRange(m_excBconst[0], m_excBconst[1]);
        Cconst = rng.inRange(m_excCconst[0], m_excCconst[1]);
        Dconst = rng.inRange(m_excDconst[0], m_excDconst[1]);
    } else {
        // inhibitory neuron
        Aconst = rng.inRange(m_inhAconst[0], m_inhAconst[1]);
        Bconst = rng.inRange(m_inhBconst[0], m_inhBconst[1]);
        Cconst = rng.inRange(m_inhCconst[0], m_inhCconst[1]);
        Dconst= rng.inRange(m_inhDconst[0], m_inhDconst[1]);
    }

    u = 0;

    initNeuronPropConstsFromParamValues(neuron_index, sim_info->deltaT);

    DEBUG_HI(cout << "CREATE NEURON[" << neuron_layout_index << "] {" << endl
            << "\tAconst = " << Aconst << endl
            << "\tBconst = " << Bconst << endl
            << "\tCconst = " << Cconst << endl
            << "\tDconst = " << Dconst << endl
            << "\tC3 = " << C3 << endl
            << "}" << endl
    ;)

}

/*
 *  Set the Neuron at the indexed location to default values.
 *
 *  @param  neuron_index    Index of the Neuron to refer.
 */
void AllIZHNeuronsProps::setNeuronPropDefaults(const int index)
{
    BGFLOAT &Aconst = this->Aconst[index];
    BGFLOAT &Bconst = this->Bconst[index];
    BGFLOAT &Cconst = this->Cconst[index];
    BGFLOAT &Dconst = this->Dconst[index];
    BGFLOAT &Trefract = this->Trefract[index];

    AllIFNeuronsProps::setNeuronPropDefaults(index);

    // no refractory period
    Trefract = 0;

    Aconst = DEFAULT_a;
    Bconst = DEFAULT_b;
    Cconst = DEFAULT_c;
    Dconst = DEFAULT_d;
}

/*
 *  Initializes the Neuron constants at the indexed location.
 *
 *  @param  neuron_index    Index of the Neuron.
 *  @param  deltaT          Inner simulation step duration
 */
void AllIZHNeuronsProps::initNeuronPropConstsFromParamValues(int neuron_index, const BGFLOAT deltaT)
{
    AllIFNeuronsProps::initNeuronPropConstsFromParamValues(neuron_index, deltaT);

    BGFLOAT &C3 = this->C3[neuron_index];
    C3 = deltaT * 1000;
}

