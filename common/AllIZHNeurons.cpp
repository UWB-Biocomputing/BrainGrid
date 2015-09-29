#include "AllIZHNeurons.h"
#include "ParseParamError.h"

// Default constructor
AllIZHNeurons::AllIZHNeurons() : AllIFNeurons()
{
    Aconst = NULL;
    Bconst = NULL;
    Cconst = NULL;
    Dconst = NULL;
    u = NULL;
    C3 = NULL;
}

AllIZHNeurons::~AllIZHNeurons()
{
    freeResources();
}

void AllIZHNeurons::setupNeurons(SimulationInfo *sim_info)
{
    AllIFNeurons::setupNeurons(sim_info);

    Aconst = new BGFLOAT[size];
    Bconst = new BGFLOAT[size];
    Cconst = new BGFLOAT[size];
    Dconst = new BGFLOAT[size];
    u = new BGFLOAT[size];
    C3 = new BGFLOAT[size];
}

void AllIZHNeurons::cleanupNeurons()
{
    freeResources();
    AllIFNeurons::cleanupNeurons();
}

void AllIZHNeurons::freeResources()
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

/**
 * Returns the number of required parameters.
 */
int AllIZHNeurons::numParameters()
{
    return AllIFNeurons::numParameters() + 4;
}

/**
 *  Attempts to read parameters from a XML file.
 *  @param  @param  element TiXmlElement to examine.
 *  @return the number of parameters that have been read.
 */
bool AllIZHNeurons::readParameters(const TiXmlElement& element)
{
    AllIFNeurons::readParameters(element);

    if (element.ValueStr().compare("Aconst") == 0) {
        if (element.QueryFLOATAttribute("min", &m_Aconst[0]) != TIXML_SUCCESS) {
            throw ParseParamError("Aconst min", "Aconst missing minimum value in XML.");
        }
        if (m_Aconst[0] < 0) {
            throw ParseParamError("Aconst min", "Invalid negative Aconst value.");
        }
        if (element.QueryFLOATAttribute("max", &m_Aconst[1]) != TIXML_SUCCESS) {
            throw ParseParamError("Aconst max", "Aconst missing maximum value in XML.");
        }
        if (m_Aconst[0] > m_Aconst[1]) {
            throw ParseParamError("Aconst max", "Invalid range for Aconst value.");
        }
        return true;
    }

    if (element.ValueStr().compare("Bconst") == 0) {
        if (element.QueryFLOATAttribute("min", &m_Bconst[0]) != TIXML_SUCCESS) {
            throw ParseParamError("Bconst min", "Bconst missing minimum value in XML.");
        }
        if (element.QueryFLOATAttribute("max", &m_Bconst[1]) != TIXML_SUCCESS) {
            throw ParseParamError("Bconst max", "Bconst missing maximum value in XML.");
        }
        if (m_Bconst[0] > m_Bconst[1]) {
            throw ParseParamError("Bconst max", "Invalid range for Bconst value.");
        }
        return true;
    }

    if (element.ValueStr().compare("Cconst") == 0) {
        if (element.QueryFLOATAttribute("min", &m_Cconst[0]) != TIXML_SUCCESS) {
            throw ParseParamError("Cconst min", "Cconst missing minimum value in XML.");
        }
        if (element.QueryFLOATAttribute("max", &m_Cconst[1]) != TIXML_SUCCESS) {
            throw ParseParamError("Cconst max", "Cconst missing maximum value in XML.");
        }
        if (m_Cconst[0] > m_Cconst[1]) {
            throw ParseParamError("Cconst max", "Invalid range for Cconst value.");
        }
        return true;
    }

    if (element.ValueStr().compare("Dconst") == 0) {
        if (element.QueryFLOATAttribute("min", &m_Dconst[0]) != TIXML_SUCCESS) {
            throw ParseParamError("Dconst min", "Dconst missing minimum value in XML.");
        }
        if (m_Dconst[0] < 0) {
            throw ParseParamError("Dconst min", "Invalid negative Dconst value.");
        }
        if (element.QueryFLOATAttribute("max", &m_Dconst[1]) != TIXML_SUCCESS) {
            throw ParseParamError("Dconst max", "Dconst missing maximum value in XML.");
        }
        if (m_Dconst[0] > m_Dconst[1]) {
            throw ParseParamError("Dconst max", "Invalid range for Dconst value.");
        }
        return true;
    }

    return true;
}

/**
 *  Prints out all parameters of the neurons to ostream.
 *  @param  output  ostream to send output to.
 */
void AllIZHNeurons::printParameters(ostream &output) const
{
    AllIFNeurons::printParameters(output);

    output << "Interval of A constant: ["
           << m_Aconst[0] << ", " << m_Aconst[1] << "]"
           << endl;
    output << "Interval of B constant: ["
           << m_Bconst[0] << ", " << m_Bconst[1] << "]"
           << endl;
    output << "Interval of C constant: ["
           << m_Cconst[0] << ", "<< m_Cconst[1] << "]"
           << endl;
    output << "Interval of D constant: ["
           << m_Dconst[0] << ", "<< m_Dconst[1] << "]"
           << endl;
}

/**
 *  Creates all the Neurons and generates data for them.
 *  @param  sim_info    SimulationInfo class to read information from.
 */
void AllIZHNeurons::createAllNeurons(SimulationInfo *sim_info, Layout *layout)
{
    /* set their specific types */
    for (int neuron_index = 0; neuron_index < sim_info->totalNeurons; neuron_index++) {
        setNeuronDefaults(neuron_index);

        // set the neuron info for neurons
        createNeuron(sim_info, neuron_index, layout);
    }
}

void AllIZHNeurons::createNeuron(SimulationInfo *sim_info, int neuron_index, Layout *layout)
{
    // set the neuron info for neurons
    AllIFNeurons::createNeuron(sim_info, neuron_index, layout);

    Aconst[neuron_index] = rng.inRange(m_Aconst[0], m_Aconst[1]); 
    Bconst[neuron_index] = rng.inRange(m_Bconst[0], m_Bconst[1]); 
    Cconst[neuron_index] = rng.inRange(m_Cconst[0], m_Cconst[1]); 
    Dconst[neuron_index] = rng.inRange(m_Dconst[0], m_Dconst[1]); 
    u[neuron_index] = 0;

    DEBUG_HI(cout << "CREATE NEURON[" << neuron_index << "] {" << endl
            << "\tAconst = " << Aconst[neuron_index] << endl
            << "\tBconst = " << Bconst[neuron_index] << endl
            << "\tCconst = " << Cconst[neuron_index] << endl
            << "\tDconst = " << Dconst[neuron_index] << endl
            << "\tC3 = " << C3[neuron_index] << endl
            << "}" << endl
    ;)

}

/**
 *  Set the Neuron at the indexed location to default values.
 *  @param  neuron_index    index of the Neuron that the synapse belongs to.
 */
void AllIZHNeurons::setNeuronDefaults(const int index)
{
    AllIFNeurons::setNeuronDefaults(index);

    Aconst[index] = DEFAULT_a;
    Bconst[index] = DEFAULT_b;
    Cconst[index] = DEFAULT_c;
    Dconst[index] = DEFAULT_d;
}

/**
 *  Initializes the Neuron constants at the indexed location.
 *  @param  neuron_index    index of the Neuron.
 *  @param  deltaT  inner simulation step duration
 */
void AllIZHNeurons::initNeuronConstsFromParamValues(int neuron_index, const BGFLOAT deltaT)
{
    AllIFNeurons::initNeuronConstsFromParamValues(neuron_index, deltaT);

    BGFLOAT &C3 = this->C3[neuron_index];
    C3 = deltaT * 1000; 
}

/**
 *  Outputs state of the neuron chosen as a string.
 *  @param  i   index of the neuron (in neurons) to output info from.
 *  @return the complete state of the neuron.
 */
string AllIZHNeurons::toString(const int i) const
{
    stringstream ss;

    ss << AllIFNeurons::toString(i);

    ss << "Aconst: " << Aconst[i] << " ";
    ss << "Bconst: " << Bconst[i] << " ";
    ss << "Cconst: " << Cconst[i] << " ";
    ss << "Dconst: " << Dconst[i] << " ";
    ss << "u: " << u[i] << " ";
    ss << "C3: " << C3[i] << " ";
    return ss.str( );
}

/**
 *  Sets the data for Neuron #index to input's data.
 *  @param  input   istream to read from.
 *  @param  sim_info    used as a reference to set info for neuronss.
 */
void AllIZHNeurons::readNeurons(istream &input, const SimulationInfo *sim_info)
{
    for (int i = 0; i < sim_info->totalNeurons; i++) {
        readNeuron(input, sim_info, i);
    }
}

void AllIZHNeurons::readNeuron(istream &input, const SimulationInfo *sim_info, int i)
{
    AllIFNeurons::readNeuron(input, sim_info, i);

    input >> Aconst[i]; input.ignore();
    input >> Bconst[i]; input.ignore();
    input >> Cconst[i]; input.ignore();
    input >> Dconst[i]; input.ignore();
    input >> u[i]; input.ignore();
    input >> C3[i]; input.ignore();
}

/**
 *  Writes out the data in the selected Neuron.
 *  @param  output  stream to write out to.
 *  @param  sim_info    used as a reference to set info for neuronss.
 */
void AllIZHNeurons::writeNeurons(ostream& output, const SimulationInfo *sim_info) const 
{
    for (int i = 0; i < sim_info->totalNeurons; i++) {
        writeNeuron(output, sim_info, i);
    }
}

void AllIZHNeurons::writeNeuron(ostream& output, const SimulationInfo *sim_info, int i) const
{
    AllIFNeurons::writeNeuron(output, sim_info, i);

    output << Aconst[i] << ends;
    output << Bconst[i] << ends;
    output << Cconst[i] << ends;
    output << Dconst[i] << ends;
    output << u[i] << ends;
    output << C3[i] << ends;
}

#if !defined(USE_GPU)
/**
 *  Update the indexed Neuron.
 *  @param  index   index of the Neuron to update.
 *  @param  sim_info    SimulationInfo class to read information from.
 */
void AllIZHNeurons::advanceNeuron(const int index, const SimulationInfo *sim_info)
{
    const BGFLOAT deltaT = sim_info->deltaT;
    BGFLOAT &Vm = this->Vm[index];
    BGFLOAT &Vthresh = this->Vthresh[index];
    BGFLOAT &summationPoint = this->summation_map[index];
    BGFLOAT &I0 = this->I0[index];
    BGFLOAT &Inoise = this->Inoise[index];
    BGFLOAT &C1 = this->C1[index];
    BGFLOAT &C2 = this->C2[index];
    BGFLOAT &C3 = this->C3[index];
    int &nStepsInRefr = this->nStepsInRefr[index];

    BGFLOAT &a = Aconst[index];
    BGFLOAT &b = Bconst[index];
    BGFLOAT &u = this->u[index];

    if (nStepsInRefr > 0) {
        // is neuron refractory?
        --nStepsInRefr;
    } else if (Vm >= Vthresh) {
        // should it fire?
        fire(index, sim_info);
    } else {
        summationPoint += I0; // add IO
        // add noise
        BGFLOAT noise = (*rgNormrnd[0])();
        DEBUG_MID(cout << "ADVANCE NEURON[" << index << "] :: noise = " << noise << endl;)
        summationPoint += noise * Inoise; // add noise

        BGFLOAT Vint = Vm * 1000;

        // Izhikevich model integration step
        BGFLOAT Vb = Vint + C3 * (0.04 * Vint * Vint + 5 * Vint + 140 - u);
        u = u + C3 * a * (b * Vint - u);

        Vm = Vb * 0.001 + C2 * summationPoint;  // add inputs
    }

    DEBUG_MID(cout << index << " " << Vm << endl;)
        DEBUG_MID(cout << "NEURON[" << index << "] {" << endl
            << "\tVm = " << Vm << endl
            << "\ta = " << a << endl
            << "\tb = " << b << endl
            << "\tc = " << Cconst[index] << endl
            << "\td = " << Dconst[index] << endl
            << "\tu = " << u << endl
            << "\tVthresh = " << Vthresh << endl
            << "\tsummationPoint = " << summationPoint << endl
            << "\tI0 = " << I0 << endl
            << "\tInoise = " << Inoise << endl
            << "\tC1 = " << C1 << endl
            << "\tC2 = " << C2 << endl
            << "\tC3 = " << C3 << endl
            << "}" << endl
    ;)

    // clear synaptic input for next time step
    summationPoint = 0;
}

/**
 *  Fire the selected Neuron and calculate the result.
 *  @param  index   index of the Neuron to update.
 *  @param  sim_info    SimulationInfo class to read information from.
 */
void AllIZHNeurons::fire(const int index, const SimulationInfo *sim_info) const
{
    const BGFLOAT deltaT = sim_info->deltaT;
    AllSpikingNeurons::fire(index, sim_info);

    // calculate the number of steps in the absolute refractory period
    BGFLOAT &Vm = this->Vm[index];
    int &nStepsInRefr = this->nStepsInRefr[index];
    BGFLOAT &Trefract = this->Trefract[index];

    BGFLOAT &c = Cconst[index];
    BGFLOAT &d = Dconst[index];
    BGFLOAT &u = this->u[index];

    nStepsInRefr = static_cast<int> ( Trefract / deltaT + 0.5 );

    // reset to 'Vreset'
    Vm = c * 0.001;
    u = u + d;
}
#endif
