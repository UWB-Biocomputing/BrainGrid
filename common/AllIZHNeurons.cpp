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
int AllIZHNeurons::readParameters(const TiXmlElement& element)
{
    int read_params = 0;

    read_params += AllIFNeurons::readParameters(element);

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
        read_params++;
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
        read_params++;
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
        read_params++;
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
        read_params++;
    }

    return read_params;
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
void AllIZHNeurons::createAllNeurons(SimulationInfo *sim_info)
{
    /* set their specific types */
    for (int neuron_index = 0; neuron_index < sim_info->totalNeurons; neuron_index++) {
        setNeuronDefaults(neuron_index);

        // set the neuron info for neurons
        createNeuron(sim_info, neuron_index);
    }
}

void AllIZHNeurons::createNeuron(SimulationInfo *sim_info, int neuron_index)
{
    // set the neuron info for neurons
    AllIFNeurons::createNeuron(sim_info, neuron_index);

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
