/*
 * AllIZHNeurons.cpp
 *
 */
// Updated 2/8/2017 by Jewel 
// look for "IZH03" for modifications

#include "AllIZHNeurons.h"
#include "ParseParamError.h"
#include <stdlib.h>

// Default constructor
AllIZHNeurons::AllIZHNeurons() : AllIFNeurons()
{
    Aconst = NULL;
    Bconst = NULL;
    Cconst = NULL;
    Dconst = NULL;
    u = NULL;
}

AllIZHNeurons::~AllIZHNeurons()
{
    freeResources();
}

/*
 *  Setup the internal structure of the class (allocate memories).
 *
 *  @param  sim_info  SimulationInfo class to read information from.
 */
void AllIZHNeurons::setupNeurons(SimulationInfo *sim_info)
{
    AllIFNeurons::setupNeurons(sim_info);

    Aconst = new BGFLOAT[size];
    Bconst = new BGFLOAT[size];
    Cconst = new BGFLOAT[size];
    Dconst = new BGFLOAT[size];
    u = new BGFLOAT[size];
}

/*
 *  Cleanup the class (deallocate memories).
 */
void AllIZHNeurons::cleanupNeurons()
{
    freeResources();
    AllIFNeurons::cleanupNeurons();
}

/*
 *  Deallocate all resources
 */
void AllIZHNeurons::freeResources()
{
    if (size != 0) {
        delete[] Aconst;
        delete[] Bconst;
        delete[] Cconst;
        delete[] Dconst;
        delete[] u;
    }

    Aconst = NULL;
    Bconst = NULL;
    Cconst = NULL;
    Dconst = NULL;
    u = NULL;
}

/*
 * Checks the number of required parameters.
 *
 * @return true if all required parameters were successfully read, false otherwise.
 */
bool AllIZHNeurons::checkNumParameters()
{
    return (nParams >= 12);
}

/*
 *  Attempts to read parameters from a XML file.
 *
 *  @param  element TiXmlElement to examine.
 *  @return true if successful, false otherwise.
 */
bool AllIZHNeurons::readParameters(const TiXmlElement& element)
{
    if (AllIFNeurons::readParameters(element)) {
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

/*
    if (element.ValueStr().compare("Aconst") == 0) {
        // Min/max values of Aconst for excitatory neurons.
        if (element.QueryFLOATAttribute("minExc", &m_excAconst[0]) != TIXML_SUCCESS) {
            throw ParseParamError("Aconst minExc", "Aconst missing minimum value in XML.");
        }
        if (m_excAconst[0] < 0) {
            throw ParseParamError("Aconst minExc", "Invalid negative Aconst value.");
        }
        if (element.QueryFLOATAttribute("maxExc", &m_excAconst[1]) != TIXML_SUCCESS) {
            throw ParseParamError("Aconst maxExc", "Aconst missing maximum value in XML.");
        }
        if (m_excAconst[0] > m_excAconst[1]) {
            throw ParseParamError("Aconst maxExc", "Invalid range for Aconst value.");
        }

        // Min/max values of Aconst for inhibitory neurons.
        if (element.QueryFLOATAttribute("minInh", &m_inhAconst[0]) != TIXML_SUCCESS) {
            throw ParseParamError("Aconst minInh", "Aconst missing minimum value in XML.");
        }
        if (m_inhAconst[0] < 0) {
            throw ParseParamError("Aconst minInh", "Invalid negative Aconst value.");
        }
        if (element.QueryFLOATAttribute("maxInh", &m_inhAconst[1]) != TIXML_SUCCESS) {
            throw ParseParamError("Aconst maxInh", "Aconst missing maximum value in XML.");
        }
        if (m_inhAconst[0] > m_inhAconst[1]) {
            throw ParseParamError("Aconst maxInh", "Invalid range for Aconst value.");
        }
        nParams++;
        return true;
    }

    if (element.ValueStr().compare("Bconst") == 0) {
        // Min/max values of Bconst for excitatory neurons.
        if (element.QueryFLOATAttribute("minExc", &m_excBconst[0]) != TIXML_SUCCESS) {
            throw ParseParamError("Bconst minExc", "Bconst missing minimum value in XML.");
        }
        if (element.QueryFLOATAttribute("maxExc", &m_excBconst[1]) != TIXML_SUCCESS) {
            throw ParseParamError("Bconst maxExc", "Bconst missing maximum value in XML.");
        }
        if (m_excBconst[0] > m_excBconst[1]) {
            throw ParseParamError("Bconst maxExc", "Invalid range for Bconst value.");
        }

        // Min/max values of Bconst for inhibitory neurons.
        if (element.QueryFLOATAttribute("minInh", &m_inhBconst[0]) != TIXML_SUCCESS) {
            throw ParseParamError("Bconst minInh", "Bconst missing minimum value in XML.");
        }
        if (element.QueryFLOATAttribute("maxInh", &m_inhBconst[1]) != TIXML_SUCCESS) {
            throw ParseParamError("Bconst maxInh", "Bconst missing maximum value in XML.");
        }
        if (m_inhBconst[0] > m_inhBconst[1]) {
            throw ParseParamError("Bconst maxInh", "Invalid range for Bconst value.");
        }
        nParams++;
        return true;
    }

    if (element.ValueStr().compare("Cconst") == 0) {
        // Min/max values of Cconst for excitatory neurons. 
        if (element.QueryFLOATAttribute("minExc", &m_excCconst[0]) != TIXML_SUCCESS) {
            throw ParseParamError("Cconst minExc", "Cconst missing minimum value in XML.");
        }
        if (element.QueryFLOATAttribute("maxExc", &m_excCconst[1]) != TIXML_SUCCESS) {
            throw ParseParamError("Cconst maxExc", "Cconst missing maximum value in XML.");
        }
        if (m_excCconst[0] > m_excCconst[1]) {
            throw ParseParamError("Cconst maxExc", "Invalid range for Cconst value.");
        }

        // Min/max values of Cconst for inhibitory neurons.
        if (element.QueryFLOATAttribute("minInh", &m_inhCconst[0]) != TIXML_SUCCESS) {
            throw ParseParamError("Cconst minInh", "Cconst missing minimum value in XML.");
        }
        if (element.QueryFLOATAttribute("maxInh", &m_inhCconst[1]) != TIXML_SUCCESS) {
            throw ParseParamError("Cconst maxInh", "Cconst missing maximum value in XML.");
        }
        if (m_inhCconst[0] > m_inhCconst[1]) {
            throw ParseParamError("Cconst maxInh", "Invalid range for Cconst value.");
        }
        nParams++;
        return true;
    }

    if (element.ValueStr().compare("Dconst") == 0) {
        // Min/max values of Dconst for excitatory neurons.
        if (element.QueryFLOATAttribute("minExc", &m_excDconst[0]) != TIXML_SUCCESS) {
            throw ParseParamError("Dconst minExc", "Dconst missing minimum value in XML.");
        }
        if (m_excDconst[0] < 0) {
            throw ParseParamError("Dconst minExc", "Invalid negative Dconst value.");
        }
        if (element.QueryFLOATAttribute("maxExc", &m_excDconst[1]) != TIXML_SUCCESS) {
            throw ParseParamError("Dconst maxExc", "Dconst missing maximum value in XML.");
        }
        if (m_excDconst[0] > m_excDconst[1]) {
            throw ParseParamError("Dconst maxExc", "Invalid range for Dconst value.");
        }

        // Min/max values of Dconst for inhibitory neurons.
        if (element.QueryFLOATAttribute("minInh", &m_inhDconst[0]) != TIXML_SUCCESS) {
            throw ParseParamError("Dconst minInh", "Dconst missing minimum value in XML.");
        }
        if (m_inhDconst[0] < 0) {
            throw ParseParamError("Dconst minInh", "Invalid negative Dconst value.");
        }
        if (element.QueryFLOATAttribute("maxInh", &m_inhDconst[1]) != TIXML_SUCCESS) {
            throw ParseParamError("Dconst maxInh", "Dconst missing maximum value in XML.");
        }
        if (m_inhDconst[0] > m_inhDconst[1]) {
            throw ParseParamError("Dconst maxInh", "Invalid range for Dconst value.");
        }
        nParams++;
        return true;
    }
*/
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
void AllIZHNeurons::printParameters(ostream &output) const
{
    AllIFNeurons::printParameters(output);

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
 *  Creates all the Neurons and generates data for them.
 *
 *  @param  sim_info    SimulationInfo class to read information from.
 *  @param  layout      Layout information of the neunal network.
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

/*
 *  Creates a single Neuron and generates data for it.
 *
 *  @param  sim_info     SimulationInfo class to read information from.
 *  @param  neuron_index Index of the neuron to create.
 *  @param  layout       Layout information of the neunal network.
 */
void AllIZHNeurons::createNeuron(SimulationInfo *sim_info, int neuron_index, Layout *layout)
{
    // set the neuron info for neurons
    AllIFNeurons::createNeuron(sim_info, neuron_index, layout);

    // TODO: we may need another distribution mode besides flat distribution
    if (layout->neuron_type_map[neuron_index] == EXC) {
        // excitatory neuron
        Aconst[neuron_index] = rng.inRange(m_excAconst[0], m_excAconst[1]); 
        Bconst[neuron_index] = rng.inRange(m_excBconst[0], m_excBconst[1]); 
        Cconst[neuron_index] = rng.inRange(m_excCconst[0], m_excCconst[1]); 
        Dconst[neuron_index] = rng.inRange(m_excDconst[0], m_excDconst[1]); 
    } else {
        // inhibitory neuron
        Aconst[neuron_index] = rng.inRange(m_inhAconst[0], m_inhAconst[1]); 
        Bconst[neuron_index] = rng.inRange(m_inhBconst[0], m_inhBconst[1]); 
        Cconst[neuron_index] = rng.inRange(m_inhCconst[0], m_inhCconst[1]); 
        Dconst[neuron_index] = rng.inRange(m_inhDconst[0], m_inhDconst[1]); 
    }
	// IZH03: initial value u = b*v instead of = 0 
    u[neuron_index] = Bconst[neuron_index] * Vm[neuron_index] * 1000;

    DEBUG_HI(cout << "CREATE NEURON[" << neuron_index << "] {" << endl
            << "\tAconst = " << Aconst[neuron_index] << endl
            << "\tBconst = " << Bconst[neuron_index] << endl
            << "\tCconst = " << Cconst[neuron_index] << endl
            << "\tDconst = " << Dconst[neuron_index] << endl
            << "}" << endl
    ;)

}

/*
 *  Set the Neuron at the indexed location to default values.
 *
 *  @param  neuron_index    Index of the Neuron to refer.
 */
void AllIZHNeurons::setNeuronDefaults(const int index)
{
    AllIFNeurons::setNeuronDefaults(index);

    Aconst[index] = DEFAULT_a;
    Bconst[index] = DEFAULT_b;
    Cconst[index] = DEFAULT_c;
    Dconst[index] = DEFAULT_d;
}

/*
 *  Initializes the Neuron constants at the indexed location.
 *
 *  @param  neuron_index    Index of the Neuron.
 *  @param  deltaT          Inner simulation step duration
 */
void AllIZHNeurons::initNeuronConstsFromParamValues(int neuron_index, const BGFLOAT deltaT)
{
    AllIFNeurons::initNeuronConstsFromParamValues(neuron_index, deltaT);
}

/*
 *  Outputs state of the neuron chosen as a string.
 *
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
    return ss.str( );
}

/*
 *  Sets the data for Neurons to input's data.
 *
 *  @param  input       istream to read from.
 *  @param  sim_info    used as a reference to set info for neurons.
 */
void AllIZHNeurons::deserialize(istream &input, const SimulationInfo *sim_info)
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
void AllIZHNeurons::readNeuron(istream &input, const SimulationInfo *sim_info, int i)
{
    AllIFNeurons::readNeuron(input, sim_info, i);

    input >> Aconst[i]; input.ignore();
    input >> Bconst[i]; input.ignore();
    input >> Cconst[i]; input.ignore();
    input >> Dconst[i]; input.ignore();
    input >> u[i]; input.ignore();
}

/*
 *  Writes out the data in Neurons.
 *
 *  @param  output      stream to write out to.
 *  @param  sim_info    used as a reference to set info for neuronss.
 */
void AllIZHNeurons::serialize(ostream& output, const SimulationInfo *sim_info) const 
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
void AllIZHNeurons::writeNeuron(ostream& output, const SimulationInfo *sim_info, int i) const
{
    AllIFNeurons::writeNeuron(output, sim_info, i);

    output << Aconst[i] << ends;
    output << Bconst[i] << ends;
    output << Cconst[i] << ends;
    output << Dconst[i] << ends;
    output << u[i] << ends;
}

#if !defined(USE_GPU)
/*
 *  Update internal state of the indexed Neuron (called by every simulation step).
 * 	
 *  @param  index       Index of the Neuron to update.
 *  @param  sim_info    SimulationInfo class to read information from.
 * 
 *  [IZH03] Modifications made to match Izhikevich matlab implementation:
 *  - remove I0 
 *  - use Simple Euler function instead of Exponential Euler
 * 		- remove C1, C2	
 *  - no refractory period: nStepsInRefr = 0 
 *  - unit conversion V -> mV, s -> ms to match dimensionless izh integration 
 */
void AllIZHNeurons::advanceNeuron(const int index, const SimulationInfo *sim_info)
{
    const BGFLOAT deltaT = sim_info->deltaT;
    BGFLOAT &Vm = this->Vm[index];
    BGFLOAT &Vthresh = this->Vthresh[index];
    BGFLOAT &summationPoint = this->summation_map[index];
    //BGFLOAT &I0 = this->I0[index];
    BGFLOAT &Inoise = this->Inoise[index];
    //BGFLOAT &C1 = this->C1[index];
    //BGFLOAT &C2 = this->C2[index];
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
        // add noisy thalamic input
        BGFLOAT noise = (*rgNormrnd[0])();
        //DEBUG_MID(cout << "ADVANCE NEURON[" << index << "] :: noise = " << noise << endl;)
        summationPoint += noise * Inoise; 
        
		// IZH03: Izhikevich model integration step
        BGFLOAT Vint = Vm * 1000; 
		Vint = Vint + deltaT * 1000 * (0.04 * Vint * Vint + 5 * Vint + 140 - u + summationPoint);
        u = u + a * (b * Vint - u);
        Vm = Vint * 0.001 ; 
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
            << "\tInoise = " << Inoise << endl
            << "}" << endl
    ;)

    // clear synaptic input for next time step
    summationPoint = 0;
}

/*
 *  Fire the selected Neuron and calculate the result.
 *
 *  @param  index       Index of the Neuron to update.
 *  @param  sim_info    SimulationInfo class to read information from.
 */
void AllIZHNeurons::fire(const int index, const SimulationInfo *sim_info) const
{
    //const BGFLOAT deltaT = sim_info->deltaT;
    AllSpikingNeurons::fire(index, sim_info);

    // calculate the number of steps in the absolute refractory period
    BGFLOAT &Vm = this->Vm[index];
    int &nStepsInRefr = this->nStepsInRefr[index];
    //BGFLOAT &Trefract = this->Trefract[index];

    BGFLOAT &c = Cconst[index];
    BGFLOAT &d = Dconst[index];
    BGFLOAT &u = this->u[index];
	// IZH03: make nStepsInRefr = 0; no refractory period
    // nStepsInRefr = static_cast<int> ( Trefract / deltaT + 0.5 );
    nStepsInRefr = 0; 

    // reset to 'Vreset'
    Vm = c * 0.001;
    u = u + d;

}
#endif
