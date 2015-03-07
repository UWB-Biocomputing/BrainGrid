#include "Model.h"

#include "../tinyxml/tinyxml.h"

#include "ParseParamError.h"
#include "Util.h"


/**
 *  Constructor
 */
Model::Model(Connections *conns, AllNeurons *neurons, AllSynapses *synapses, Layout *layout) :
    m_read_params(0),
    m_conns(conns),
    m_neurons(neurons),
    m_synapses(synapses),
    m_layout(layout)
{
}

/**
 *  Destructor
 */
Model::~Model()
{
    if (m_conns != NULL) {
        delete m_conns;
        m_conns = NULL;
    }

    if (m_neurons != NULL) {
        delete m_neurons;
        m_neurons = NULL;
    }

    if (m_synapses != NULL) {
        delete m_synapses;
        m_synapses = NULL;
    }

    if (m_layout != NULL) {
        delete m_layout;
        m_layout = NULL;
    }
}

/**
 *  Attempts to read parameters from a XML file.
 *  @param  source  the TiXmlElement to read from.
 *  @return true if successful, false otherwise.
 */
bool Model::readParameters(TiXmlElement *source)
{
    m_read_params = 0;
    try {
         source->Accept(this);
    } catch (ParseParamError &error) {
        error.print(cerr);
        cerr << endl;
        return false;
    }
    
    return m_read_params == m_neurons->numParameters();
}

/**
 *  Takes an XmlElement and checks for errors. If not, calls getValueList().
 *  @param  element TiXmlElement to examine.
 *  @param  firstAttribute  ***NOT USED***.
 *  @return true if method finishes without errors.
 */
bool Model::VisitEnter(const TiXmlElement& element, const TiXmlAttribute* firstAttribute)
//TODO: firstAttribute does not seem to be used! Delete?
{
    // Read neurons parameters
    m_read_params += m_neurons->readParameters(element);
    
    // Read connections parameters (growth parameters)
    if (m_conns->readParameters(element) != true) {
        throw ParseParamError("Connections", "Failed in readParameters.");
    }

    // Read layout parameters
    if (m_layout->readParameters(element) != true) {
        throw ParseParamError("Layout", "Failed in readParameters.");
    }

    return true;
}

/**
 *  Prints out all parameters of the model to ostream.
 *  @param  output  ostream to send output to.
 */
void Model::printParameters(ostream &output) const
{
    // Prints all neurons parameters
    m_neurons->printParameters(output);

    // Prints all connections parameters
    m_conns->printParameters(output);

    // Prints all layout parameters
    m_layout->printParameters(output);
}

/**
 *  Loads the simulation based on istream input.
 *  @param  input   istream to read from.
 *  @param  sim_info    used as a reference to set info for neurons and synapses.
 */
void Model::loadMemory(istream& input, const SimulationInfo *sim_info)
{
    // read the neurons data & create neurons
    m_neurons->readNeurons(input, sim_info);

    // read the synapse data & create synapses
    m_synapses->readSynapses(input, *m_neurons, sim_info);

    // read the connections data
    m_conns->readConns(input, sim_info);
}

/**
 *  Write the simulation's memory image.
 *  @param  output  the filestream to write.
 *  @param  simulation_step the step of the simulation at the current time.
 */
void Model::saveMemory(ostream& output, const SimulationInfo *sim_info)
{
    // write the neurons data
    output << sim_info->totalNeurons << ends;
    m_neurons->writeNeurons(output, sim_info);

    // write the synapse data
    m_synapses->writeSynapses(output, sim_info);

    // write the connections data
    m_conns->writeConns(output, sim_info);

    output << flush;
}

/**
 *  Prepares a stream with data from the model and Neurons.
 */
void Model::saveState(IRecorder* simRecorder)
{
    simRecorder->saveSimState(*m_neurons);

}

/**
 *  Creates all the Neurons and generates data for them.
 *  @param  sim_info    SimulationInfo class to read information from.
 */
void Model::createAllNeurons(SimulationInfo *sim_info)
{
    DEBUG(cout << "\nAllocating neurons..." << endl;)

    // init neuron's map with layout
    m_layout->generateNeuronTypeMap(m_neurons->neuron_type_map, sim_info->totalNeurons);
    m_layout->initStarterMap(m_neurons->starter_map, sim_info->totalNeurons, m_neurons->neuron_type_map);

    // set their specific types
    m_neurons->createAllNeurons(sim_info);

    DEBUG(cout << "Done initializing neurons..." << endl;)
}

/**
 *  Sets up the Simulation.
 *  @param  sim_info    SimulationInfo class to read information from.
 *  @param  simRecorder Pointer to the simulation recordig object.
 */
void Model::setupSim(SimulationInfo *sim_info, IRecorder* simRecorder)
{
    m_neurons->setupNeurons(sim_info);
    m_synapses->setupSynapses(sim_info);
    m_conns->setupConnections(sim_info);

    // Init radii and rates history matrices with default values
    simRecorder->initDefaultValues(m_conns->m_growth.startRadius);

    // Creates all the Neurons and generates data for them.
    createAllNeurons(sim_info);
}

/**
 *  Clean up the simulation.
 *  @param  sim_info    SimulationInfo to refer.
 */
void Model::cleanupSim(SimulationInfo *sim_info)
{
    m_neurons->cleanupNeurons();
    m_synapses->cleanupSynapses();
    m_conns->cleanupConnections();
}

/**
 *  Log this simulation step.
 *  @param  sim_info    SimulationInfo to reference.
 */
void Model::logSimStep(const SimulationInfo *sim_info) const
{
    cout << "format:\ntype,radius,firing rate" << endl;

    for (int y = 0; y < sim_info->height; y++) {
        stringstream ss;
        ss << fixed;
        ss.precision(1);

        for (int x = 0; x < sim_info->width; x++) {
            switch (m_neurons->neuron_type_map[x + y * sim_info->width]) {
            case EXC:
                if (m_neurons->starter_map[x + y * sim_info->width])
                    ss << "s";
                else
                    ss << "e";
                break;
            case INH:
                ss << "i";
                break;
            case NTYPE_UNDEF:
                assert(false);
                break;
            }

            ss << " " << (*m_conns->radii)[x + y * sim_info->width];

            if (x + 1 < sim_info->width) {
                ss.width(2);
                ss << "|";
                ss.width(2);
            }
        }

        ss << endl;

        for (int i = ss.str().length() - 1; i >= 0; i--) {
            ss << "_";
        }

        ss << endl;
        cout << ss.str();
    }
}

/**
 *  Update the Neuron's history.
 *  @param  currentStep 	current step of the simulation
 *  @param  epochDuration    	duration of the epoch
 *  @param  sim_info  		Pointer to the simulation information.
 *  @param  simRecorder 	Pointer to the simulation recordig object.
 */
void Model::updateHistory(const int currentStep, BGFLOAT epochDuration, const SimulationInfo *sim_info, IRecorder* simRecorder)
{
    // Update Connections data
    m_conns->updateConns(*m_neurons, sim_info);

    // Compile history information in every epoch
    simRecorder->compileHistories(*m_neurons, m_conns->m_growth.minRadius);
}

AllNeurons* Model::getNeurons()
{
    return m_neurons;
}

Connections* Model::getConnections()
{
    return m_conns;
}

Layout* Model::getLayout()
{
    return m_layout;
}
