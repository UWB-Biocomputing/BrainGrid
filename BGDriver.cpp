/*
 *  The driver for braingrid.
 *  The driver performs the following steps:
 *  1) reads parameters from an xml file (specified as the first argument)
 *  2) creates the network
 *  3) launches the simulation
 *
 *  @authors Allan Ortiz and Cory Mayberry.
 */

#include <fstream>
#include "Global.h"
#include "paramcontainer/ParamContainer.h"

#include "Network.h"
#include "IModel.h"
#include "FClassOfCategory.h"
#include "IRecorder.h"
#include "FSInput.h"
#include "Simulator.h"

// Uncomment to use visual leak detector (Visual Studios Plugin)
// #include <vld.h>

#if defined(USE_GPU)
    #include "GPUSpikingModel.h"
#elif defined(USE_OMP)
//    #include "MultiThreadedSim.h"
#else 
    #include "SingleThreadedSpikingModel.h"
#endif

using namespace std;

// state file name
string stateOutputFileName;
string stateInputFileName;

// memory dump file name
string memOutputFileName;
string memInputFileName;
bool fReadMemImage = false;  // True if dumped memory image is read before
                             // starting simulation
bool fWriteMemImage = false;  // True if dumped memory image is written after
                              // simulation

// stimulus input file name
string stimulusInputFileName;

IModel *model = NULL;
IAllNeurons *neurons = NULL;
IAllSynapses *synapses = NULL;
Connections *conns = NULL;
Layout *layout = NULL;

SimulationInfo *simInfo = NULL;


// functions
SimulationInfo makeSimulationInfo(int cols, int rows,
    BGFLOAT growthEpochDuration, BGFLOAT maxGrowthSteps,
    int maxFiringRate, int maxSynapsesPerNeuron, BGFLOAT new_deltaT,
    long seed);
bool LoadAllParameters(const string &sim_param_filename);
void LoadSimulationParameters(TiXmlElement*);
//void SaveSimState(ostream &);
void printParams();
bool parseCommandLine(int argc, char* argv[]);
bool createAllClassInstances(TiXmlElement*);

/*
 *  Main for Simulator. Handles command line arguments and loads parameters
 *  from parameter file. All initial loading before running simulator in Network
 *  is here.
 *
 *  @param  argc    argument count.
 *  @param  argv    arguments.
 *  @return -1 if error, else if success.
 */
int main(int argc, char* argv[]) {
    if (!parseCommandLine(argc, argv)) {
        cerr << "! ERROR: failed during command line parse" << endl;
        return -1;
    }

    DEBUG(cout << "reading parameters from xml file" << endl;)

    if (!LoadAllParameters(stateInputFileName)) {
        cerr << "! ERROR: failed while parsing simulation parameters." << endl;
        return -1;
    }

    /*    verify that params were read correctly */
    DEBUG(printParams();)

    // create the model
    #if defined(USE_GPU)
	 model = new GPUSpikingModel(conns, neurons, synapses, layout);
    #else
	 model = new SingleThreadedSpikingModel(conns, neurons, synapses, layout);
    #endif
    
    // create & init simulation recorder
    IRecorder* simRecorder = conns->createRecorder(stateOutputFileName, model, simInfo);

    if (simRecorder == NULL) {
        cerr << "! ERROR: invalid state output file name extension." << endl;
        return -1;
    }

    // Create a stimulus input object
    ISInput* pInput = NULL;     // pointer to a stimulus input object
    pInput = FSInput::get()->CreateInstance(model, simInfo, stimulusInputFileName);

    // create the network
    Network network(model, simInfo, simRecorder);

    time_t start_time, end_time;
    time(&start_time);

    // create the simulator
    Simulator *simulator;
    simulator = new Simulator(&network, simInfo);
	
    // setup simulation
    DEBUG(cout << "Setup simulation." << endl;);
    network.setup(pInput);

    // Deserializes internal state from a prior run of the simulation
    if (fReadMemImage) {
        ifstream memory_in;
        memory_in.open(memInputFileName.c_str(), ofstream::binary | ofstream::in);
        simulator->deserialize(memory_in);
        memory_in.close();
    }

    // Run simulation
    simulator->simulate(pInput);

    // Terminate the stimulus input 
    if (pInput != NULL)
    {
        pInput->term(model, simInfo);
        delete pInput;
    }

    // Writes simulation results to an output destination
    simulator->saveData();

    // Serializes internal state for the current simulation
    ofstream memory_out;
    if (fWriteMemImage) {
        memory_out.open(memOutputFileName.c_str(),ofstream::binary | ofstream::trunc);
        simulator->serialize(memory_out);
        memory_out.close();
    }

    // Tell network to clean-up and run any post-simulation logic.
    network.finish();

    // terminates the simulation recorder
    if (simRecorder != NULL) {
        simRecorder->term();
    }

    for(unsigned int i = 0; i < rgNormrnd.size(); ++i) {
        delete rgNormrnd[i];
    }

    rgNormrnd.clear();

    time(&end_time);
    double time_elapsed = difftime(end_time, start_time);
    double ssps = simInfo->epochDuration * simInfo->maxSteps / time_elapsed;
    cout << "time simulated: " << simInfo->epochDuration * simInfo->maxSteps << endl;
    cout << "time elapsed: " << time_elapsed << endl;
    cout << "ssps (simulation seconds / real time seconds): " << ssps << endl;
    
    delete model;
    model = NULL;
    
    if (simRecorder != NULL) {
        delete simRecorder;
        simRecorder = NULL;
    }

    delete simInfo;
    simInfo = NULL;

    delete simulator;
    simulator = NULL;

    return 0;
}

/*
 *  Init SimulationInfo parameters.
 *
 *  @param  cols    number of columns for the simulation.
 *  @param  rows    number of rows for the simulation.
 *  @param  growthEpochDuration  duration in between each growth.
 *  @param  maxGrowthSteps  TODO
 *  @param  maxFiringRate   maximum firing rate for the simulation.
 *  @param  maxSynapsesPerNeuron    cap limit for the number of Synapses each Neuron can have.
 *  @param  new_deltaT  TODO (model independent)
 *  @param  seed    seeding for random numbers.
 *  @return SimulationInfo object encapsulating info given.
 */
SimulationInfo makeSimulationInfo(int cols, int rows,
        BGFLOAT growthEpochDuration, BGFLOAT maxGrowthSteps,
        int maxFiringRate, int maxSynapsesPerNeuron, BGFLOAT new_deltaT,
        long seed)
{
    SimulationInfo simInfo;
    // Init SimulationInfo parameters
    int max_neurons = cols * rows;

    simInfo.totalNeurons = max_neurons;
    simInfo.epochDuration = growthEpochDuration;
    simInfo.maxSteps = (int)maxGrowthSteps;

    // May be model-dependent
    simInfo.maxFiringRate = maxFiringRate;
    simInfo.maxSynapsesPerNeuron = maxSynapsesPerNeuron;

// NETWORK MODEL VARIABLES NMV-BEGIN {
    simInfo.width = cols;
    simInfo.height = rows;
// } NMV-END

    simInfo.deltaT = new_deltaT;  // Model Independent
    simInfo.seed = seed;  // Model Independent

    return simInfo;
}

/*
 *  Prints loaded parameters out to console.
 */
void printParams() {
    cout << "\nPrinting parameters...\n";
    cout << "poolsize x:" << simInfo->width
         << " y:" << simInfo->height
         //z dimmension is for future expansion and not currently supported
         //<< " z:" <<
         << endl;
    cout << "Simulation Parameters:\n";
    cout << "\tTime between growth updates (in seconds): " << simInfo->epochDuration << endl;
    cout << "\tNumber of simulations to run: " << simInfo->maxSteps << endl;

    cout << "Model Parameters:" << endl;
    FClassOfCategory::get()->printParameters(cout);
    cout << "Done printing parameters" << endl;
}

bool createAllClassInstances(TiXmlElement* parms)
{
    // create neurons, synapses, connections, and layout objects specified in the description file
    neurons = FClassOfCategory::get()->createNeurons(parms);
    synapses = FClassOfCategory::get()->createSynapses(parms);
    conns = FClassOfCategory::get()->createConnections(parms);
    layout = FClassOfCategory::get()->createLayout(parms);

    if (neurons == NULL || synapses == NULL || conns == NULL || layout == NULL) {
        cerr << "!ERROR: failed to create classes" << endl;
        return false;
    }

    return FClassOfCategory::get()->readParameters(parms);
}

/*
 *  Load parameters from a file.
 *
 *  @param  sim_param_filename  filename of file to read from
 *  @return true if successful, false if not
 */
bool LoadAllParameters(const string &sim_param_filename)
{
    TiXmlDocument simDoc(sim_param_filename.c_str());
    if (!simDoc.LoadFile()) {
        cerr << "Failed loading simulation parameter file "
             << sim_param_filename << ":" << "\n\t" << simDoc.ErrorDesc()
             << endl;
        cerr << " error: " << simDoc.ErrorRow() << ", " << simDoc.ErrorCol()
             << endl;
        return false;
    }

    TiXmlElement* parms = NULL;

    if ((parms = simDoc.FirstChildElement("SimParams")) == NULL) {
        cerr << "Could not find <SimParms> in simulation parameter file "
             << sim_param_filename << endl;
        return false;
    }

    try {
        LoadSimulationParameters(parms);
    } catch (KII_exception &e) {
        cerr << "Failure loading simulation parameters from file "
             << sim_param_filename << ":\n\t" << e.what()
             << endl;
        return false;
    }

    if (createAllClassInstances(parms) != true) {
        return false;
    }
 
    return true;
}

/*
 *  Handles loading of parameters using tinyxml from the parameter file.
 *
 *  @param  parms   tinyxml element to load from.
 */
void LoadSimulationParameters(TiXmlElement* parms)
{
    
    TiXmlElement* temp = NULL;

    // Simulation Parameters
    int poolsize[2];  // size of pool of neurons [x y z]
                      //z currently not supported
    BGFLOAT Tsim;  // Simulation time (s) (between growth updates) rename: epochLength
    int numSims;  // Number of Tsim simulation to run
    int maxFiringRate;  // Maximum firing rate (only used by GPU version)
    int maxSynapsesPerNeuron;  // Maximum number of synapses per neuron
                               // (only used by GPU version)
    long seed;  // Seed for random generator (single-threaded)


    // Flag indicating that the variables were correctly read
    bool fSet = true;

    // Note that we just grab the first child with the right value;
    // multiple children with the same values are ignored. This might
    // not be as quick as iterating through the children and setting the
    // parameters as each one's element is found, but the code is
    // simpler this way and the performance penalty is insignificant.

    if ((temp = parms->FirstChildElement("PoolSize")) != NULL) {
        if (temp->QueryIntAttribute("x", &poolsize[0]) != TIXML_SUCCESS) {
            fSet = false;
            cerr << "error x" << endl;
        }
        if (temp->QueryIntAttribute("y", &poolsize[1]) != TIXML_SUCCESS) {
            fSet = false;
            cerr << "error y" << endl;
        }

        //z dimmension is for future expansion and not currently supported
        /*if (temp->QueryIntAttribute("z", &poolsize[2]) != TIXML_SUCCESS) {
            fSet = false;
            cerr << "error z" << endl;
        }*/
    } else {
        fSet = false;
        cerr << "missing PoolSize" << endl;
    }

    if ((temp = parms->FirstChildElement("SimParams")) != NULL) {
        if (temp->QueryFLOATAttribute("Tsim", &Tsim) != TIXML_SUCCESS) {
            fSet = false;
            cerr << "error Tsim" << endl;
        }
        if (temp->QueryIntAttribute("numSims", &numSims) != TIXML_SUCCESS) {
            fSet = false;
            cerr << "error numSims" << endl;
        }
        if (temp->QueryIntAttribute("maxFiringRate", &maxFiringRate) != TIXML_SUCCESS) {
            fSet = false;
            cerr << "error maxFiringRate" << endl;
        }
        if (temp->QueryIntAttribute("maxSynapsesPerNeuron", &maxSynapsesPerNeuron) != TIXML_SUCCESS) {
            fSet = false;
            cerr << "error maxSynapsesPerNeuron" << endl;
        }
    } else {
        fSet = false;
        cerr << "missing SimParams" << endl;
    }

    if ((temp = parms->FirstChildElement("OutputParams")) != NULL) {
        if (stateOutputFileName.empty() 
            && (temp->QueryValueAttribute("stateOutputFileName", &stateOutputFileName) != TIXML_SUCCESS)) {
            fSet = false;
            cerr << "error stateOutputFileName" << endl;
        }
    } else {
        fSet = false;
        cerr << "missing OutputParams" << endl;
    }

    if ((temp = parms->FirstChildElement("Seed")) != NULL) {
        if (temp->QueryValueAttribute("value", &seed) != TIXML_SUCCESS) {
            fSet = false;
            cerr << "error value" << endl;
        }
    } else {
        fSet = false;
        cerr << "missing Seed" << endl;
    }

    // Ideally, an error message would be output for each failed Query
    // above, but that's just too much code for me right now.
    if (!fSet) throw KII_exception("Failed to initialize one or more simulation parameters; check XML");

    simInfo = new SimulationInfo(makeSimulationInfo(poolsize[0], poolsize[1],Tsim, numSims,
            maxFiringRate, maxSynapsesPerNeuron, DEFAULT_dt, seed));
}

/*
 *  Handles parsing of the command line
 *
 *  @param  argc    argument count.
 *  @param  argv    arguments.
 *  @returns    true if successful, false otherwise.
 */
bool parseCommandLine(int argc, char* argv[])
{
    ParamContainer cl;
    cl.initOptions(false);  // don't allow unknown parameters
    cl.setHelpString(string("The DCT growth modeling simulator\nUsage: ") + argv[0] + " ");

#if defined(USE_GPU)
    if ((cl.addParam("stateoutfile", 'o', ParamContainer::filename, "simulation state output filename") != ParamContainer::errOk)
            || (cl.addParam("stateinfile", 't', ParamContainer::filename | ParamContainer::required, "simulation state input filename") != ParamContainer::errOk)
            || (cl.addParam("deviceid", 'd', ParamContainer::regular, "CUDA device id") != ParamContainer::errOk)
            || (cl.addParam( "stiminfile", 's', ParamContainer::filename, "stimulus input file" ) != ParamContainer::errOk)
            || (cl.addParam("meminfile", 'r', ParamContainer::filename, "simulation memory image input filename") != ParamContainer::errOk)
            || (cl.addParam("memoutfile", 'w', ParamContainer::filename, "simulation memory image output filename") != ParamContainer::errOk)) {
        cerr << "Internal error creating command line parser" << endl;
        return false;
    }
#else    // !USE_GPU
    if ((cl.addParam("stateoutfile", 'o', ParamContainer::filename, "simulation state output filename") != ParamContainer::errOk)
            || (cl.addParam("stateinfile", 't', ParamContainer::filename | ParamContainer::required, "simulation state input filename") != ParamContainer::errOk)
            || (cl.addParam( "stiminfile", 's', ParamContainer::filename, "stimulus input file" ) != ParamContainer::errOk)
            || (cl.addParam("meminfile", 'r', ParamContainer::filename, "simulation memory image filename") != ParamContainer::errOk)
            || (cl.addParam("memoutfile", 'w', ParamContainer::filename, "simulation memory image output filename") != ParamContainer::errOk)) {
        cerr << "Internal error creating command line parser" << endl;
        return false;
    }
#endif  // USE_GPU

    // Parse the command line
    if (cl.parseCommandLine(argc, argv) != ParamContainer::errOk) {
        cl.dumpHelp(stderr, true, 78);
        return false;
    }

    // Get the values
    stateOutputFileName = cl["stateoutfile"];
    stateInputFileName = cl["stateinfile"];
    memInputFileName = cl["meminfile"];
    memOutputFileName = cl["memoutfile"];
    stimulusInputFileName = cl["stiminfile"];

    if (!memInputFileName.empty())
        fReadMemImage = true;
    if (!memOutputFileName.empty())
        fWriteMemImage = true;
#if defined(USE_GPU)
    if (EOF == sscanf(cl["deviceid"].c_str(), "%d", &g_deviceId))
        g_deviceId = 0;
#endif  // USE_GPU
    return true;
}
