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
bool LoadAllParameters(const string &sim_param_filename);
void printParams();
bool parseCommandLine(int argc, char* argv[]);
bool createAllClassInstances(TiXmlDocument* simDoc);

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

    if (!LoadAllParameters(stateInputFileName)) {
        cerr << "! ERROR: failed while parsing simulation parameters." << endl;
        return -1;
    }

    if (stateOutputFileName.empty()) {
        if (!simInfo->stateOutputFileName.empty()) {
            stateOutputFileName = simInfo->stateOutputFileName;
        }
        else {
            cerr << "! ERROR: no stateOutputFileName is specified." << endl;
            return -1;
        }
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

    time_t start_time, end_time;
    time(&start_time);

    // create the simulator
    Simulator *simulator;
    simulator = new Simulator(model, simRecorder, pInput, simInfo);
	
    // setup simulation
    DEBUG(cout << "Setup simulation." << endl;);
    simulator->setup();

    // Deserializes internal state from a prior run of the simulation
    if (fReadMemImage) {
        ifstream memory_in;
        memory_in.open(memInputFileName.c_str(), ofstream::binary | ofstream::in);
        simulator->deserialize(memory_in);
        memory_in.close();
    }

    // Run simulation
    simulator->simulate();

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

    // Tell simulation to clean-up and run any post-simulation logic.
    simulator->finish();

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

bool createAllClassInstances(TiXmlDocument* simDoc)
{
    TiXmlElement* parms = NULL;

    if ((parms = simDoc->FirstChildElement("ModelParams")) == NULL) {
        cerr << "Could not find <MoelParms> in simulation parameter file " << endl;
        return false;
    }

    // create neurons, synapses, connections, and layout objects specified in the description file
    neurons = FClassOfCategory::get()->createNeurons(parms);
    synapses = FClassOfCategory::get()->createSynapses(parms);
    conns = FClassOfCategory::get()->createConnections(parms);
    layout = FClassOfCategory::get()->createLayout(parms);

    if (neurons == NULL || synapses == NULL || conns == NULL || layout == NULL) {
        cerr << "!ERROR: failed to create classes" << endl;
        return false;
    }

    return true;
}

/*
 *  Load parameters from a file.
 *
 *  @param  sim_param_filename  filename of file to read from
 *  @return true if successful, false if not
 */
bool LoadAllParameters(const string &sim_param_filename)
{
    DEBUG(cout << "reading parameters from xml file" << endl;)

    TiXmlDocument simDoc(sim_param_filename.c_str());
    if (!simDoc.LoadFile()) {
        cerr << "Failed loading simulation parameter file "
             << sim_param_filename << ":" << "\n\t" << simDoc.ErrorDesc()
             << endl;
        cerr << " error: " << simDoc.ErrorRow() << ", " << simDoc.ErrorCol()
             << endl;
        return false;
    }

    // create simulation info object
    simInfo = new SimulationInfo();

    // load simulation parameters
    if (simInfo->readParameters(&simDoc) != true) {
        return false;
    }

    // create instances of all model classes
    DEBUG(cout << "creating instances of all classes" << endl;)
    if (createAllClassInstances(&simDoc) != true) {
        return false;
    }

    // load parameters for all models
    return FClassOfCategory::get()->readParameters(&simDoc);
}

/*
 *  Prints loaded parameters out to console.
 */
void printParams() {
    cout << "\nPrinting simulation parameters...\n";
    simInfo->printParameters(cout);

    cout << "Model Parameters:" << endl;
    FClassOfCategory::get()->printParameters(cout);
    cout << "Done printing parameters" << endl;
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

    if (!memInputFileName.empty()) {
        fReadMemImage = true;
    }

    if (!memOutputFileName.empty()) {
        fWriteMemImage = true;
    }

#if defined(USE_GPU)
    if (EOF == sscanf(cl["deviceid"].c_str(), "%d", &g_deviceId)) {
        g_deviceId = 0;
    }
#endif  // USE_GPU

    return true;
}
