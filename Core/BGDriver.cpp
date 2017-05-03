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
#include "ParamContainer.h"

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

// functions
bool LoadAllParameters(SimulationInfo *simInfo);
void printParams(SimulationInfo *simInfo);
bool parseCommandLine(int argc, char* argv[], SimulationInfo *simInfo);
bool createAllModelClassInstances(TiXmlDocument* simDoc, SimulationInfo *simInfo);

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
    SimulationInfo *simInfo = NULL;    // simulation information
    Simulator *simulator = NULL;       // Simulator object

    // create simulation info object
    simInfo = new SimulationInfo();

    // Handles parsing of the command line
    if (!parseCommandLine(argc, argv, simInfo)) {
        cerr << "! ERROR: failed during command line parse" << endl;
        return -1;
    }

    // Create all model instances and load parameters from a file.
    if (!LoadAllParameters(simInfo)) {
        cerr << "! ERROR: failed while parsing simulation parameters." << endl;
        return -1;
    }

    // create & init simulation recorder
    simInfo->simRecorder = simInfo->model->getConnections()->createRecorder(simInfo);
    if (simInfo->simRecorder == NULL) {
        cerr << "! ERROR: invalid state output file name extension." << endl;
        return -1;
    }

    // Create a stimulus input object
    simInfo->pInput = FSInput::get()->CreateInstance(simInfo);

    time_t start_time, end_time;
    time(&start_time);

    // create the simulator
    simulator = new Simulator();
	
    // setup simulation
    DEBUG(cerr << "Setup simulation." << endl;)
    simulator->setup(simInfo);

    // Deserializes internal state from a prior run of the simulation
    if (!simInfo->memInputFileName.empty()) {
        DEBUG(cerr << "Deserializing state from file." << endl;)
        ifstream memory_in;
        memory_in.open(simInfo->memInputFileName.c_str(), ofstream::binary | ofstream::in);
        simulator->deserialize(memory_in, simInfo);
        memory_in.close();
    }

    // Run simulation
    simulator->simulate(simInfo);

    // Terminate the stimulus input 
    if (simInfo->pInput != NULL)
    {
        simInfo->pInput->term(simInfo);
        delete simInfo->pInput;
    }

    // Writes simulation results to an output destination
    simulator->saveData(simInfo);

    // Serializes internal state for the current simulation
    ofstream memory_out;
    if (!simInfo->memOutputFileName.empty()) {
        memory_out.open(simInfo->memOutputFileName.c_str(),ofstream::binary | ofstream::trunc);
        simulator->serialize(memory_out, simInfo);
        memory_out.close();
    }

    // Tell simulation to clean-up and run any post-simulation logic.
    simulator->finish(simInfo);

    // terminates the simulation recorder
    if (simInfo->simRecorder != NULL) {
        simInfo->simRecorder->term();
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
    
    delete simInfo->model;
    simInfo->model = NULL;
    
    if (simInfo->simRecorder != NULL) {
        delete simInfo->simRecorder;
        simInfo->simRecorder = NULL;
    }

    delete simInfo;
    simInfo = NULL;

    delete simulator;
    simulator = NULL;

    return 0;
}

/*
 *  Create instances of all model classes.
 *
 *  @param  simDoc  the TiXmlDocument to read from.
 *  @param  simInfo   SimulationInfo class to read information from.
 *  @retrun true if successful, false if not
 */
bool createAllModelClassInstances(TiXmlDocument* simDoc, SimulationInfo *simInfo)
{
    TiXmlElement* parms = NULL;

    //cout << "Child:" <<  simDoc->FirstChildElement()->Value() << endl;

    if ((parms = simDoc->FirstChildElement()->FirstChildElement("ModelParams")) == NULL) {
        cerr << "Could not find <MoelParms> in simulation parameter file " << endl;
        return false;
    }

    // create neurons, synapses, connections, and layout objects specified in the description file
    IAllNeurons *neurons = NULL;
    IAllSynapses *synapses = NULL;
    Connections *conns = NULL;
    Layout *layout = NULL;
    const TiXmlNode* pNode = NULL;

    while ((pNode = parms->IterateChildren(pNode)) != NULL) {
        if (strcmp(pNode->Value(), "NeuronsParams") == 0) {
            neurons = FClassOfCategory::get()->createNeurons(pNode);
        } else if (strcmp(pNode->Value(), "SynapsesParams") == 0) {
            synapses = FClassOfCategory::get()->createSynapses(pNode);
        } else if (strcmp(pNode->Value(), "ConnectionsParams") == 0) {
            conns = FClassOfCategory::get()->createConnections(pNode);
        } else if (strcmp(pNode->Value(), "LayoutParams") == 0) {
            layout = FClassOfCategory::get()->createLayout(pNode);
        }
    }

    if (neurons == NULL){ cout << "N" << endl;}
    if (synapses == NULL){ cout << "S" << endl;}
    if (conns == NULL){ cout << "C" << endl;}
    if (layout == NULL){ cout << "L" << endl;}

    if (neurons == NULL || synapses == NULL || conns == NULL || layout == NULL) {
        cerr << "!ERROR: failed to create classes" << endl;
        return false;
    }

    // create the model
    #if defined(USE_GPU)
         simInfo->model = new GPUSpikingModel(conns, neurons, synapses, layout);
    #else
         simInfo->model = new SingleThreadedSpikingModel(conns, neurons, synapses, layout);
    #endif

    return true;
}

/*
 *  Load parameters from a file.
 *
 *  @param  simInfo   SimulationInfo class to read information from.
 *  @return true if successful, false if not
 */
bool LoadAllParameters(SimulationInfo *simInfo)
{
    DEBUG(cerr << "reading parameters from xml file" << endl;)

    TiXmlDocument simDoc(simInfo->stateInputFileName.c_str());
    if (!simDoc.LoadFile()) {
        cerr << "Failed loading simulation parameter file "
             << simInfo->stateInputFileName << ":" << "\n\t" << simDoc.ErrorDesc()
             << endl;
        cerr << " error: " << simDoc.ErrorRow() << ", " << simDoc.ErrorCol()
             << endl;
        return false;
    }

    // load simulation parameters
    if (simInfo->readParameters(&simDoc) != true) {
        return false;
    }

    // create instances of all model classes
    DEBUG(cerr << "creating instances of all classes" << endl;)
    if (createAllModelClassInstances(&simDoc, simInfo) != true) {
        return false;
    }

    // load parameters for all models
    if (FClassOfCategory::get()->readParameters(&simDoc) != true) {
        return false;
    }

    if (simInfo->stateOutputFileName.empty()) {
        cerr << "! ERROR: no stateOutputFileName is specified." << endl;
        return -1;
    }

    /*    verify that params were read correctly */
    DEBUG(printParams(simInfo);)

    return true;
}

/*
 *  Prints loaded parameters out to console.
 *
 *  @param  simInfo   SimulationInfo class to read information from.
 */
void printParams(SimulationInfo *simInfo) {
    cout << "\nPrinting simulation parameters...\n";
    simInfo->printParameters(cout);

    cout << "Model Parameters:" << endl;
    FClassOfCategory::get()->printParameters(cout);
    cout << "Done printing parameters" << endl;
}

/*
 *  Handles parsing of the command line
 *
 *  @param  argc      argument count.
 *  @param  argv      arguments.
 *  @param  simInfo   SimulationInfo class to read information from.
 *  @returns    true if successful, false otherwise.
 */
bool parseCommandLine(int argc, char* argv[], SimulationInfo *simInfo)
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
    simInfo->stateOutputFileName = cl["stateoutfile"];
    simInfo->stateInputFileName = cl["stateinfile"];
    simInfo->memInputFileName = cl["meminfile"];
    simInfo->memOutputFileName = cl["memoutfile"];
    simInfo->stimulusInputFileName = cl["stiminfile"];

#if defined(USE_GPU)
    if (EOF == sscanf(cl["deviceid"].c_str(), "%d", &g_deviceId)) {
        g_deviceId = 0;
    }
#endif  // USE_GPU

    return true;
}
