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
#include "ParameterManager.h"
#include "Model.h"
#include "FNeurons.h"
#include "IRecorder.h"
#include "FSInput.h"
#include "Simulator.h"
#include <vector>

//! Cereal
#include <cereal/archives/xml.hpp>
#include <cereal/archives/binary.hpp>
#include "ConnGrowth.h"

// Uncomment to use visual leak detector (Visual Studios Plugin)
// #include <vld.h>

#if defined(USE_GPU)
    #include "GPUSpikingCluster.h"
#else 
    #include "SingleThreadedCluster.h"
#endif

using namespace std;

// functions
void printParams(SimulationInfo *simInfo);
bool parseCommandLine(int argc, char* argv[], SimulationInfo *simInfo);
bool createAllModelClassInstances(ParameterManager* parameterManager, SimulationInfo *simInfo, vector<Cluster *> &vtClr, vector<ClusterInfo *> &vtClrInfo);
void printKeyStateInfo(SimulationInfo *simInfo, vector<Cluster *> &vtClr);
void serializeSynapseInfo(SimulationInfo *simInfo, Simulator *simulator, vector<Cluster *> &vtClr);
bool deserializeSynapseInfo(SimulationInfo *simInfo, Simulator *simulator, vector<Cluster *> &vtClr, vector<ClusterInfo *> &vtClrInfo);

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

    vector<ClusterInfo *> vtClrInfo;   // Vector of Cluster information
    vector<Cluster *> vtClr;           // Vector of Cluster object
    ParameterManager* parameterManager = NULL;  // XML param reader

    // create simulation info object
    simInfo = new SimulationInfo();

    // Handles parsing of the command line
    if (!parseCommandLine(argc, argv, simInfo)) {
        cerr << "! ERROR: failed during command line parse" << endl;
        return -1;
    }

    // create XML parameter reader
    parameterManager = new ParameterManager();
    // load XML parameter file into parameter reader
    if (!parameterManager->loadParameterFile(simInfo->stateInputFileName)) {
        cerr << "! ERROR: failed loading XML parameter file" << endl;
        return -1;
    }

    // initialize global simulation parameters
    if (!simInfo->readParameters(parameterManager)) {
        cerr << "! ERROR: failed loading global simulation parameters" << endl;
        return -1;
    } else {
        cout << "Printing global simulation parameters..." << endl;
        simInfo->printParameters(cout);
    }
    
    // create instances of all model classes & load parameters
    DEBUG(cerr << "creating instances of all classes" << endl;)
    if (createAllModelClassInstances(parameterManager, simInfo, vtClr, vtClrInfo) != true) {
        return -1;
    }

    /*
    if (simInfo->stateOutputFileName.empty()) {
        cerr << "! ERROR: no stateOutputFileName is specified." << endl;
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
      
        DEBUG(
        // Prints out internal state information before deserialization
        cout << "------------------------------Before Deserialization:------------------------------" << endl;
        printKeyStateInfo(simInfo,vtClr);
        )

        // Deserialization
        if(!deserializeSynapseInfo(simInfo, simulator, vtClr, vtClrInfo)) {
            cerr << "! ERROR: failed while deserializing objects" << endl;
            return -1;
        }
    
        DEBUG(
        // Prints out internal state information after deserialization
        cout << "------------------------------After Deserialization:------------------------------" << endl;
        printKeyStateInfo(simInfo,vtClr);
        )

    }

    // Run simulation
    simulator->simulate(simInfo);

    // Terminate the stimulus input 
    if (simInfo->pInput != NULL)
    {
        simInfo->pInput->term(simInfo, vtClrInfo);
        delete simInfo->pInput;
    }

    // Writes simulation results to an output destination
    simulator->saveData(simInfo);

    // Serializes internal state for the current simulation
    if (!simInfo->memOutputFileName.empty()) {
        
        // Serialization
        serializeSynapseInfo(simInfo, simulator, vtClr);

        DEBUG(
        // Prints out internal state information after serialization
        cout << "------------------------------After Serialization:------------------------------" << endl;
        printKeyStateInfo(simInfo,vtClr);
        )
    }

    // Tell simulation to clean-up and run any post-simulation logic.
    simulator->finish(simInfo);

    // terminates the simulation recorder
    if (simInfo->simRecorder != NULL) {
        simInfo->simRecorder->term();
    }

    time(&end_time);
    double time_elapsed = difftime(end_time, start_time);
    double ssps = simInfo->epochDuration * simInfo->maxSteps / time_elapsed;
    cout << "time simulated: " << simInfo->epochDuration * simInfo->maxSteps << endl;
    cout << "time elapsed: " << time_elapsed << endl;
    cout << "ssps (simulation seconds / real time seconds): " << ssps << endl;
    
    // delete clusters
    for (vector<Cluster *>::iterator clr = vtClr.begin(); clr !=vtClr.end(); clr++) {
        delete *clr;
    }

    // delete cluster information
    for (vector<ClusterInfo *>::iterator clrInfo = vtClrInfo.begin(); clrInfo !=vtClrInfo.end(); clrInfo++) {
        delete *clrInfo;
    }
    */
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
 *  @param  simDoc        the TiXmlDocument to read from.
 *  @param  simInfo       SimulationInfo class to read information from.
 *  @param  cluster       Cluster class object to be created.
 *  @param  clusterInfo   ClusterInfo class to be ceated.
 *  @retrun true if successful, false if not
 */
bool createAllModelClassInstances(ParameterManager* pm, SimulationInfo *simInfo, 
        vector<Cluster *> &vtClr, vector<ClusterInfo *> &vtClrInfo)
{
    // create neurons, synapses, connections, and layout objects specified in the description file
    IAllNeurons *neurons = NULL;
    IAllSynapses *synapses = NULL;
    Connections *conns = NULL;
    Layout *layout = NULL;

    string neuronClassName;
    string synapseClassName;
    string connectionClassName;
    string layoutClassName;

    if (parameterManager->getStringByXpath("//NeuronsParams/@class", neuronClassName) {
        neurons = FNeurons::get()->createNeurons(neuronClassName);
    }
    if (parameterManager->getStringByXpath("//SynapsesParams/@class", synapseClassName) {
        synapses = FSynapses::get()->createSynapses(synapseClassName);
    }
    if (parameterManager->getStringByXpath("//ConnectionsParams/@class", connectionClassName) {
        conns = FConnections::get()->createConnections(connectionClassName);
    }
    if (parameterManager->getStringByXpath("//LayoutParams/@class", layoutClassName) {
        layout = FLayout::get()->createLayout(layoutClassName);
    }

    // TODO: refactor this so the error message shows which class(es) failed
    if (neurons == NULL || synapses == NULL || conns == NULL || layout == NULL) {
        cerr << "!ERROR: failed to create classes" << endl;
        return false;
    }

    // load parameters for all simulator components
    if (!neurons->readParameters(parameterManager) ||
        !synapses->readParameters(parameterManager) ||
        !conns->readParameters(parameterManager) ||
        !layout->readParameters(parameterManager)) {
        return false;
    }

    /* Cannot test until factory work is completed
     * 
    // create clusters
    int numClusterNeurons = simInfo->totalNeurons / g_numClusters;	// number of neurons in cluster

    for (int iCluster = 0; iCluster < g_numClusters; iCluster++) {
        // create a cluster information
        ClusterInfo *clusterInfo = new ClusterInfo();
        clusterInfo->clusterID = iCluster;
        clusterInfo->clusterNeuronsBegin = numClusterNeurons * iCluster;
        if (iCluster == g_numClusters - 1) {
            clusterInfo->totalClusterNeurons = simInfo->totalNeurons - numClusterNeurons * (g_numClusters - 1);
        } else {
            clusterInfo->totalClusterNeurons = numClusterNeurons;
        }
        clusterInfo->seed = simInfo->seed + iCluster;
#if defined(USE_GPU)
        clusterInfo->deviceId = g_deviceId + iCluster;
#endif // USE_GPU

        // save the cluser information to the vector
        vtClrInfo.push_back(clusterInfo); 

        // create a cluster
        Cluster *cluster;
        if (iCluster == 0) {
#if defined(USE_GPU)
            cluster = new GPUSpikingCluster(neurons, synapses);
#else
            cluster = new SingleThreadedCluster(neurons, synapses);
#endif
        } else {
            // create a new neurons class object and copy properties from the reference neurons class object
            IAllNeurons *neurons_1 = FNeurons::get()->createNeurons();

            // create a new synapses class object and copy properties from the reference synapses class object
            IAllSynapses *synapses_1 = FClassOfCategory::get()->createSynapses();

           // create a cluster class object
#if defined(USE_GPU)
            cluster = new GPUSpikingCluster(neurons_1, synapses_1);
#else
            cluster = new SingleThreadedCluster(neurons_1, synapses_1);
#endif
        }

        // save the cluster to the vector
        vtClr.push_back(cluster);
    }

    // create the model
    simInfo->model = new Model(conns, layout, vtClr, vtClrInfo);
    */
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

    /*
    cout << "Model Parameters:" << endl;
    FClassOfCategory::get()->printParameters(cout);
    cout << "Done printing parameters" << endl;
    */
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
            || (cl.addParam("stateinfile", 't', ParamContainer::filename | ParamContainer::required, "simulation parameter filename") != ParamContainer::errOk)
            || (cl.addParam("deviceid", 'd', ParamContainer::regular, "CUDA device id") != ParamContainer::errOk)
            || (cl.addParam("numclusters", 'c', ParamContainer::regular, "number of clusters") != ParamContainer::errOk)
            || (cl.addParam( "stiminfile", 's', ParamContainer::filename, "stimulus input file" ) != ParamContainer::errOk)
            || (cl.addParam("meminfile", 'r', ParamContainer::filename, "simulation memory image input filename") != ParamContainer::errOk)
            || (cl.addParam("memoutfile", 'w', ParamContainer::filename, "simulation memory image output filename") != ParamContainer::errOk)) {
        cerr << "Internal error creating command line parser" << endl;
        return false;
    }
#else    // !USE_GPU
    if ((cl.addParam("stateoutfile", 'o', ParamContainer::filename, "simulation state output filename") != ParamContainer::errOk)
            || (cl.addParam("stateinfile", 't', ParamContainer::filename | ParamContainer::required, "simulation parameter filename") != ParamContainer::errOk)
            || (cl.addParam("numclusters", 'c', ParamContainer::regular, "number of clusters") != ParamContainer::errOk)
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

    // Number of clusters
    if (EOF == sscanf(cl["numclusters"].c_str(), "%d", &g_numClusters)) {
        g_numClusters = 1;
    }

    simInfo->numClusters = g_numClusters;

#if defined(USE_GPU)
    if (EOF == sscanf(cl["deviceid"].c_str(), "%d", &g_deviceId)) {
        g_deviceId = 0;
    }
#endif  // USE_GPU

    return true;
}

/*
 *  Prints key internal state information 
 *  (Used for serialization/deserialization verification)
 *
 *  @param  simInfo   SimulationInfo class to read information from.
 *  @param  cluster   Cluster class object to be created.
 */
void printKeyStateInfo(SimulationInfo *simInfo, vector<Cluster *> &vtClr)
{        
#if defined(USE_GPU)
    // Prints out SynapsesProps on the GPU
    for(int i = 0; i < vtClr.size(); i++) {
        dynamic_cast<GPUSpikingCluster *> (vtClr[i])->printGPUSynapsesPropsCluster();
    }
    // Prints out radii on the GPU (only if it is a connGrowth model)
    if(dynamic_cast<ConnGrowth *>(dynamic_cast<Model *>(simInfo->model)->m_conns) != nullptr) {
        dynamic_cast<ConnGrowth *>(dynamic_cast<Model *>(simInfo->model)->m_conns)->printRadii();
    }
#else
    // Prints out SynapsesProps on the CPU
    for(int i = 0; i < vtClr.size(); i++) {
        dynamic_cast<AllSynapses *>(vtClr[i]->m_synapses)->m_pSynapsesProps->printSynapsesProps(); 
    }
    // Prints out radii on the CPU (only if it is a connGrowth model)
    if(dynamic_cast<ConnGrowth *>(dynamic_cast<Model *>(simInfo->model)->m_conns) != nullptr) {
        dynamic_cast<ConnGrowth *>(dynamic_cast<Model *>(simInfo->model)->m_conns)->printRadii();
    }
#endif       
}

/*
 *  Serializes synapse weights, source neurons, destination neurons, 
 *  maxSynapsesPerNeuron, totalClusterNeurons, and 
 *  if running a connGrowth model, serializes radii as well 
 *
 *  @param  simInfo   SimulationInfo class to read information from.
 *  @param  simulator Simulator class to perform actions.
 *  @param  cluster   Cluster class object to be created.
 */
void serializeSynapseInfo(SimulationInfo *simInfo, Simulator *simulator, vector<Cluster *> &vtClr)
{
    // We can serialize to a variety of archive file formats. Below, comment out
    // all but the two lines that correspond to the desired format.
    ofstream memory_out (simInfo->memOutputFileName.c_str());
    cereal::XMLOutputArchive archive(memory_out);
    //ofstream memory_out (simInfo->memOutputFileName.c_str(), std::ios::binary);
    //cereal::BinaryOutputArchive archive(memory_out);

#if defined(USE_GPU)        
    // Copies GPU Synapse props data to CPU for serialization
    simulator->copyGPUSynapseToCPU(simInfo);
#endif // USE_GPU

    // Serializes synapse weights along with each synapse's source neuron and destination neuron
    for(int i = 0; i < vtClr.size(); i++) {
        archive(*vtClr[i]);
    }
    // Serializes radii (only if it is a connGrowth model)
    if(dynamic_cast<ConnGrowth *>(dynamic_cast<Model *>(simInfo->model)->m_conns) != nullptr) {
        archive(*(dynamic_cast<ConnGrowth *>(dynamic_cast<Model *>(simInfo->model)->m_conns)));
    }

}

/*
 *  Deserializes synapse weights, source neurons, destination neurons, 
 *  maxSynapsesPerNeuron, totalClusterNeurons, and 
 *  if running a connGrowth model and radii is in serialization file, deserializes radii as well
 *
 *  @param  simInfo   SimulationInfo class to read information from.
 *  @param  simulator Simulator class to perform actions.
 *  @param  cluster   Cluster class object to be created.
 *  @param  clusterInfo   ClusterInfo class to be ceated.
 *  @returns    true if successful, false otherwise.
 */
bool deserializeSynapseInfo(SimulationInfo *simInfo, Simulator *simulator, vector<Cluster *> &vtClr, vector<ClusterInfo *> &vtClrInfo)
{
   // We can deserialize from a variety of archive file formats. Below, comment
   // out all but the line that is compatible with the desired format.
    ifstream memory_in(simInfo->memInputFileName.c_str());
    //ifstream memory_in (simInfo->memInputFileName.c_str(), std::ios::binary);
    
    // Checks to see if serialization file exists
    if(!memory_in) {
        cerr << "The serialization file doesn't exist" << endl;
        return false;
    }

   // We can deserialize from a variety of archive file formats. Below, comment
   // out all but the line that corresponds to the desired format.
    cereal::XMLInputArchive archive(memory_in);
    //cereal::BinaryInputArchive archive(memory_in);

    // Deserializes synapse weights along with each synapse's source neuron and destination neuron
    for(int i = 0; i < vtClr.size(); i++) {
        // Uses "try catch" to catch any cereal exception
        try {
            archive(*vtClr[i]);
        }
        catch(cereal::Exception e) {
            cerr << "Failed deserializing synapse weights, source neurons, and/or destination neurons." << endl;
            return false;
        }
    }

    // Creates synapses from weights 
    dynamic_cast<Model *>(simInfo->model)->m_conns->createSynapsesFromWeights(simInfo, dynamic_cast<Model *>(simInfo->model)->m_layout, vtClr, vtClrInfo);

#if defined(USE_GPU)
    // Copies CPU Synapse data to GPU after deserialization, if we're doing
    // a GPU-based simulation.
    simulator->copyCPUSynapseToGPU(simInfo);
#endif // USE_GPU

    // Creates synapse index map (includes copy CPU index map to GPU)
    SynapseIndexMap::createSynapseImap(simInfo, vtClr, vtClrInfo);

    // Deserializes radii (only when running a connGrowth model and radii is in serialization file)
    if( dynamic_cast<ConnGrowth *>(dynamic_cast<Model *>(simInfo->model)->m_conns) != nullptr) {
        // Uses "try catch" to catch any cereal exception
        try {
            archive(*(dynamic_cast<ConnGrowth *>(dynamic_cast<Model *>(simInfo->model)->m_conns)));
        }
        catch(cereal::Exception e) {
            cerr << "Failed deserializing radii." << endl;
            return false;
        }
    }

    return true;
    
}
