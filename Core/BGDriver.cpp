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

#include "Model.h"
#include "FClassOfCategory.h"
#include "IRecorder.h"
#include "FSInput.h"
#include "Simulator.h"
#include <vector>

//! Cereal
#include <cereal/archives/xml.hpp>
#include <cereal/types/vector.hpp>
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
bool LoadAllParameters(SimulationInfo *simInfo, vector<Cluster *> &vtClr, vector<ClusterInfo *> &vtClrInfo);
void printParams(SimulationInfo *simInfo);
bool parseCommandLine(int argc, char* argv[], SimulationInfo *simInfo);
bool createAllModelClassInstances(TiXmlDocument* simDoc, SimulationInfo *simInfo, vector<Cluster *> &vtClr, vector<ClusterInfo *> &vtClrInfo);

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

    // create simulation info object
    simInfo = new SimulationInfo();

    // Handles parsing of the command line
    if (!parseCommandLine(argc, argv, simInfo)) {
        cerr << "! ERROR: failed during command line parse" << endl;
        return -1;
    }

    // Create all model instances and load parameters from a file.
    if (!LoadAllParameters(simInfo, vtClr, vtClrInfo)) {
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
        ifstream memory_in(simInfo->memInputFileName.c_str());
        cereal::XMLInputArchive archive(memory_in);

        // Prints out SynapsesProps before deserialization
        cout << "------------------------------Before Deserialization:--------------------------" << endl;
        for(int i = 0; i < vtClr.size(); i++) {
            dynamic_cast<AllSynapses *>(vtClr[i]->m_synapses)->m_pSynapsesProps->printSynapsesProps(); 
        }
        //dynamic_cast<ConnGrowth *>(dynamic_cast<Model *>(simInfo->model)->m_conns)->radii->printVector();

        // Deserializes synapse weight(s) along with each synapse's source neuron and destination neuron
        for(int i = 0; i < vtClr.size(); i++) {
            archive(*vtClr[i]);
        }

        // Creates synapse(s) from weight(s) 
        dynamic_cast<Model *>(simInfo->model)->m_conns->createSynapsesFromWeights(simInfo, dynamic_cast<Model *>(simInfo->model)->m_layout, vtClr, vtClrInfo);

        // Copy CPU Synapse data to GPU after deserialization
        simulator->copyCPUSynapseToGPU(simInfo);

        // Creates synapse index map
        SynapseIndexMap::createSynapseImap(simInfo, vtClr, vtClrInfo);

#if defined(USE_GPU)
        //print out weights on the GPU
        for(int i = 0; i < vtClr.size(); i++) {
            dynamic_cast<GPUSpikingCluster *> vtClr[i]->printGPUPropsCluster();
        }
#endif // USE_GPU


        // Deserializes radii
        //archive(*(dynamic_cast<ConnGrowth *>(dynamic_cast<Model *>(simInfo->model)->m_conns)));

        // Prints out SynapsesProps after deserialization
        cout << "------------------------------After Deserialization:--------------------------" << endl;
        for(int i = 0; i < vtClr.size(); i++) {
            dynamic_cast<AllSynapses *>(vtClr[i]->m_synapses)->m_pSynapsesProps->printSynapsesProps(); 
        }
        //dynamic_cast<ConnGrowth *>(dynamic_cast<Model *>(simInfo->model)->m_conns)->radii->printVector();
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
        ofstream memory_out (simInfo->memOutputFileName.c_str());
        cereal::XMLOutputArchive archive(memory_out);
        
        // Copy GPU Synapse data to CPU for serialization
        simulator->copyGPUSynapseToCPU(simInfo);
        
        // Serializes synapse weight(s) along with each synapse's source neuron and destination neuron
        for(int i = 0; i < vtClr.size(); i++) {
            archive(*vtClr[i]);
        }
        // Serializes radii
        //archive(*(dynamic_cast<ConnGrowth *>(dynamic_cast<Model *>(simInfo->model)->m_conns)));
    }

    //cout << "---------------------------After Serialization---------------------------------" <<endl;
    //for(int i = 0; i < vtClr.size(); i++) {
      //  dynamic_cast<AllSynapses *>(vtClr[i]->m_synapses)->m_pSynapsesProps->printSynapsesProps(); 
    //}

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
bool createAllModelClassInstances(TiXmlDocument* simDoc, SimulationInfo *simInfo, vector<Cluster *> &vtClr, vector<ClusterInfo *> &vtClrInfo)
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

    // load parameters for all models
    if (FClassOfCategory::get()->readParameters(simDoc) != true) {
        return false;
    }

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
            IAllNeurons *neurons_1 = FClassOfCategory::get()->createNeurons();

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

    return true;
}

/*
 *  Load parameters from a file.
 *
 *  @param  simInfo       SimulationInfo class to read information from.
 *  @param  cluster       Cluster class object to be created.
 *  @param  clusterInfo   ClusterInfo class to be ceated.
 *  @return true if successful, false if not
 */
bool LoadAllParameters(SimulationInfo *simInfo, vector<Cluster *> &vtClr, vector<ClusterInfo *> &vtClrInfo)
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
        return false; }

    // create instances of all model classes & load parameters
    DEBUG(cerr << "creating instances of all classes" << endl;)
    if (createAllModelClassInstances(&simDoc, simInfo, vtClr, vtClrInfo) != true) {
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
