#include "Model.h"

#include "tinyxml.h"

#include "ParseParamError.h"
#include "Util.h"
#include "ConnGrowth.h"
#include "ISInput.h"
#if defined(USE_GPU)
#include "GPUSpikingCluster.h"
#endif

/*
 *  Constructor
 */
Model::Model(Connections *conns, Layout *layout, vector<Cluster *> &vtClr, vector<ClusterInfo *> &vtClrInfo) :
    m_read_params(0),
    m_conns(conns),
    m_layout(layout),
    m_vtClrInfo(vtClrInfo),
    m_vtClr(vtClr)
{
}

/*
 *  Destructor
 */
Model::~Model()
{
    if (m_conns != NULL) {
        delete m_conns;
        m_conns = NULL;
    }

    if (m_layout != NULL) {
        delete m_layout;
        m_layout = NULL;
    }
}

/*
 * Deserializes internal state from a prior run of the simulation.
 * This allows simulations to be continued from a particular point, to be restarted, or to be
 * started from a known state.
 *
 *  @param  input       istream to read from.
 *  @param  sim_info    used as a reference to set info for neurons and synapses.
 */
void Model::deserialize(istream& input, const SimulationInfo *sim_info)
{
    // read the clusters data
    m_vtClr[0]->deserialize(input, sim_info, m_vtClrInfo[0]);

    // create a synapse index map
    SynapseIndexMap::createSynapseImap(sim_info, m_vtClr, m_vtClrInfo);

    // read the connections data
    m_conns->deserialize(input, sim_info);
}

/*
 * Serializes internal state for the current simulation.
 * This allows simulations to be continued from a particular point, to be restarted, or to be
 * started from a known state.
 *
 *  @param  output      The filestream to write.
 *  @param  sim_info    used as a reference to set info for neurons and synapses.
 */
void Model::serialize(ostream& output, const SimulationInfo *sim_info)
{
    // write the neurons data
    output << sim_info->totalNeurons << ends;

    // write clusters data
    m_vtClr[0]->serialize(output, sim_info, m_vtClrInfo[0]);

    // write the connections data
    m_conns->serialize(output, sim_info);

    output << flush;
}

/*
 *  Save simulation results to an output destination.
 *
 *  @param  sim_info    parameters for the simulation. 
 */
void Model::saveData(SimulationInfo *sim_info)
{
    if (sim_info->simRecorder != NULL) {
        sim_info->simRecorder->saveSimData(m_vtClr, m_vtClrInfo);
    }
}

/*
 *  Creates all the Neurons and generates data for them.
 *
 *  @param  sim_info    SimulationInfo class to read information from.
 */
void Model::setupClusters(SimulationInfo *sim_info)
{
    // init neuron's map with layout
    m_layout->setupLayout(sim_info);
    m_layout->generateNeuronTypeMap(sim_info->totalNeurons);
    m_layout->initStarterMap(sim_info->totalNeurons);

    // create & initialize InterClustersEventHandler
    m_eventHandler = new InterClustersEventHandler();
    m_eventHandler->initEventHandler(m_vtClr.size());

    // setup each cluster
    for (unsigned int i = 0; i < m_vtClr.size(); i++) {
        m_vtClrInfo[i]->eventHandler = m_eventHandler;

        // creates all the Neurons and generates data for them in the cluster
        m_vtClr[i]->setupCluster(sim_info, m_layout, m_vtClrInfo[i]);

        // create advance threads
        m_vtClr[i]->createAdvanceThread(sim_info, m_vtClrInfo[i], m_vtClrInfo.size());
    }

    // set up the connection of all the Neurons and Synapses of the simulation
    m_conns->setupConnections(sim_info, m_layout, m_vtClr, m_vtClrInfo);
}

/*
 *  Sets up the Simulation.
 *
 *  @param  sim_info    SimulationInfo class to read information from.
 */
void Model::setupSim(SimulationInfo *sim_info)
{
    // Init radii and rates history matrices with default values
    if (sim_info->simRecorder != NULL) {
        sim_info->simRecorder->initDefaultValues();
    }

    // Creates all the Neurons and generates data for them.
    setupClusters(sim_info);

    // init stimulus input object
    if (sim_info->pInput != NULL) {
        cout << "Initializing input." << endl;
        sim_info->pInput->init(sim_info, m_vtClrInfo);
    }
}

/*
 *  Clean up the simulation.
 *
 *  @param  sim_info    SimulationInfo to refer.
 */
void Model::cleanupSim(SimulationInfo *sim_info)
{
    Cluster::quitAdvanceThread();

    for (unsigned int i = 0; i < m_vtClr.size(); i++) {
        m_vtClr[i]->cleanupCluster(sim_info, m_vtClrInfo[i]);
    }

    m_conns->cleanupConnections();

    delete m_eventHandler;
}

/*
 *  Log this simulation step.
 *
 *  @param  sim_info    SimulationInfo to reference.
 */
void Model::logSimStep(const SimulationInfo *sim_info) const
{
    ConnGrowth* pConnGrowth = dynamic_cast<ConnGrowth*>(m_conns);
    if (pConnGrowth == NULL)
        return;

    cout << "format:\ntype,radius,firing rate" << endl;

    for (int y = 0; y < sim_info->height; y++) {
        stringstream ss;
        ss << fixed;
        ss.precision(1);

        for (int x = 0; x < sim_info->width; x++) {
            switch (m_layout->neuron_type_map[x + y * sim_info->width]) {
            case EXC:
                if (m_layout->starter_map[x + y * sim_info->width])
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

            ss << " " << (*pConnGrowth->radii)[x + y * sim_info->width];

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

/*
 *  Update the simulation history of every epoch.
 *
 *  @param  sim_info    SimulationInfo to refer from.
 */
void Model::updateHistory(const SimulationInfo *sim_info)
{
    // Compile history information in every epoch
    if (sim_info->simRecorder != NULL) {
        sim_info->simRecorder->compileHistories(m_vtClr, m_vtClrInfo);
    }
}

/*
 *  Get the Connections class object.
 *
 *  @return Pointer to the Connections class object.
 */
Connections* Model::getConnections()
{
    return m_conns;
}

/*
 *  Get the Layout class object.
 *
 *  @return Pointer to the Layout class object.
 */
Layout* Model::getLayout()
{
    return m_layout;
}

/*
 *  Advance everything in the model one time step. In this case, that
 *  means advancing just the Neurons and Synapses.
 *
 *  @param  sim_info    SimulationInfo class to read information from.
 *  @param  iStep       Simulation steps to advance.
 */
void Model::advance(const SimulationInfo *sim_info, int iStep)
{
    // run advance of all waiting threads
    Cluster::runAdvance(sim_info, iStep);
}

/*
 *  Update the connection of all the Neurons and Synapses of the simulation.
 *
 *  @param  sim_info    SimulationInfo class to read information from.
 */
void Model::updateConnections(const SimulationInfo *sim_info)
{
#if defined(USE_GPU)
    // for each cluster
    for (CLUSTER_INDEX_TYPE iCluster = 0; iCluster < m_vtClr.size(); iCluster++) {
        // copy neuron's data from device memory to host
        AllSpikingNeurons *neurons = dynamic_cast<AllSpikingNeurons*>(m_vtClr[iCluster]->m_neurons);
        GPUSpikingCluster *GPUClr = dynamic_cast<GPUSpikingCluster *>(m_vtClr[iCluster]);
        neurons->copyNeuronDeviceSpikeCountsToHost(GPUClr->m_allNeuronsDevice, m_vtClrInfo[iCluster]);
        neurons->copyNeuronDeviceSpikeHistoryToHost(GPUClr->m_allNeuronsDevice, sim_info, m_vtClrInfo[iCluster]);
    }
#endif // USE_GPU

    // Update Connections data
    if (m_conns->updateConnections(sim_info, m_layout, m_vtClr, m_vtClrInfo)) {
        m_conns->updateSynapsesWeights(sim_info, m_layout, m_vtClr, m_vtClrInfo);
    }
}

/*
 *  Print cluster physical CPU core assignment data.
 */
void Model::printThreadCoreData(){
    // get the number of physical cores on a machine
    unsigned int nCores = std::thread::hardware_concurrency();

    cout << endl;
    for (unsigned i = 0; i < m_vtClr.size(); i++) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);

        pthread_getaffinity_np(m_vtClrInfo[i]->thread_native, sizeof(cpu_set_t), &cpuset);

        for(unsigned int j = 0; j < nCores; j++) {
            if (CPU_ISSET(j, &cpuset)) {
                cout << "Cluster " << i << " is running on core " << j << endl;
            }
        }
    }
    cout << endl;
}

#if defined(PERFORMANCE_METRICS)

/*
 *  Print performance metrics statistics
 *
 *  @param  total_time    Total time since simulation start.
 *  @param  steps         Number of epochs.
 */
void Model::printPerformanceMetrics(double total_time, int steps)
{
    cout << "Total time since simulation start:" << endl;
#ifdef USE_GPU
    // Print total time (in seconds) of each procedure since simulation start
    // for each cluster
    for (CLUSTER_INDEX_TYPE iCluster = 0; iCluster < m_vtClrInfo.size(); iCluster++) {
        cout << "\nCluster " << iCluster << endl;
        cout << "GPU random number generation: " << m_vtClrInfo[iCluster]->t_gpu_rndGeneration << " seconds ("
           << m_vtClrInfo[iCluster]->t_gpu_rndGeneration / total_time * 100 << "%)" << endl;
        cout << "GPU advanceNeurons: " << m_vtClrInfo[iCluster]->t_gpu_advanceNeurons << " seconds ("
           << m_vtClrInfo[iCluster]->t_gpu_advanceNeurons / total_time * 100 << "%)" << endl;
        cout << "GPU advanceSynapses: " << m_vtClrInfo[iCluster]->t_gpu_advanceSynapses << " seconds ("
           << m_vtClrInfo[iCluster]->t_gpu_advanceSynapses / total_time * 100 << "%)" << endl;
        cout << "GPU calcSummation: " << m_vtClrInfo[iCluster]->t_gpu_calcSummation << " seconds ("
           << m_vtClrInfo[iCluster]->t_gpu_calcSummation / total_time * 100 << "%)" << endl;
    }
#endif
    cout << "\nHost adjustSynapses: " << t_host_adjustSynapses << " seconds ("
       << t_host_adjustSynapses / total_time * 100 << "%)" << endl;

    cout << "\nAverage time per simulation epoch:" << endl;
#ifdef USE_GPU
    // Print average time per per simulation epoch  time (in seconds) of each procedure since simulation start
    // for each cluster
    for (CLUSTER_INDEX_TYPE iCluster = 0; iCluster < m_vtClrInfo.size(); iCluster++) {
        cout << "\nCluster " << iCluster << endl;
        cout << "GPU random number generation: " << m_vtClrInfo[iCluster]->t_gpu_rndGeneration/steps
           << " seconds/epoch" << endl;
        cout << "GPU advanceNeurons: " << m_vtClrInfo[iCluster]->t_gpu_advanceNeurons/steps
           << " seconds/epoch" << endl;
        cout << "GPU advanceSynapses: " << m_vtClrInfo[iCluster]->t_gpu_advanceSynapses/steps
           << " seconds/epoch" << endl;
        cout << "GPU calcSummation: " << m_vtClrInfo[iCluster]->t_gpu_calcSummation/steps
       << " seconds/epoch" << endl;
    }
#endif
    cout << "Host adjustSynapses: " << t_host_adjustSynapses/steps
       << " seconds/epoch" << endl;
}

#endif // PERFORMANCE_METRICS
