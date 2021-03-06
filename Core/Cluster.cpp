#include "Cluster.h"
#include "ISInput.h"

// Initialize the Barrier Synchnonize object for advanceThreads.
Barrier *Cluster::m_barrierAdvance = NULL;

// Initialize the flag for advanceThreads. true if terminating advanceThreads.
bool Cluster::m_isAdvanceExit = false;

// Initialize the synaptic transmission delay, descretized into time steps.
int Cluster::m_nSynapticTransDelay = 0;

/*
 *  Constructor
 */
Cluster::Cluster(IAllNeurons *neurons, IAllSynapses *synapses) :
    m_neurons(neurons),
    m_synapses(synapses),
    m_synapseIndexMap(NULL)
{
}

/*
 * Destructor
 */
Cluster::~Cluster()
{
    if (m_neurons != NULL) {
        delete m_neurons;
        m_neurons = NULL;
    }

    if (m_synapses != NULL) {
        delete m_synapses;
        m_synapses = NULL;
    }

    if (m_synapseIndexMap != NULL) {
        delete m_synapseIndexMap;
        m_synapseIndexMap = NULL;
    }
}

/*
 *  Creates all the Neurons and generates data for them.
 *
 *  @param  sim_info    SimulationInfo class to read information from.
 *  @param  layout      A class to define neurons' layout information in the network.
 *  @param  clr_info    ClusterInfo class to read information from.
 */
void Cluster::setupCluster(SimulationInfo *sim_info, Layout *layout, ClusterInfo *clr_info)
{
    DEBUG(cerr << "\nAllocating neurons..." << endl;)
    DEBUG(cerr << "\tSetting up neurons....";)
    m_neurons->setupNeurons(sim_info, clr_info);

    DEBUG(cerr << "done.\n\tSetting up synapses....";)
    m_synapses->setupSynapses(sim_info, clr_info);

    DEBUG(cerr << "done." << endl;)

    // Creates all the Neurons and generates data for them.
    m_neurons->createAllNeurons(sim_info, layout, clr_info);

    DEBUG(cerr << "Done initializing neurons..." << endl;)
}

/*
 *  Clean up the cluster.
 *
 *  @param  sim_info    SimulationInfo to refer.
 *  @param  clr_info    ClusterInfo to refer.
 */
void Cluster::cleanupCluster(SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    m_neurons->cleanupNeurons();
    m_synapses->cleanupSynapses();
}

/*
 *  Create an advanceThread.
 *  If barrier synchronize object has not been created, create it.
 *
 *  @param  sim_info    SimulationInfo class to read information from.
 *  @param  clr_info    ClusterInfo class to read information from.
 *  @param  count       Number of total clusters.
 */
void Cluster::createAdvanceThread(const SimulationInfo *sim_info, ClusterInfo *clr_info, int count)
{
    // If barrier synchronize object has not been created, create it
    if (m_barrierAdvance == NULL) {
        // number of cluster thread + 1 (for main thread)
        m_barrierAdvance = new Barrier(count + 1);
    }

    // Create an advanceThread
    std::thread thAdvance(&Cluster::advanceThread, this, sim_info, clr_info);

    // Leave it running
    thAdvance.detach();
}

/*
 *  Thread for advance a cluster.
 *
 *  @param  sim_info    SimulationInfo class to read information from.
 *  @param  clr_info    ClusterInfo class to read information from.
 */
void Cluster::advanceThread(const SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    while (true) {
        // wait until the main thread notify that the advance is ready to go
        // or exit if quit is posted.
        m_barrierAdvance->Sync();

        // Check if Cluster::quitAdvanceThread() is called
        if (m_isAdvanceExit == true) {
            break;
        }

#if defined(VALIDATION)
        // Generate random numbers
        genRandNumbers(sim_info, clr_info);

        // wait until all threads are complete
        m_barrierAdvance->Sync();
#endif // VALIDATION

        // Advance neurons and synapses indepedently (without barrier synchronization)
        // within synaptic transmission delay period. 
        for (int iStepOffset = 0; iStepOffset < m_nSynapticTransDelay; iStepOffset++) {
            if (sim_info->pInput != NULL) {
                // input stimulus
                sim_info->pInput->inputStimulus(sim_info, clr_info, iStepOffset);
            }

            // Advances neurons network state one simulation step
            advanceNeurons(sim_info, clr_info, iStepOffset);

            // When advanceNeurons and advanceSynapses in different clusters 
            // are running concurrently, there might be race condition at
            // event queues. For example, EventQueue::addAnEvent() is called
            // from advanceNeurons in cluster 0 and EventQueue::checkAnEvent()
            // is called from advanceSynapses in cluster 1. These functions
            // contain memory read/write operation at event queue and 
            // consequntltly data race happens. (host version)

            // Now we could eliminate all barrier synchronization within
            // synaptic transmission delay period, because
            // EventQueue::addAnEvent() and EventQueue::checkAnEvent() handle
            // atomic read/write operations.

            // Advances synapses network state one simulation step
            advanceSynapses(sim_info, clr_info, iStepOffset);
        } // end synaptic transmission delay loop

        // wait until all threads are complete the synaptic transmission delay loop
        m_barrierAdvance->Sync();

        // Process outgoing spiking data between clusters
        if (sim_info->numClusters >= 2) {
            processInterClustesOutgoingSpikes(clr_info);
        }

        // wait until all threads are complete
        m_barrierAdvance->Sync();

        // Process incoming spiking data between clusters
        if (sim_info->numClusters >= 2) {
            processInterClustesIncomingSpikes(clr_info);
        }

        // wait until all threads are complete
        m_barrierAdvance->Sync();

        // Advance event queue state m_nSynapticTransDelay simulation steps
        advanceSpikeQueue(sim_info, clr_info, m_nSynapticTransDelay);

        // wait until all threads are complete 
        m_barrierAdvance->Sync();
    }
}

/*
 *  Run advance of all waiting threads.
 *
 *  @param  sim_info    SimulationInfo class to read information from.
 *  @param  iStep       Simulation steps to advance.
 */
void Cluster::runAdvance(const SimulationInfo *sim_info, int iStep)
{
    // set the synaptic transmission delay
    m_nSynapticTransDelay = iStep;

#if defined(VALIDATION)
    // generate random numbers
    m_barrierAdvance->Sync();
#endif // VALIDATION

    // start advanceThread
    m_barrierAdvance->Sync();

    // wait until the advance of all advanceThread complete the synaptic transmission delay loop
    m_barrierAdvance->Sync();

    // wait until the process outgoing spiking data between clusters complete
    m_barrierAdvance->Sync();

    // wait until the process incoming spiking data between clusters complete
    m_barrierAdvance->Sync();

    // wait until the advance of event queue state
    m_barrierAdvance->Sync();
}

/*
 *  Quit all advanceThread.
 */
void Cluster::quitAdvanceThread()
{
    // notify all advanceThread to quit
    m_isAdvanceExit = true;
    m_barrierAdvance->Sync();

    // delete barrier synchronize object
    delete m_barrierAdvance;
    m_barrierAdvance = NULL;
}
