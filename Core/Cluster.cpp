#include "Cluster.h"

// Initialize the Barrier Synchnonize object for advanceThreads.
Barrier *Cluster::m_barrierAdvance = NULL;

// Initialize the flag for advanceThreads. true if terminating advanceThreads.
bool Cluster::m_isAdvanceExit = false;

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
 * Deserializes internal state from a prior run of the simulation.
 * This allows simulations to be continued from a particular point, to be restarted, or to be
 * started from a known state.
 *
 *  @param  input       istream to read from.
 *  @param  sim_info    used as a reference to set info for neurons and synapses.
 *  @param  clr_info    cluster informaion, used as a reference to set info for neurons and synapses.
 */
void Cluster::deserialize(istream& input, const SimulationInfo *sim_info, const ClusterInfo *clr_info)
{
    // read the neurons data & create neurons
    m_neurons->deserialize(input, clr_info);

    // read the synapse data & create synapses
    m_synapses->deserialize(input, *m_neurons, clr_info);
}

/*
 * Serializes internal state for the current simulation.
 * This allows simulations to be continued from a particular point, to be restarted, or to be
 * started from a known state.
 *
 *  @param  output      The filestream to write.
 *  @param  sim_info    used as a reference to set info for neurons and synapses.
 *  @param  clr_info    cluster informaion, used as a reference to set info for neurons and synapses.
 */
void Cluster::serialize(ostream& output, const SimulationInfo *sim_info, const ClusterInfo *clr_info)
{
    // write the neurons data
    m_neurons->serialize(output, clr_info);

    // write the synapse data
    m_synapses->serialize(output, clr_info);
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
 *  Thread for advance a cluster.
 *
 *  @param  sim_info    SimulationInfo class to read information from.
 *  @param  clr_info    ClusterInfo class to read information from.
 */
void Cluster::advanceThread(const SimulationInfo *sim_info, const ClusterInfo *clr_info)
{
    while (true) {
        // wait until the main thread notify that the advance is ready to go
        // or exit if quit is posted.
        m_barrierAdvance->Sync();

        // Check if Cluster::quitAdvanceThread() is called
        if (m_isAdvanceExit == true) {
            break;
        }

        // Advances neurons network state one simulation step
        advanceNeurons(sim_info, clr_info);

        // We don't need barrier synchronization between advanceNeurons
        // and advanceSynapses,
        // because all incoming synapses should be in the same cluster of
        // the target neuron. Therefore summation point of the neuron
        // should not be modified from synapses of other clusters.  

        // The above mention is not true. 
        // When advanceNeurons and advanceSynapses in different clusters 
        // are running concurrently, there might be race condition at
        // event queues. For example, EventQueue::addAnEvent() is called
        // from advanceNeurons in cluster 0 and EventQueue::checkAnEvent()
        // is called from advanceSynapses in cluster 1. These functions
        // contain memory read/write operation at event queue and 
        // consequntltly data race happens. 

        // wait until all threads are complete 
        m_barrierAdvance->Sync();

        // Advances synapses network state one simulation step
        advanceSynapses(sim_info, clr_info);

        // wait until all threads are complete 
        m_barrierAdvance->Sync();

        // Advance event queue state one simulation step
        advanceSpikeQueue();

        // wait until all threads are complete 
        m_barrierAdvance->Sync();
    }
}

/*
 *  Create an advanceThread.
 *  If barrier synchronize object has not been created, create it.
 *
 *  @param  sim_info    SimulationInfo class to read information from.
 *  @param  clr_info    ClusterInfo class to read information from.
 *  @param  count       Number of total clusters.
 */
void Cluster::createAdvanceThread(const SimulationInfo *sim_info, const ClusterInfo *clr_info, int count)
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
 *  Run advance of all waiting threads.
 */
void Cluster::runAdvance()
{
    // notify all advanceThread that the advanceNeurons is ready to go
    m_barrierAdvance->Sync();

    // notify all advanceThread that the advanceSynapses is ready to go
    m_barrierAdvance->Sync();

    // notify all advanceThread that the advanceSpikeQueue is ready to go
    m_barrierAdvance->Sync();

    // wait until the advance of all advanceThread complete
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
