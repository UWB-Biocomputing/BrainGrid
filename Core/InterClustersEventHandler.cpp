#include "InterClustersEventHandler.h"
#include "EventQueue.h"

InterClustersEventHandler::InterClustersEventHandler() : m_vtEventQueue(NULL)
{
}

InterClustersEventHandler::~InterClustersEventHandler()
{
    if (m_vtEventQueue != NULL) {
        delete m_vtEventQueue;
    }
}

/*
 * Initializes the event handler.
 *
 * @param size  Size of the EventQueue vector.
 */
void InterClustersEventHandler::initEventHandler(const int size)
{
    m_vtEventQueue = new vector<EventQueue *>(size);
}

/*
 * Register the eventQueue of the cluster specified by clusterID.
 *
 * @param clusterID   Cluster ID of the EventQueue.
 * @param eventQueue  Pointer to the EventQueue.
 */
void InterClustersEventHandler::addEventQueue(const CLUSTER_INDEX_TYPE clusterID, EventQueue* eventQueue)
{
    m_vtEventQueue->at(clusterID) = eventQueue;

    eventQueue->regEventHandler(this);
}

/*
 * Add an eventin the queue of specified cluster.
 *
 * @param idx        The queue index of the collection.
 * @param clusterID  The cluster ID where the event to be added.
 * @param iStepOffset  offset from the current simulation step.
 */
void InterClustersEventHandler::addAnEvent(const BGSIZE idx, const CLUSTER_INDEX_TYPE clusterID, int iStepOffset)
{
#if !defined(USE_GPU)
    m_vtEventQueue->at(clusterID)->addAnEvent(idx, clusterID, iStepOffset);
#else // USE_GPU
    m_vtEventQueue->at(clusterID)->addAnInterClustersIncomingEvent(idx);
#endif // USE_GPU
}

