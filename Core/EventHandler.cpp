#include "EventHandler.h"
#include "EventQueue.h"

EventHandler::EventHandler() : m_vtEventQueue(NULL)
{
}

EventHandler::~EventHandler()
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
void EventHandler::initEventHandler(const int size)
{
    m_vtEventQueue = new vector<EventQueue *>(size);
}

/*
 * Register the eventQueue of the cluster specified by clusterID.
 *
 * @param clusterID   Cluster ID of the EventQueue.
 * @param eventQueue  Pointer to the EventQueue.
 */
void EventHandler::addEventQueue(const CLUSTER_INDEX_TYPE clusterID, EventQueue* eventQueue)
{
    m_vtEventQueue->at(clusterID) = eventQueue;

    eventQueue->regEventHandler(this);
}

/*
 * Add an eventin the queue of specified cluster.
 *
 * @param idx        The queue index of the collection.
 * @param clusterID  The cluster ID where the event to be added.
 */
void EventHandler::addAnEvent( const BGSIZE idx, const CLUSTER_INDEX_TYPE clusterID)
{
    m_vtEventQueue->at(clusterID)->addAnEvent(idx, clusterID);
}

