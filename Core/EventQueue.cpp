#include "EventQueue.h"

CUDA_CALLABLE EventQueue::EventQueue() 
{
    m_clusterID = 0;
    m_queueEvent = NULL;
    m_nMaxEvent = 0;
    m_idxQueue = 0;
    m_eventHandler = NULL;
}

CUDA_CALLABLE EventQueue::~EventQueue()
{
    if (m_queueEvent != NULL) {
        delete[] m_queueEvent;
        m_queueEvent = NULL;
    }
}

/*
 * Initializes the collection of queue.
 * 
 * @param nMaxEvent The number of event queue.
 * @param clusterID The cluster ID of cluster to be initialized.
 */
CUDA_CALLABLE void EventQueue::initEventQueue(BGSIZE nMaxEvent, CLUSTER_INDEX_TYPE clusterID)
{
    m_clusterID = clusterID;

    // allocate & initialize a memory for the event queue
    m_nMaxEvent = nMaxEvent;
    m_queueEvent = new BGQUEUE_ELEMENT[nMaxEvent];
}

/*
 * Initializes the collection of queue.
 * 
 * @param nMaxEvent   The number of event queue.
 * @param pQueueEvent Pointer to the collection of event queue.
 * @param clusterID   The cluster ID of cluster to be initialized.
 */
CUDA_CALLABLE void EventQueue::initEventQueue(BGSIZE nMaxEvent, BGQUEUE_ELEMENT* pQueueEvent, CLUSTER_INDEX_TYPE clusterID)
{
    m_clusterID = clusterID;

    // allocate & initialize a memory for the event queue
    m_nMaxEvent = nMaxEvent;
    m_queueEvent = pQueueEvent;
}

/*
 * Add an event in the queue.
 * 
 * @param idx The queue index of the collection.
 * @param clusterID The cluster ID where the event to be added.
 */
CUDA_CALLABLE void EventQueue::addAnEvent(const BGSIZE idx, const CLUSTER_INDEX_TYPE clusterID)
{
    if (clusterID != m_clusterID) {
        // notify the event to other cluster
        //assert( m_eventHandler != NULL );
        //m_eventHandler->addAnEvent(idx, clusterID);
    } else {
        BGQUEUE_ELEMENT &queue = m_queueEvent[idx];

        // Add to event queue

        // set a spike
        assert( !(queue & (0x1 << m_idxQueue)) );
        queue |= (0x1 << m_idxQueue);
    }
}

/*
 * Add an event in the queue.
 * 
 * @param idx The queue index of the collection.
 *e @param delay The delay descretized into time steps when the event will be triggered.
 */
CUDA_CALLABLE void EventQueue::addAnEvent(const BGSIZE idx, const int delay)
{
    BGQUEUE_ELEMENT &queue = m_queueEvent[idx];

    // Add to event queue

    // calculate index where to insert the event into queueEvent
    uint32_t idxQueue = m_idxQueue + delay;
    if ( idxQueue >= LENGTH_OF_DELAYQUEUE ) {
        idxQueue -= LENGTH_OF_DELAYQUEUE;
    }

    // set a spike
    assert( !(queue & (0x1 << idxQueue)) );
    queue |= (0x1 << idxQueue);
}

/*
 * Checks if there is an event in the queue.
 * 
 * @param idx The queue index of the collection.
 * @return true if there is an event.
 */
CUDA_CALLABLE bool EventQueue::checkAnEvent(const BGSIZE idx)
{
    BGQUEUE_ELEMENT &queue = m_queueEvent[idx];

    // check and reset the event
    bool r = queue & (0x1 << m_idxQueue);
    queue &= ~(0x1 << m_idxQueue);

    return r;
}

/*
 * Checks if there is an event in the queue.
 * 
 * @param idx The queue index of the collection.
 * @param delay The delay descretized into time steps when the event will be triggered.
 * @return true if there is an event.
 */
CUDA_CALLABLE bool EventQueue::checkAnEvent(const BGSIZE idx, const int delay)
{
    BGQUEUE_ELEMENT &queue = m_queueEvent[idx];

    // check and reset the event

    // calculate index where to check if there is an event
    int idxQueue = m_idxQueue - delay;
    if ( idxQueue < 0 ) {
        idxQueue += LENGTH_OF_DELAYQUEUE;
    }

    // check and reset a spike
    bool r = queue & (0x1 << idxQueue);
    queue &= ~(0x1 << idxQueue);

    return r;
}

/*
 * Clears events in the queue.
 * 
 * @param idx The queue index of the collection.
 */
CUDA_CALLABLE void EventQueue::clearAnEvent(const BGSIZE idx)
{
    BGQUEUE_ELEMENT &queue = m_queueEvent[idx];

    queue = 0;
}

/*
 * Advance one simulation step.
 * 
 */
CUDA_CALLABLE void EventQueue::advanceEventQueue()
{
    if ( ++m_idxQueue >= LENGTH_OF_DELAYQUEUE ) {
        m_idxQueue = 0;
    }
}

/**
 * Register an event handler.
 *
 * @param eventHandler  Pointer to the InterClustersEventHandler.
 */
CUDA_CALLABLE void EventQueue::regEventHandler(InterClustersEventHandler* eventHandler)
{
    m_eventHandler = eventHandler;
}

/*
 * Writes the queue data to the stream.
 *
 * output  stream to print out to.
 */
void EventQueue::serialize(ostream& output)
{
    output << m_idxQueue << ends;
    output << m_nMaxEvent << ends;

    for (BGSIZE idx = 0; idx < m_nMaxEvent; idx++) {
        output << m_queueEvent[idx] << ends;
    }
}

/*
 * Sets the data for the queue to input's data.
 *
 * input istream to read from.
 */
void EventQueue::deserialize(istream& input)
{
    input >> m_idxQueue; input.ignore();
    input >> m_nMaxEvent; input.ignore();

    for (BGSIZE idx = 0; idx < m_nMaxEvent; idx++) {
        input >> m_queueEvent[idx]; input.ignore();
    }
}
