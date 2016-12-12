#include "EventQueue.h"

CUDA_CALLABLE EventQueue::EventQueue() :
    m_queueEvent(NULL),
    m_nMaxEvent(0),
    m_idxQueue(0)
{
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
 */
CUDA_CALLABLE void EventQueue::initEventQueue(BGSIZE nMaxEvent)
{
    // allocate & initialize a memory for the event queue
    m_nMaxEvent = nMaxEvent;
    m_queueEvent = new BGQUEUE_ELEMENT[nMaxEvent];
}

/*
 * Add an event in the queue.
 * 
 * @param idx The queue index of the collection.
 * @param delay The delay descretized into time steps when the event will be triggered.
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
