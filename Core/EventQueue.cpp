#include "EventQueue.h"
#if defined(USE_GPU)
#include <helper_cuda.h>
#endif // USE_GPU

CUDA_CALLABLE EventQueue::EventQueue() 
{
    m_clusterID = 0;
    m_queueEvent = NULL;
    m_nMaxEvent = 0;
    m_idxQueue = 0;
    m_eventHandler = NULL;

#if defined(USE_GPU)
    m_nMaxInterClustersOutgoingEvents = 0;
    m_nInterClustersOutgoingEvents = 0;
    m_interClustersOutgoingEvents = NULL;

    m_nMaxInterClustersIncomingEvents = 0;
    m_nInterClustersIncomingEvents = 0;
    m_interClustersIncomingEvents = NULL;
#endif // USE_GPU
}

CUDA_CALLABLE EventQueue::~EventQueue()
{
    // de-alloacte memory for the collection of event queue
    if (m_queueEvent != NULL) {
        delete[] m_queueEvent;
        m_queueEvent = NULL;
    }

#if defined(USE_GPU)
    // In the device side memories, it should use cudaFree() and set NULL 
    // before destroying the object

    // de-allocate memory for the inter clusters outgoing event queue
    if (m_interClustersOutgoingEvents != NULL) {
        delete[] m_interClustersOutgoingEvents;
        m_interClustersOutgoingEvents = NULL;
    }

    // de-allocate memory for the inter clusters incoming event queue
    if (m_interClustersIncomingEvents != NULL) {
        delete[] m_interClustersIncomingEvents;
        m_interClustersIncomingEvents = NULL;
    }
#endif // USE_GPU
}

#if !defined(USE_GPU)
/*
 * Initializes the collection of queue.
 * 
 * @param clusterID The cluster ID of cluster to be initialized.
 * @param nMaxEvent The number of event queue.
 */
void EventQueue::initEventQueue(CLUSTER_INDEX_TYPE clusterID, BGSIZE nMaxEvent)
{
    m_clusterID = clusterID;

    // allocate & initialize a memory for the event queue
    m_nMaxEvent = nMaxEvent;
    m_queueEvent = new BGQUEUE_ELEMENT[nMaxEvent];
}

#else // USE_GPU
/*
 * Initializes the collection of queue in device memory.
 * 
 * @param clusterID                 The cluster ID of cluster to be initialized.
 * @param nMaxEvent                 The number of event queue.
 * @param pQueueEvent               Pointer to the collection of event queue.
 * @param nMaxInterClustersOutgoingEvents   The maximum number of the inter clusters 
 *                                          outgoing event queue.
 * @param interClustersOutgoingEvents       Pointer to the inter clusters outgoing event queue.
 * @param nMaxInterClustersIncomingEvents   The maximum number of the inter clusters 
 *                                          incoming event queue.
 * @param interClustersIncomingEvents       Pointer to the inter clusters incoming event queue.
 */
__device__ void EventQueue::initEventQueue(CLUSTER_INDEX_TYPE clusterID, BGSIZE nMaxEvent, BGQUEUE_ELEMENT* pQueueEvent, int nMaxInterClustersOutgoingEvents, interClustersOutgoingEvents_t* interClustersOutgoingEvents, int nMaxInterClustersIncomingEvents, interClustersIncomingEvents_t* interClustersIncomingEvents)
{
    m_clusterID = clusterID;

    // initialize a memory for the event queue
    m_nMaxEvent = nMaxEvent;
    m_queueEvent = pQueueEvent;

    // initialize a memory for the inter clusters outgoing event queue
    m_nMaxInterClustersOutgoingEvents = nMaxInterClustersOutgoingEvents;
    m_interClustersOutgoingEvents = interClustersOutgoingEvents;

    // initialize a memory for the inter clusters incoming event queue
    m_nMaxInterClustersIncomingEvents = nMaxInterClustersIncomingEvents;
    m_interClustersIncomingEvents = interClustersIncomingEvents;
}

/*
 * Initializes the collection of queue in host memory.
 * 
 * @param clusterID                 The cluster ID of cluster to be initialized.
 * @param nMaxEvent                 The number of event queue.
 * @param nMaxInterClustersOutgoingEvents   The maximum number of the inter clusters 
 *                                          outgoing event queue.
 * @param nMaxInterClustersIncomingEvents   The maximum number of the inter clusters 
 *                                          incoming event queue.
 */
__host__ void EventQueue::initEventQueue(CLUSTER_INDEX_TYPE clusterID, BGSIZE nMaxEvent, int nMaxInterClustersOutgoingEvents, int nMaxInterClustersIncomingEvents)
{
    m_clusterID = clusterID;

    // allocate & initialize a memory for the event queue
    m_nMaxEvent = nMaxEvent;
    m_queueEvent = new BGQUEUE_ELEMENT[nMaxEvent];

    // initialize a memory for the inter clusters outgoing event queue
    m_nMaxInterClustersOutgoingEvents = nMaxInterClustersOutgoingEvents;
    if (nMaxInterClustersOutgoingEvents != 0) {
        m_interClustersOutgoingEvents = new interClustersOutgoingEvents_t[nMaxInterClustersOutgoingEvents];
    }

    // initialize a memory for the inter clusters incoming event queue
    m_nMaxInterClustersIncomingEvents = nMaxInterClustersIncomingEvents;
    if (nMaxInterClustersIncomingEvents != 0) {
        m_interClustersIncomingEvents = new interClustersIncomingEvents_t[nMaxInterClustersIncomingEvents];
    }
}
#endif // USE_GPU

/*
 * Add an event in the queue.
 * 
 * @param idx The queue index of the collection.
 * @param clusterID The cluster ID where the event to be added.
 * @param iStepOffset  offset from the current simulation step.
 */
CUDA_CALLABLE void EventQueue::addAnEvent(const BGSIZE idx, const CLUSTER_INDEX_TYPE clusterID, int iStepOffset)
{
    if (clusterID != m_clusterID) {
#if !defined(__CUDA_ARCH__)
        // notify the event to other cluster
        assert( m_eventHandler != NULL );
        m_eventHandler->addAnEvent(idx, clusterID, iStepOffset);
#else // __CUDA_ARCH__
        // save the inter clusters outgoing events in the outgoing events queue
        OUTGOING_SYNAPSE_INDEX_TYPE idxOutSyn = SynapseIndexMap::getOutgoingSynapseIndex(clusterID, idx);

        // Because the function in multi-threads try to save events concurrently, we need to 
        // atomicaly store events data.
        int oldEventIdx, newEventIdx, currEventIdx  = m_nInterClustersOutgoingEvents;
        do {
            oldEventIdx = currEventIdx;
            newEventIdx = currEventIdx + 1;
            currEventIdx = atomicCAS(&m_nInterClustersOutgoingEvents, oldEventIdx, newEventIdx);
        } while (currEventIdx != oldEventIdx);
        
        // the thread acquires an empty slot (currEventIdx) in the queue
        // and save the inter clusters outgoing events there.
        m_interClustersOutgoingEvents[currEventIdx].idxSyn = idxOutSyn;
        m_interClustersOutgoingEvents[currEventIdx].iStepOffset = iStepOffset;
        assert( m_nInterClustersOutgoingEvents <= m_nMaxInterClustersOutgoingEvents );
#endif // __CUDA_ARCH__
    } else {
        // Add to event queue

        // adjust offset
        uint32_t idxQueue = m_idxQueue + iStepOffset;
        idxQueue = (idxQueue < LENGTH_OF_DELAYQUEUE) ? idxQueue : idxQueue - LENGTH_OF_DELAYQUEUE;

#if !defined(__CUDA_ARCH__)
        // When advanceNeurons and advanceSynapses in different clusters
        // are running concurrently, there might be race condition at
        // event queues (host only). For example, EventQueue::addAnEvent() is called
        // from advanceNeurons in cluster 0 and EventQueue::checkAnEvent()
        // is called from advanceSynapses in cluster 1. These functions
        // contain memory read/write operation at event queue and
        // consequntltly data race happens.
        // Therefore we need atomicaly set spiking data here.

        BGQUEUE_ELEMENT oldQueueEvent, newQueueEvent, currQueueEvent = m_queueEvent[idx];
        assert( !(currQueueEvent & (BGQUEUE_ELEMENT(0x1) << idxQueue)) );
        do {
            oldQueueEvent = currQueueEvent;
            newQueueEvent = currQueueEvent | (BGQUEUE_ELEMENT(0x1) << idxQueue);
            currQueueEvent = __sync_val_compare_and_swap(&m_queueEvent[idx], oldQueueEvent, newQueueEvent);
        } while (currQueueEvent != oldQueueEvent);
#else // __CUDA_ARCH__
        // set a spike
        BGQUEUE_ELEMENT &queue = m_queueEvent[idx];
        assert( !(queue & (BGQUEUE_ELEMENT(0x1) << idxQueue)) );
        queue |= (BGQUEUE_ELEMENT(0x1) << idxQueue);
#endif // __CUDA_ARCH__
    }
}

#if defined(USE_GPU)
/*
 * Add an event in the inter clusters incoming event queue
 *
 * @param idx The queue index of the collection.
 * @param iStepOffset  offset from the current simulation step.
 */
__host__ void EventQueue::addAnInterClustersIncomingEvent(const BGSIZE idx, int iStepOffset)
{
    // Because the function in multi-threads try to save events concurrently, we need to
    // atomicaly store events data.
    int oldEventIdx, newEventIdx, currEventIdx  = m_nInterClustersIncomingEvents;
    do {
        oldEventIdx = currEventIdx;
        newEventIdx = currEventIdx + 1;
        currEventIdx = __sync_val_compare_and_swap(&m_nInterClustersIncomingEvents, oldEventIdx, newEventIdx);
    } while (currEventIdx != oldEventIdx);

    // the thread acquires an empty slot (currEventIdx) in the queue
    // and save the inter clusters outgoing events there.
    m_interClustersIncomingEvents[currEventIdx].idxSyn = idx;
    m_interClustersIncomingEvents[currEventIdx].iStepOffset = iStepOffset;
    assert( m_nInterClustersIncomingEvents <= m_nMaxInterClustersIncomingEvents );
}

/*
 * Process inter clusters outgoing events that are stored in the buffer.
 *
 * @param pEventQueue   Pointer to the EventQueue object in device.
 */
__host__ void EventQueue::processInterClustersOutgoingEvents(EventQueue* pEventQueue)
{
    // copy preSpikeQueue data from device to host.
    interClustersOutgoingEvents_t *pInterClustersOutgoingEvents_h;
    int nInterClustersOutgoingEvents_h;

    // get event queue data in device
    getInterClustersOutgoingEventPointerInDevice(pEventQueue, &pInterClustersOutgoingEvents_h);
    getNInterClustersOutgoingEventsInDevice(pEventQueue, &nInterClustersOutgoingEvents_h);

    // set the queue index in host
    m_nInterClustersOutgoingEvents = nInterClustersOutgoingEvents_h;

    // copy the inter clusters outgoing event queue data to host
    checkCudaErrors( cudaMemcpy ( m_interClustersOutgoingEvents, pInterClustersOutgoingEvents_h, nInterClustersOutgoingEvents_h * sizeof( interClustersOutgoingEvents_t ), cudaMemcpyDeviceToHost ) );

    for (int i = 0; i < m_nInterClustersOutgoingEvents; i++) {
        OUTGOING_SYNAPSE_INDEX_TYPE idx = m_interClustersOutgoingEvents[i].idxSyn;
        int iStepOffset = m_interClustersOutgoingEvents[i].iStepOffset;
        CLUSTER_INDEX_TYPE iCluster = SynapseIndexMap::getClusterIndex(idx);
        BGSIZE iSyn = SynapseIndexMap::getSynapseIndex(idx);
        assert( iCluster != m_clusterID);

        // notify the event to other cluster
        m_eventHandler->addAnEvent(iSyn, iCluster, iStepOffset);
    }

    m_nInterClustersOutgoingEvents = 0;
}

/*
 * Process inter clusters incoming events that are stored in the buffer.
 *
 * @param pEventQueue   Pointer to the EventQueue object in device.
 */
__host__ void EventQueue::processInterClustersIncomingEvents(EventQueue* pEventQueue)
{
    // copy preSpikeQueue from host to device
    BGQUEUE_ELEMENT *pInterClustersIncomingEvents_h;

    // get the buffer pointer in device
    getInterClustersIncomingEventPointerInDevice(pEventQueue, &pInterClustersIncomingEvents_h);

    // copy the inter clusters incoming event queue data from host to device
    checkCudaErrors( cudaMemcpy ( pInterClustersIncomingEvents_h, m_interClustersIncomingEvents, m_nInterClustersIncomingEvents * sizeof( interClustersIncomingEvents_t ), cudaMemcpyHostToDevice ) );

    // set the number of events stored in the inter clusters outgoing/incoming event queue to device
    setNInterClustersOutgoingEventsDevice <<< 1, 1 >>> ( pEventQueue, m_nInterClustersOutgoingEvents );
    setNInterClustersIncomingEventsDevice <<< 1, 1 >>> ( pEventQueue, m_nInterClustersIncomingEvents );

    // process inter clusters incoming spikes -- call device side preSpikeQueue object
    // CUDA parameters
    const int threadsPerBlock = 256;
    int blocksPerGrid = ( m_nInterClustersIncomingEvents + threadsPerBlock - 1 ) / threadsPerBlock;

    processInterClustersIncomingEventsDevice <<< blocksPerGrid, threadsPerBlock >>> ( m_nInterClustersIncomingEvents, pEventQueue );

    // reset the host & device side number of incoming events
    m_nInterClustersIncomingEvents = 0;
    setNInterClustersIncomingEventsDevice <<< 1, 1 >>> ( pEventQueue, m_nInterClustersIncomingEvents );
}

/*
 * Process inter clusters incoming events that are stored in the buffer.
 *
 * @param idx The queue index of the collection.
 */
__device__ void EventQueue::processInterClustersIncomingEventsInDevice(int idx)
{
    BGSIZE iSyn = m_interClustersIncomingEvents[idx].idxSyn;
    int iStepOffset = m_interClustersIncomingEvents[idx].iStepOffset;

    BGQUEUE_ELEMENT &queue = m_queueEvent[iSyn];

    // Add to event queue

    // calculate index where to insert the event into queueEvent
    uint32_t idxQueue = m_idxQueue + iStepOffset;
    idxQueue = ( idxQueue >= LENGTH_OF_DELAYQUEUE ) ? idxQueue - LENGTH_OF_DELAYQUEUE : idxQueue;

    // set a spike
    assert( !(queue & (BGQUEUE_ELEMENT(0x1) << idxQueue)) );
    queue |= (BGQUEUE_ELEMENT(0x1) << idxQueue);
}

#endif // USE_GPU

/*
 * Add an event in the queue.
 * 
 * @param idx The queue index of the collection.
 * @param delay The delay descretized into time steps when the event will be triggered.
 * @param iStepOffset  offset from the current simulation step.
 */
CUDA_CALLABLE void EventQueue::addAnEvent(const BGSIZE idx, const int delay, int iStepOffset)
{

    // Add to event queue
    assert( static_cast<uint32_t>(delay + iStepOffset) < LENGTH_OF_DELAYQUEUE );

    // calculate index where to insert the event into queueEvent
    uint32_t idxQueue = m_idxQueue + delay + iStepOffset;
    if ( idxQueue >= LENGTH_OF_DELAYQUEUE ) {
        idxQueue -= LENGTH_OF_DELAYQUEUE;
    }

    // set a spike
    BGQUEUE_ELEMENT &queue = m_queueEvent[idx];
    assert( !(queue & (BGQUEUE_ELEMENT(0x1) << idxQueue)) );
    queue |= (BGQUEUE_ELEMENT(0x1) << idxQueue);
}

/*
 * Checks if there is an event in the queue.
 * 
 * @param idx The queue index of the collection.
 * @param iStepOffset  offset from the current simulation step.
 * @return true if there is an event.
 */
CUDA_CALLABLE bool EventQueue::checkAnEvent(const BGSIZE idx, int iStepOffset)
{
    // check and reset the event
    uint32_t idxQueue = m_idxQueue + iStepOffset;
    if ( idxQueue >= LENGTH_OF_DELAYQUEUE ) {
        idxQueue -= LENGTH_OF_DELAYQUEUE;
    }
    BGQUEUE_ELEMENT &queue = m_queueEvent[idx];
    bool r = queue & (BGQUEUE_ELEMENT(0x1) << idxQueue);
    queue &= ~(BGQUEUE_ELEMENT(0x1) << idxQueue);

    return r;
}

/*
 * Checks if there is an event in the queue.
 * 
 * @param idx The queue index of the collection.
 * @param delay The delay descretized into time steps when the event will be triggered.
 * @param iStepOffset  offset from the current simulation step.
 * @return true if there is an event.
 */
CUDA_CALLABLE bool EventQueue::checkAnEvent(const BGSIZE idx, const int delay, int iStepOffset)
{

    // check and reset the event

    // calculate index where to check if there is an event
    assert( delay > iStepOffset );
    int idxQueue = m_idxQueue - delay + iStepOffset;
    idxQueue = ( idxQueue < 0 ) ? idxQueue + LENGTH_OF_DELAYQUEUE : idxQueue;

#if !defined(USE_GPU)
    // When advanceNeurons and advanceSynapses in different clusters
    // are running concurrently, there might be race condition at
    // event queues. For example, EventQueue::addAnEvent() is called
    // from advanceNeurons in cluster 0 and EventQueue::checkAnEvent()
    // is called from advanceSynapses in cluster 1. These functions
    // contain memory read/write operation at event queue and
    // consequntltly data race happens.
    // Therefore we need atomicaly check and reset spiking data here.

    BGQUEUE_ELEMENT oldQueueEvent, newQueueEvent, currQueueEvent = m_queueEvent[idx];
    bool r = currQueueEvent & (BGQUEUE_ELEMENT(0x1) << idxQueue);
    do {
        oldQueueEvent = currQueueEvent;
        newQueueEvent = currQueueEvent & ~(BGQUEUE_ELEMENT(0x1) << idxQueue);
        currQueueEvent = __sync_val_compare_and_swap(&m_queueEvent[idx], oldQueueEvent, newQueueEvent);
    } while (currQueueEvent != oldQueueEvent);
#else // USE_GPU
    // check and reset a spike
    BGQUEUE_ELEMENT &queue = m_queueEvent[idx];
    bool r = queue & (BGQUEUE_ELEMENT(0x1) << idxQueue);
    queue &= ~(BGQUEUE_ELEMENT(0x1) << idxQueue);
#endif // USE_GPU

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
 * Advance simulation step.
 *
 * @param iStep        simulation step to advance.
 */
CUDA_CALLABLE void EventQueue::advanceEventQueue(int iStep)
{
    m_idxQueue += iStep;
    m_idxQueue = (m_idxQueue < LENGTH_OF_DELAYQUEUE) ? m_idxQueue : m_idxQueue - LENGTH_OF_DELAYQUEUE;
}

/*
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
 * We don't need to save inter clusters event data, because
 * these data are temporary used between advanceNeurons() and advanceSynapses().
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
    BGSIZE nMaxEvent;

    input >> m_idxQueue; input.ignore();
    input >> nMaxEvent; input.ignore();

    // If the assertion hits, that means we restore simulation by using stored data
    // of different configuration file.
    assert( m_nMaxEvent == nMaxEvent );
    m_nMaxEvent = nMaxEvent;

    for (BGSIZE idx = 0; idx < m_nMaxEvent; idx++) {
        input >> m_queueEvent[idx]; input.ignore();
    }
}

#if defined(USE_GPU)
/*
 * Create an EventQueue class object in device
 *
 * @param pEventQueue_d    Device memory address to save the pointer of created EventQueue object.
 */
__host__ void EventQueue::createEventQueueInDevice(EventQueue** pEventQueue_d)
{
    EventQueue **pEventQueue_t; // temporary buffer to save pointer to EventQueue object.

    // allocate device memory for the buffer.
    checkCudaErrors( cudaMalloc( ( void ** ) &pEventQueue_t, sizeof( EventQueue * ) ) );

    // allocate three device memories used in the EventQueue
    BGSIZE nMaxEvent = m_nMaxEvent;
    BGQUEUE_ELEMENT* queueEvent;
    checkCudaErrors( cudaMalloc(&queueEvent, nMaxEvent * sizeof(BGQUEUE_ELEMENT)) );

    int nMaxInterClustersOutgoingEvents = m_nMaxInterClustersOutgoingEvents;
    interClustersOutgoingEvents_t* interClustersOutgoingEvents = NULL;
    if (nMaxInterClustersOutgoingEvents != 0) {
        checkCudaErrors( cudaMalloc(&interClustersOutgoingEvents, nMaxInterClustersOutgoingEvents * sizeof(interClustersOutgoingEvents_t)) );
    }

    int nMaxInterClustersIncomingEvents = m_nMaxInterClustersIncomingEvents;
    interClustersIncomingEvents_t* interClustersIncomingEvents = NULL;
    if (nMaxInterClustersIncomingEvents != 0) {
        checkCudaErrors( cudaMalloc(&interClustersIncomingEvents, nMaxInterClustersIncomingEvents * sizeof(interClustersIncomingEvents_t)) );
    }

    // create an EventQueue object in device memory.
    allocEventQueueDevice <<< 1, 1 >>> ( pEventQueue_t, m_clusterID, nMaxEvent, queueEvent, nMaxInterClustersOutgoingEvents, interClustersOutgoingEvents, nMaxInterClustersIncomingEvents, interClustersIncomingEvents);

    // save the pointer of the object.
    checkCudaErrors( cudaMemcpy ( pEventQueue_d, pEventQueue_t, sizeof( EventQueue * ), cudaMemcpyDeviceToHost ) );

    // free device memory for the buffer.
    checkCudaErrors( cudaFree( pEventQueue_t ) );
}

/*
 * Delete an EventQueue class object in device
 *
 * @param pEventQueue_d    Pointer to the EventQueue object to be deleted in device.
 */
void EventQueue::deleteEventQueueInDevice(EventQueue* pEventQueue_d)
{
    BGQUEUE_ELEMENT *pQueueEvent_h;
    interClustersOutgoingEvents_t *pInterClustersOutgoingEvents_h;
    BGQUEUE_ELEMENT *pInterClustersIncomingEvents_h;

    // get pointers to the queue data buffers in device memory
    getQueueEventPointerInDevice(pEventQueue_d, &pQueueEvent_h);
    getInterClustersOutgoingEventPointerInDevice(pEventQueue_d, &pInterClustersOutgoingEvents_h);
    getInterClustersIncomingEventPointerInDevice(pEventQueue_d, &pInterClustersIncomingEvents_h);

    // free three device memories used in the EventQueue
    checkCudaErrors( cudaFree( pQueueEvent_h ) );
    if (pInterClustersOutgoingEvents_h != NULL) {
        checkCudaErrors( cudaFree( pInterClustersOutgoingEvents_h ) );
    }
    if (pInterClustersIncomingEvents_h != NULL) {
        checkCudaErrors( cudaFree( pInterClustersIncomingEvents_h ) );
    }

    // delete preSpikeQueue object in device memory.
    deleteEventQueueDevice <<< 1, 1 >>> ( pEventQueue_d );
}

/*
 * Copy EventQueue data from host to device
 *
 * @param pEventQueue   Pointer to the EventQueue object in device.
 */
__host__ void EventQueue::copyEventQueueHostToDevice(EventQueue* pEventQueue)
{
    // We need deep copy event queue data here, when we resume simulation by
    // restoring the previous status data.
    // However we don't need to copy inter clusters event data, because
    // these data are temporary used between advanceNeurons() and advanceSynapses().
    BGQUEUE_ELEMENT *pQueueEvent_h;
    getQueueEventPointerInDevice(pEventQueue, &pQueueEvent_h);

    // copy event queue data from host to device (deep copy)
    checkCudaErrors( cudaMemcpy ( pQueueEvent_h, m_queueEvent,
            m_nMaxEvent * sizeof( BGQUEUE_ELEMENT ), cudaMemcpyHostToDevice ) );

    // set queue index to device
    setQueueIndexDevice <<< 1, 1 >>> ( pEventQueue, m_idxQueue );
}

/*
 * Copy EventQueue data from device to host
 *
 * @param pEventQueue   Pointer to the EventQueue object in device.
 */
__host__ void EventQueue::copyEventQueueDeviceToHost(EventQueue* pEventQueue)
{
    // We need deep copy event queue data here, when we store simulation status.
    // However we don't need to copy inter clusters event data, because
    // these data are temporary used between advanceNeurons() and advanceSynapses().
    BGQUEUE_ELEMENT *pQueueEvent_h;
    getQueueEventPointerInDevice(pEventQueue, &pQueueEvent_h);

    // copy event queue data from device to host (deep copy)
    checkCudaErrors( cudaMemcpy ( pQueueEvent_h, m_queueEvent,
            m_nMaxEvent * sizeof( BGQUEUE_ELEMENT ), cudaMemcpyHostToDevice ) );

    // set queue index to host
    uint32_t idxQueue_h;
    EventQueue::getQueueIndexInDevice(pEventQueue, &idxQueue_h);

    // set the queue index in host
    m_idxQueue = idxQueue_h;
}

/*
 * Get index indicating the current time slot in the delayed queue in device
 *
 * @param pEventQueue   Pointer to the EventQueue object in device.
 * @param idxQueue_h    Address to the data to get.
 */
void EventQueue::getQueueIndexInDevice(EventQueue* pEventQueue, uint32_t* idxQueue_h)
{
    uint32_t *pIdxQueue_d;  // temporary buffer to save queue index

    // allocate device memory for the buffer.
    checkCudaErrors( cudaMalloc( ( void ** ) &pIdxQueue_d, sizeof( uint32_t ) ) );

    // get queue index in device memory
    getQueueIndexDevice <<< 1, 1 >>> ( pEventQueue, pIdxQueue_d );

   // copy the queue index to host.
    checkCudaErrors( cudaMemcpy ( idxQueue_h, pIdxQueue_d, sizeof( uint32_t ), cudaMemcpyDeviceToHost ) );

    // free device memory for the buffer.
    checkCudaErrors( cudaFree( pIdxQueue_d ) );
}

/*
 * Get pointer to the collection of event queue in device
 *
 * @param pEventQueue     Pointer to the EventQueue object in device.
 * @param pQueueEvent_h   Address to the data to get.
 */
void EventQueue::getQueueEventPointerInDevice(EventQueue* pEventQueue, BGQUEUE_ELEMENT** pQueueEvent_h) 
{
    BGQUEUE_ELEMENT **pQueueEvent_d; // temporary buffer to save pointer to event queue data

    // allocate device memory for the buffer.
    checkCudaErrors( cudaMalloc( ( void ** ) &pQueueEvent_d, sizeof( BGQUEUE_ELEMENT* ) ) );

    // get the address of event queue data in device memory
    // EventQueue object in device memory resides in device heap,
    // so we can not use cudaMemcpy to copy the object to host memory,
    // and therefore use the kernel function to get the pointer.

    getQueueEventPointerDevice <<< 1, 1 >>> ( pEventQueue, pQueueEvent_d );

    // copy the pointer of the object to host.
    checkCudaErrors( cudaMemcpy ( pQueueEvent_h, pQueueEvent_d, sizeof( BGQUEUE_ELEMENT* ), cudaMemcpyDeviceToHost ) );

    // free device memory for the buffer.
    checkCudaErrors( cudaFree( pQueueEvent_d ) );
}

/*
 * Get number of events stored in the inter clusters outgoing event queue in device
 *
 * @param pEventQueue     Pointer to the EventQueue object in device.
 * @param nInterClustersOutgoingEvents_h   Address to the data to get.
 */
void EventQueue::getNInterClustersOutgoingEventsInDevice(EventQueue* pEventQueue, int* nInterClustersOutgoingEvents_h)
{
    int *pNInterClustersOutgoingEvents_d; // temporary buffer to save the number

    // allocate device memory for the buffer.
    checkCudaErrors( cudaMalloc( ( void ** ) &pNInterClustersOutgoingEvents_d, sizeof( int ) ) );

    // get the number in device memory
    getNInterClustersOutgoingEventsDevice <<< 1, 1 >>> ( pEventQueue, pNInterClustersOutgoingEvents_d );

    // copy the queue index to host.
    checkCudaErrors( cudaMemcpy ( nInterClustersOutgoingEvents_h, pNInterClustersOutgoingEvents_d, sizeof( int ), cudaMemcpyDeviceToHost ) );

    // free device memory for the buffer.
    checkCudaErrors( cudaFree( pNInterClustersOutgoingEvents_d ) );
}

/*
 * Get pointer to the inter clusters outgoing event queue in device
 *
 * @param pEventQueue     Pointer to the EventQueue object in device.
 * @param pInterClustersOutgoingEvents_h   Address to the data to get.
 */
void EventQueue::getInterClustersOutgoingEventPointerInDevice(EventQueue* pEventQueue, interClustersOutgoingEvents_t** pInterClustersOutgoingEvents_h)
{
    interClustersOutgoingEvents_t **pInterClustersOutgoingEvents_d; // temporary buffer to save pointer to inter clusters outgoing event queue data

    // allocate device memory for the buffer
    checkCudaErrors( cudaMalloc( ( void ** ) &pInterClustersOutgoingEvents_d, sizeof( interClustersOutgoingEvents_t * ) ) );

    // get the address of inter clusters outgoing event queue data in device memory
    // EventQueue object in device memory resides in device heap, 
    // so we can not use cudaMemcpy to copy the object to host memory,
    // and therefore use the kernel function to get the pointer.

    getInterClustersOutgoingEventPointerDevice <<< 1, 1 >>> ( pEventQueue, pInterClustersOutgoingEvents_d );

    // copy the pointer of the object to host.
    checkCudaErrors( cudaMemcpy ( pInterClustersOutgoingEvents_h, pInterClustersOutgoingEvents_d, sizeof( interClustersOutgoingEvents_t* ), cudaMemcpyDeviceToHost ) );

    // free device memory for the buffer.
    checkCudaErrors( cudaFree( pInterClustersOutgoingEvents_d ) );
}

/*
 * Get pointer to the inter clusters incoming event queue in device
 *
 * @param pEventQueue     Pointer to the EventQueue object in device.
 * @param pInterClustersIncomingEvents_h   Address to the data to get.
 */
void EventQueue::getInterClustersIncomingEventPointerInDevice(EventQueue* pEventQueue, BGQUEUE_ELEMENT** pInterClustersIncomingEvents_h)
{
    interClustersIncomingEvents_t **pInterClustersIncomingEvents_d; // temporary buffer to save pointer to inter clusters incoming event queue data

    // allocate device memory for the buffer
    checkCudaErrors( cudaMalloc( ( void ** ) &pInterClustersIncomingEvents_d, sizeof( interClustersIncomingEvents_t * ) ) );

    // get the address of inter clusters incoming event queue data in device memory
    // EventQueue object in device memory resides in device heap, 
    // so we can not use cudaMemcpy to copy the object to host memory,
    // and therefore use the kernel function to get the pointer.

    getInterClustersIncomingEventPointerDevice <<< 1, 1 >>> ( pEventQueue, pInterClustersIncomingEvents_d );

    // copy the pointer of the object to host.
    checkCudaErrors( cudaMemcpy ( pInterClustersIncomingEvents_h, pInterClustersIncomingEvents_d, sizeof( interClustersIncomingEvents_t* ), cudaMemcpyDeviceToHost ) );

    // free device memory for the buffer.
    checkCudaErrors( cudaFree( pInterClustersIncomingEvents_d ) );
}


/* -------------------------------------*\
|* # CUDA Global Functions
\* -------------------------------------*/

/*
 * Creates a EventQueue object in device memory.
 *
 * @param[in/out] pQueueEvent           Pointer to the collection of event queue.
 * @param[in] clusterID                 The cluster ID of cluster to be initialized.
 * @param[in] nMaxEvent                 The number of event queue.
 * @param[in] pQueueEvent               Pointer to the collection of event queue.
 * @param[in] nMaxInterClustersOutgoingEvents   The maximum number of the inter clusters
 *                                              outgoing event queue.
 * @param[in] interClustersOutgoingEvents       Pointer to the inter clusters outgoing event queue.
 * @param[in] nMaxInterClustersIncomingEvents   The maximum number of the inter clusters
 *                                              incoming event queue.
 * @param[in] interClustersIncomingEvents       Pointer to the inter clusters incoming event queue.
 */
__global__ void allocEventQueueDevice(EventQueue **pEventQueue, CLUSTER_INDEX_TYPE clusterID, BGSIZE nMaxEvent, BGQUEUE_ELEMENT* pQueueEvent, int nMaxInterClustersOutgoingEvents, interClustersOutgoingEvents_t* interClustersOutgoingEvents, int nMaxInterClustersIncomingEvents, interClustersIncomingEvents_t* interClustersIncomingEvents)
{
    *pEventQueue = new EventQueue();
    (*pEventQueue)->initEventQueue(clusterID, nMaxEvent, pQueueEvent, nMaxInterClustersOutgoingEvents, interClustersOutgoingEvents, nMaxInterClustersIncomingEvents, interClustersIncomingEvents);
}

/*
 * Delete a EventQueue object in device memory.
 *
 * @param[in] pEventQueue          Pointer to the EventQueue object to be deleted.
 */
__global__ void deleteEventQueueDevice(EventQueue *pEventQueue)
{
    if (pEventQueue != NULL) {
        // event queue buffers were freed by calling cudaFree()
        pEventQueue->m_queueEvent = NULL;
        pEventQueue->m_interClustersOutgoingEvents = NULL;
        pEventQueue->m_interClustersIncomingEvents = NULL;

        delete pEventQueue;
    }
}

/*
 * Get the address of event queue data in device memory
 *
 * @param[in] pEventQueue          Pointer to the EventQueue object.
 * @param[in/out] pQueueEvent      Buffer to save pointer to event queue data.
 */
__global__ void getQueueEventPointerDevice(EventQueue *pEventQueue, BGQUEUE_ELEMENT **pQueueEvent)
{
    *pQueueEvent = pEventQueue->m_queueEvent;
}

/*
 * Set queue index to device
 *
 * @param[in] pEventQueue          Pointer to the EventQueue object.
 * @param[in] idxQueue             Queue index.
 */
__global__ void setQueueIndexDevice(EventQueue *pEventQueue, uint32_t idxQueue)
{
    pEventQueue->m_idxQueue = idxQueue;
}

/*
 * get queue index in device memory
 *
 * @param[in] pEventQueue          Pointer to the EventQueue object.
 * @param[in/out] idxQueue         Buffer to save Queue index.
 */
__global__ void getQueueIndexDevice(EventQueue *pEventQueue, uint32_t *idxQueue)
{
    *idxQueue = pEventQueue->m_idxQueue;
}

/*
 * get the address of inter clusters outgoing event queue data in device memory
 *
 * @param[in] pEventQueue                        Pointer to the EventQueue object.
 * @param[in/out] pInterClustersOutgoingEvents   Buffer to save pointer to inter clusters outgoing
 *                                               event queue data.
 */
__global__ void getInterClustersOutgoingEventPointerDevice(EventQueue *pEventQueue, interClustersOutgoingEvents_t **pInterClustersOutgoingEvents)
{
    *pInterClustersOutgoingEvents = pEventQueue->m_interClustersOutgoingEvents;
}

/*
 * get the address of inter clusters incoming event queue data in device memory
 *
 * @param[in] pEventQueue                        Pointer to the EventQueue object.
 * @param[in/out] pInterClustersIncomingEvents   Buffer to save pointer to inter clusters incoming
 *                                               event queue data.
 */
__global__ void getInterClustersIncomingEventPointerDevice(EventQueue *pEventQueue, interClustersIncomingEvents_t **pInterClustersIncomingEvents)
{
    *pInterClustersIncomingEvents = pEventQueue->m_interClustersIncomingEvents;
}

/*
 * get the number events stored in the inter clusters outgoing event queue in device memory
 *
 * @param[in] pEventQueue                         Pointer to the EventQueue object.
 * @param[in/out] pNInterClustersOutgoingEvents   Buffer to save the number.
 */
__global__ void getNInterClustersOutgoingEventsDevice(EventQueue *pEventQueue, int *pNInterClustersOutgoingEvents)
{
    *pNInterClustersOutgoingEvents = pEventQueue->m_nInterClustersOutgoingEvents;
}

/*
 * set the number events stored in the inter clusters outgoing event queue in device memory
 *
 * @param[in] pEventQueue                         Pointer to the EventQueue object.
 * @param[in] nInterClustersOutgoingEvents        The number events in the queue..
 */
__global__ void setNInterClustersOutgoingEventsDevice(EventQueue *pEventQueue, int nInterClustersOutgoingEvents)
{
    pEventQueue->m_nInterClustersOutgoingEvents = nInterClustersOutgoingEvents;
}

/*
 * set the number events stored in the inter clusters incoming event queue in device memory
 *
 * @param[in] pEventQueue                         Pointer to the EventQueue object.
 * @param[in] nInterClustersIncomingEvents        The number events in the queue..
 */
__global__ void setNInterClustersIncomingEventsDevice(EventQueue *pEventQueue, int nInterClustersIncomingEvents)
{
    pEventQueue->m_nInterClustersIncomingEvents = nInterClustersIncomingEvents;
}

/*
 * Process inter clusters incoming events that are stored in the buffer.
 *
 * @param[in] nInterClustersIncomingEvents        The number events in the queue.
 */
__global__ void processInterClustersIncomingEventsDevice(int nInterClustersIncomingEvents, EventQueue *pEventQueue)
{
    // The usual thread ID calculation and guard against excess threads
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= nInterClustersIncomingEvents )
      return;

    pEventQueue->processInterClustersIncomingEventsInDevice(idx);
}

#endif // USE_GPU

