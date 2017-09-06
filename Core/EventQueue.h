/**
 *	@file EventQueue.h
 *
 *	@brief A collection of event queue.
 */

/**
 **
 ** @class EventQueue EventQueue.h "EventQueue.h"
 **
 ** \latexonly  \subsubsection*{Implementation} \endlatexonly
 ** \htmlonly   <h3>Implementation</h3> \endhtmlonly
 **
 ** The EventQueue class is a collection of event queue. The number of
 ** event queue is specified by nMaxEvent at initEventQueue(). 
 ** Each event queue stores events by calling addAnEvent() method,
 ** where idx is the queue index of the collection and delay is the delay
 ** in simulation step when the event will be triggered. 
 **
 ** \latexonly  \subsubsection*{Credits} \endlatexonly
 ** \htmlonly   <h3>Credits</h3> \endhtmlonly
 **
 ** Some models in this simulator is a rewrite of CSIM (2006) and other
 ** work (Stiber and Kawasaki (2007?))
 **
 **
 **     @author Fumitaka Kawasaki
 **/

#pragma once

#include "SimulationInfo.h"
#include "SynapseIndexMap.h"
#include "InterClustersEventHandler.h"

typedef struct {
    OUTGOING_SYNAPSE_INDEX_TYPE idxSyn;
    int iStepOffset;
}  interClustersOutgoingEvents_t;

typedef struct {
    BGSIZE idxSyn;
    int iStepOffset;
} interClustersIncomingEvents_t;
            
class EventQueue
{
    //! Type of a queue element.
    #define BGQUEUE_ELEMENT uint32_t
    //! The bit length of a queue element.
    #define LENGTH_OF_DELAYQUEUE        ( sizeof(BGQUEUE_ELEMENT) * 8 )

    public:
        //! The constructor for EventQueue.
        CUDA_CALLABLE EventQueue();

        //! The destructor for EventQueue.
        CUDA_CALLABLE virtual ~EventQueue();

#if !defined(USE_GPU)
        /**
         * Initializes the collection of queue.
         *
         * @param clusterID The cluster ID of cluster to be initialized.
         * @param nMaxEvent The number of event queue.
         */
        void initEventQueue(CLUSTER_INDEX_TYPE clusterID, BGSIZE nMaxEvent);

#else // USE_GPU
        /**
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
        __device__ void initEventQueue(CLUSTER_INDEX_TYPE clusterID, BGSIZE nMaxEvent, BGQUEUE_ELEMENT* pQueueEvent, int nMaxInterClustersOutgoingEvents, interClustersOutgoingEvents_t* interClustersOutgoingEvents, int nMaxInterClustersIncomingEvents, interClustersIncomingEvents_t* interClustersIncomingEvents);

        /**
         * Initializes the collection of queue in host memory.
         *
         * @param clusterID                 The cluster ID of cluster to be initialized.
         * @param nMaxEvent                 The number of event queue.
         * @param nMaxInterClustersOutgoingEvents   The maximum number of the inter clusters
         *                                          outgoing event queue.
         * @param nMaxInterClustersIncomingEvents   The maximum number of the inter clusters
         *                                          incoming event queue.
         */
        __host__ void initEventQueue(CLUSTER_INDEX_TYPE clusterID, BGSIZE nMaxEvent, int nMaxInterClustersOutgoingEvents, int nMaxInterClustersIncomingEvents);

        /**
         * Add an event in the inter clusters incoming event queue
         *
         * @param idx The queue index of the collection.
         * @param iStepOffset  offset from the current simulation step.
         */
        __host__ void addAnInterClustersIncomingEvent(const BGSIZE idx, int iStepOffset);

        /**
         * Process inter clusters outgoing events that are stored in the buffer.
         *
         * @param pEventQueue   Pointer to the EventQueue object in device.
         */
        __host__ void processInterClustersOutgoingEvents(EventQueue* pEventQueue);

        /**
         * Process inter clusters incoming events that are stored in the buffer.
         *
         * @param pEventQueue   Pointer to the EventQueue object in device.
         */
        __host__ void processInterClustersIncomingEvents(EventQueue* pEventQueue);

        /**
         * Process inter clusters incoming events that are stored in the buffer.
         *
         * @param idx The queue index of the collection.
         */
        __device__ void processInterClustersIncomingEventsInDevice(int idx);

#endif // USE_GPU

        /**
         * Add an event in the queue.
         *
         * @param idx       The queue index of the collection.
         * @param clusterID The cluster ID where the event to be added.
         * @param iStepOffset  offset from the current simulation step.
         */
#if !defined(USE_GPU)
        void addAnEvent(const BGSIZE idx, const CLUSTER_INDEX_TYPE clusterID, int iStepOffset);
#else // USE_GPU
        __device__ void addAnEvent(const BGSIZE idx, const CLUSTER_INDEX_TYPE clusterID, int iStepOffset);
#endif // USE_GPU

        /**
         * Add an event in the queue.
         *
         * @param idx   The queue index of the collection.
         * @param delay The delay descretized into time steps when the event will be triggered.
         * @param iStepOffset  offset from the current simulation step.
         */
        CUDA_CALLABLE void addAnEvent(const BGSIZE idx, const int delay, int iStepOffset);

        /**
         * Checks if there is an event in the queue.
         *
         * @param idx The queue index of the collection.
         * @param iStepOffset  offset from the current simulation step.
         * @return true if there is an event.
         */
        CUDA_CALLABLE bool checkAnEvent(const BGSIZE, int iStepOffset);

        /**
         * Checks if there is an event in the queue.
         *
         * @param idx   The queue index of the collection.
         * @param delay The delay descretized into time steps when the event will be triggered.
         * @param iStepOffset  offset from the current simulation step.
         * @return true if there is an event.
         */
        CUDA_CALLABLE bool checkAnEvent(const BGSIZE idx, const int delay, int iStepOffset);

        /**
         * Clears events in the queue.
         *
         * @param idx The queue index of the collection.
         */
        CUDA_CALLABLE void clearAnEvent(const BGSIZE idx);

        /**
         * Advance one simulation step.
         *
         * @param iStep        simulation step to advance.
         */
        CUDA_CALLABLE void advanceEventQueue(int iStep);

        /**
         * Register an event handler.
         *
         * @param eventHandler  Pointer to the InterClustersEventHandler.
         */
        CUDA_CALLABLE void regEventHandler(InterClustersEventHandler* eventHandler);

        /**
         * Writes the queue data to the stream.
         *
         * output  stream to print out to.
         */
        void serialize(ostream& output);

        /**
         * Sets the data for the queue to input's data.
         *
         * input istream to read from.
         */
        void deserialize(istream& input);

    public:

        //! Pointer to the collection of event queue.
        BGQUEUE_ELEMENT* m_queueEvent;

        //! The cluster ID of cluster that owns the event queue.
        CLUSTER_INDEX_TYPE m_clusterID;

        //! The number of event queue.
        BGSIZE m_nMaxEvent;

        //! The index indicating the current time slot in the delayed queue.
        uint32_t m_idxQueue;

    private:

        //! Pointer to the InterClustersEventHandler.
        InterClustersEventHandler* m_eventHandler;

#if defined(USE_GPU)

    public:

        /**
         * Create an EventQueue class object in device
         *
         * @param pEventQueue_d    Device memory address to save the pointer of created EventQueue object.
         */
        __host__ void createEventQueueInDevice(EventQueue** pEventQueue_d);

        /**
         * Delete an EventQueue class object in device
         *
         * @param pEventQueue_d    Pointer to the EventQueue object to be deleted in device.
         */
        static void deleteEventQueueInDevice(EventQueue* pEventQueue_d);

        /**
         * Copy EventQueue data from host to device
         *
         * @param pEventQueue   Pointer to the EventQueue object in device.
         */
        __host__ void copyEventQueueHostToDevice(EventQueue* pEventQueue);

        /**
         * Copy EventQueue data from device to host
         *
         * @param pEventQueue   Pointer to the EventQueue object in device.
         */
        __host__ void copyEventQueueDeviceToHost(EventQueue* pEventQueue);

    private:

        /**
         * Get index indicating the current time slot in the delayed queue in device
         *
         * @param pEventQueue   Pointer to the EventQueue object in device.
         * @param idxQueue_h    Address to the data to get.
         */
        static void getQueueIndexInDevice(EventQueue* pEventQueue, uint32_t* idxQueue_h);

        /**
         * Get pointer to the collection of event queue in device
         *
         * @param pEventQueue     Pointer to the EventQueue object in device.
         * @param pQueueEvent_h   Address to the data to get.
         */
        static void getQueueEventPointerInDevice(EventQueue* pEventQueue, BGQUEUE_ELEMENT** pQueueEvent_h);

        /**
         * Get number of events stored in the inter clusters outgoing event queue in device
         *
         * @param pEventQueue     Pointer to the EventQueue object in device.
         * @param nInterClustersOutgoingEvents_h   Address to the data to get.
         */
        static void getNInterClustersOutgoingEventsInDevice(EventQueue* pEventQueue, int* nInterClustersOutgoingEvents_h);

        /**
         * Get pointer to the inter clusters outgoing event queue in device
         *
         * @param pEventQueue     Pointer to the EventQueue object in device.
         * @param pInterClustersOutgoingEvents_h   Address to the data to get.
         */
        static void getInterClustersOutgoingEventPointerInDevice(EventQueue* pEventQueue, interClustersOutgoingEvents_t** pInterClustersOutgoingEvents_h);

        /**
         * Get pointer to the inter clusters incoming event queue in device
         *
         * @param pEventQueue     Pointer to the EventQueue object in device.
         * @param pInterClustersIncomingEvents_h   Address to the data to get.
         */
        static void getInterClustersIncomingEventPointerInDevice(EventQueue* pEventQueue, BGQUEUE_ELEMENT** pInterClustersIncomingEvents_h);

    public:

        //! The maximum number of the inter clusters outgoing event queue.
        int m_nMaxInterClustersOutgoingEvents;

        //! The number of events stored in the inter clusters outgoing event queue.
        int m_nInterClustersOutgoingEvents;

        //! Pointer to the inter clusters outgoing event queue.
        interClustersOutgoingEvents_t* m_interClustersOutgoingEvents;

        //! The maximum number of the inter clusters incoming event queue.
        int m_nMaxInterClustersIncomingEvents;

        //! The number of events stored in the inter clusters incoming event queue.
        int m_nInterClustersIncomingEvents;

        //! Pointer to the inter clusters incoming event queue.
        interClustersIncomingEvents_t* m_interClustersIncomingEvents;
#endif // USE_GPU
};

#if defined(USE_GPU)

/* -------------------------------------*\
|* # CUDA Global Functions
\* -------------------------------------*/

/**
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
__global__ void allocEventQueueDevice(EventQueue **pEventQueue, CLUSTER_INDEX_TYPE clusterID, BGSIZE nMaxEvent, BGQUEUE_ELEMENT* pQueueEvent, int nMaxInterClustersOutgoingEvents, interClustersOutgoingEvents_t* interClustersOutgoingEvents, int nMaxInterClustersIncomingEvents, interClustersIncomingEvents_t* interClustersIncomingEvents);

/**
 * Delete a EventQueue object in device memory.
 *
 * @param[in] pEventQueue          Pointer to the EventQueue object to be deleted.
 */
__global__ void deleteEventQueueDevice(EventQueue *pEventQueue);

/**
 * Get the address of event queue data in device memory
 *
 * @param[in] pEventQueue          Pointer to the EventQueue object.
 * @param[in/out] pQueueEvent      Buffer to save pointer to event queue data.
 */
__global__ void getQueueEventPointerDevice(EventQueue *pEventQueue, BGQUEUE_ELEMENT **pQueueEvent);

/**
 * Set queue index to device
 *
 * @param[in] pEventQueue          Pointer to the EventQueue object.
 * @param[in] idxQueue             Queue index.
 */
__global__ void setQueueIndexDevice(EventQueue *pEventQueue, uint32_t idxQueue);

/**
 * get queue index in device memory
 *
 * @param[in] pEventQueue          Pointer to the EventQueue object.
 * @param[in/out] idxQueue         Buffer to save Queue index.
 */
__global__ void getQueueIndexDevice(EventQueue *pEventQueue, uint32_t *idxQueue);

/**
 * get the address of inter clusters outgoing event queue data in device memory
 *
 * @param[in] pEventQueue                        Pointer to the EventQueue object.
 * @param[in/out] pInterClustersOutgoingEvents   Buffer to save pointer to inter clusters outgoing
 *                                               event queue data.
 */
__global__ void getInterClustersOutgoingEventPointerDevice(EventQueue *pEventQueue, interClustersOutgoingEvents_t **pInterClustersOutgoingEvents);

/**
 * get the address of inter clusters incoming event queue data in device memory
 *
 * @param[in] pEventQueue                        Pointer to the EventQueue object.
 * @param[in/out] pInterClustersIncomingEvents   Buffer to save pointer to inter clusters incoming
 *                                               event queue data.
 */
__global__ void getInterClustersIncomingEventPointerDevice(EventQueue *pEventQueue, interClustersIncomingEvents_t **pInterClustersIncomingEvents);

/**
 * get the number events stored in the inter clusters outgoing event queue in device memory
 *
 * @param[in] pEventQueue                         Pointer to the EventQueue object.
 * @param[in/out] pNInterClustersOutgoingEvents   Buffer to save the number.
 */
__global__ void getNInterClustersOutgoingEventsDevice(EventQueue *pEventQueue, int *pNInterClustersOutgoingEvents);

/**
 * get the number events stored in the inter clusters outgoing event queue in device memory
 *
 * @param[in] pEventQueue                         Pointer to the EventQueue object.
 * @param[in/out] pNInterClustersOutgoingEvents   Buffer to save the number.
 */
__global__ void getNInterClustersOutgoingEventsDevice(EventQueue *pEventQueue, int *pNInterClustersOutgoingEvents);

/**
 * set the number events stored in the inter clusters outgoing event queue in device memory
 *
 * @param[in] pEventQueue                         Pointer to the EventQueue object.
 * @param[in] nInterClustersOutgoingEvents        The number events in the queue..
 */
__global__ void setNInterClustersOutgoingEventsDevice(EventQueue *pEventQueue, int nInterClustersOutgoingEvents);

/**
 * set the number events stored in the inter clusters incoming event queue in device memory
 *
 * @param[in] pEventQueue                         Pointer to the EventQueue object.
 * @param[in] nInterClustersIncomingEvents        The number events in the queue..
 */
__global__ void setNInterClustersIncomingEventsDevice(EventQueue *pEventQueue, int nInterClustersIncomingEvents);

/**
 * Process inter clusters incoming events that are stored in the buffer.
 *
 * @param[in] nInterClustersIncomingEvents        The number events in the queue.
 */
__global__ void processInterClustersIncomingEventsDevice(int nInterClustersIncomingEvents, EventQueue *pEventQueue);

#endif // USE_GPU
