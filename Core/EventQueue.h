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

class EventQueue
{
    //! Type of a queue element.
    #define BGQUEUE_ELEMENT uint32_t
    //! The bit length of a queue element.
    #define LENGTH_OF_DELAYQUEUE        ( sizeof(BGQUEUE_ELEMENT) * 8 )

    public:
        //! The constructor for FClassOfCategory.
        CUDA_CALLABLE EventQueue();
        CUDA_CALLABLE virtual ~EventQueue();

        /**
         * Initializes the collection of queue.
         *
         * @param nMaxEvent The number of event queue.
         */
        CUDA_CALLABLE void initEventQueue(BGSIZE nMaxEvent);

        /**
         * Add an event in the queue.
         *
         * @param idx The queue index of the collection.
         * @param delay The delay descretized into time steps when the event will be triggered.
         */
        CUDA_CALLABLE void addAnEvent(const BGSIZE idx, const int delay);

        /**
         * Checks if there is an event in the queue.
         *
         * @param idx The queue index of the collection.
         * @return true if there is an event.
         */
        CUDA_CALLABLE bool checkAnEvent(const BGSIZE idx);

        /**
         * Clears events in the queue.
         *
         * @param idx The queue index of the collection.
         */
        CUDA_CALLABLE void clearAnEvent(const BGSIZE idx);

        /**
         * Advance one simulation step.
         *
         */
        CUDA_CALLABLE void advanceEventQueue();

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

        //! The number of event queue.
        BGSIZE m_nMaxEvent;

        //! Pointer to the collection of event queue.
        BGQUEUE_ELEMENT* m_queueEvent;

        //! The index indicating the current time slot in the delayed queue.
        uint32_t m_idxQueue;
};
