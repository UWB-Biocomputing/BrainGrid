/**
 *      @file InterClustersEventHandler.h
 *
 *      @brief A class to handle inter-cluster events.
 */

/**
 **
 ** @class InterClustersEventHandler InterClustersEventHandler.h "InterClustersEventHandler.h"
 **
 ** \latexonly  \subsubsection*{Implementation} \endlatexonly
 ** \htmlonly   <h3>Implementation</h3> \endhtmlonly
 **
 ** The InterClustersEventHandler class handles inter-clusters events of EventQueue.
 ** When EventQueue::addAnEvent() is called and cluserID is different from the caller's
 ** cluser, the function calls InterClustersEventHandler::addAnEvent() and the event will be added to
 ** the event queue of clusterID specified by the parameter.
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

#include "BGTypes.h"
#include <vector>

class EventQueue;

class InterClustersEventHandler
{
    public:
        //! The constructor for InterClustersEventHandler.
        InterClustersEventHandler();

        //! The destructor for InterClustersEventHandler.
        virtual ~InterClustersEventHandler();

        /**
         * Initializes the event handler.
         *
         * @param size  Size of the EventQueue vector.
         */
        void initEventHandler(const int size);

        /**
         * Register the eventQueue of the cluster specified by clusterID.
         *
         * @param clusterID   Cluster ID of the EventQueue.
         * @param eventQueue  Pointer to the EventQueue.
         */
        void addEventQueue(CLUSTER_INDEX_TYPE clusterID, EventQueue* eventQueue);

        /**
         * Add an eventin the queue of specified cluster.
         *
         * @param idx        The queue index of the collection.
         * @param clusterID  The cluster ID where the event to be added.
         * @param iStepOffset  offset from the current simulation step.
         */
        void addAnEvent(const BGSIZE idx, const CLUSTER_INDEX_TYPE clusterID, int iStepOffset);

    private:
        //! Vector to store pointers to each cluster's EventQueue.
        std::vector<EventQueue *> *m_vtEventQueue; 
};
