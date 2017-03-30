/**
 *      @file ClusterInfo.h
 *
 *      @brief Header file for ClusterInfo.
 */
//! Cluster information.

/**
 ** \class ClusterInfo ClusterInfo.h "ClusterInfo.h"
 **
 ** \latexonly  \subsubsection*{Implementation} \endlatexonly
 ** \htmlonly   <h3>Implementation</h3> \endhtmlonly
 **
 ** The ClusterInfo contains all information necessary for a cluster.
 ** The cluster contains a group of neurons and synapses that are
 ** regarded as an execution unit. A simulation will include multiple
 ** clusters and each cluster runs concurrently. A cluster will be
 ** assigned to a thread, a GPU, or a computer node depending on a configuration. 
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

#include "Global.h"
#include "EventHandler.h"

class ClusterInfo
{
    public:
        ClusterInfo() :
            clusterID(0),
            clusterNeuronsBegin(0),
            totalClusterNeurons(0),
            pClusterSummationMap(NULL),
            seed(0),
            eventHandler(NULL)
        {
        }

        virtual ~ClusterInfo() {}

        //! The cluster ID
        CLUSTER_INDEX_TYPE clusterID;

        //! The beginning neuron index in the cluster
        int clusterNeuronsBegin;    

        //! Count of neurons in the cluster
        int totalClusterNeurons;

        //! List of summation points (either host or device memory)
        BGFLOAT* pClusterSummationMap;

        //! Seed used for the simulation random SINGLE THREADED
        long seed;

#if defined(USE_GPU)
        //! CUDA device ID
        int deviceId;
#endif // USE_GPU

#if !defined(USE_GPU)
        //! A normalized random number generator
        Norm* normRand;
#endif // !USE_GPU

        //! Pointer to the multi clusters event handler
        EventHandler* eventHandler;
};
