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

class ClusterInfo
{
    public:
        ClusterInfo() :
            totalClusterNeurons(0),
            pClusterSummationMap(NULL)
        {
        }

        virtual ~ClusterInfo() {}

        //! Count of neurons in the cluster
        int totalClusterNeurons;

        //! List of summation points (either host or device memory)
        BGFLOAT* pClusterSummationMap;
};
