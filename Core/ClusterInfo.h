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
#include "InterClustersEventHandler.h"
#if defined(USE_GPU)
#include "curand_kernel.h"
class SynapseIndexMap;
class AllDSSynapsesDeviceProperties;
#endif // USE_GPU

class IAllSynapses;

class ClusterInfo
{
    public:
        ClusterInfo() :
            clusterID(0),
            clusterNeuronsBegin(0),
            totalClusterNeurons(0),
            pClusterSummationMap(NULL),
            seed(0),
            eventHandler(NULL),
#if defined(USE_GPU)
            initValues_d(NULL),
            nShiftValues_d(NULL),
            allSynapsesDeviceSInput(NULL),
            synapseIndexMapDeviceSInput(NULL),
            nISIs_d(NULL),
            masks_d(NULL),            
            devStates_d(NULL),
#endif // USE_GPU
            synapsesSInput(NULL)
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

#if defined(USE_GPU) && defined(PERFORMANCE_METRICS) 
        //! All times in seconds
        float g_time;
        cudaEvent_t start, stop;
        double t_gpu_rndGeneration;
        double t_gpu_advanceNeurons;
        double t_gpu_advanceSynapses;
        double t_gpu_calcSummation;
#endif // USE_GPU && PERFORMANCE_METRICS

        //! Pointer to the multi clusters event handler
        InterClustersEventHandler* eventHandler;

#if defined(USE_GPU)
        //! Pointer to device input values for stimulus inputs (Regular).
        BGFLOAT* initValues_d;
        int * nShiftValues_d;

        //! variables for stimulus inputs (Poisson)

        //! Synapse structures in device memory.
        AllDSSynapsesDeviceProperties* allSynapsesDeviceSInput;

        //! Pointer to synapse index map in device memory.
        SynapseIndexMap* synapseIndexMapDeviceSInput;

        //! Pointer to device interval counter.
        int* nISIs_d;

        //! Pointer to device masks for stimulus input
        bool* masks_d;

        //! Memory to save global state for curand.
        curandState* devStates_d;
#endif // USE_GPU

        //! variables for stimulus input (Poisson)

        //! List of synapses for stimulus input (Poisson)
        IAllSynapses *synapsesSInput;
};
