/**
 * @file GPUSpikingCluster.h
 *
 * @brief Implementation of Cluster for the spiking neunal networks.
 *
 * @authors Aaron Oziel, Sean Blackbourn 
 *
 * Fumitaka Kawasaki (1/27/17):
 * Changed from Model to Cluster class.
 *
 * Fumitaka Kawasaki (5/3/14):
 * All functions were completed and working. 
 *
 * Aaron Wrote (2/3/14):
 * This file is extremely out of date and will be need to be updated to
 * reflect changes made to the corresponding .cu file. Functions will need
 * to be added/removed where necessary and the Model super class will need
 * to be edited to reflect a more abstract Model that can be used for both
 * single-threaded and GPU implementations. 
\** - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - **/

/**
 *
 * @class GPUSpikingCluster GPUSpikingCluster.h "GPUSpikingCluster.h"
 *
 * \latexonly  \subsubsection*{Implementation} \endlatexonly
 * \htmlonly   <h3>Implementation</h3> \endhtmlonly
 *
 * A cluster is a unit of execution corresponding to a thread, a GPU device, or
 * a computing node, depending on the configuration.
 * The Cluster class maintains and manages classes of objects that make up
 * essential components of the spiking neunal network.
 *    -# IAllNeurons: A class to define a list of partiular type of neurons.
 *    -# IAllSynapses: A class to define a list of partiular type of synapses.
 *
 * \image html bg_data_layout.png
 *
 * The network is composed of 3 superimposed 2-d arrays: neurons, synapses, and
 * summation points.
 *
 * Synapses in the synapse map are located at the coordinates of the neuron
 * from which they receive output.  Each synapse stores a pointer into a
 * summation point.
 *
 * If, during an advance cycle, a neuron \f$A\f$ at coordinates \f$x,y\f$ fires, every synapse
 * which receives output is notified of the spike. Those synapses then hold
 * the spike until their delay period is completed.  At a later advance cycle, once the delay
 * period has been completed, the synapses apply their PSRs (Post-Synaptic-Response) to
 * the summation points.
 * Finally, on the next advance cycle, each neuron \f$B\f$ adds the value stored
 * in their corresponding summation points to their \f$V_m\f$ and resets the summation points to
 * zero.
 *
 * The model runs on multi-threaded on a GPU.
 *
 * \latexonly  \subsubsection*{Credits} \endlatexonly
 * \htmlonly   <h3>Credits</h3> \endhtmlonly
 *
 * Some models in this simulator is a rewrite of CSIM (2006) and other
 * work (Stiber and Kawasaki (2007?))
 *
 */

#pragma once
#include "Model.h"
#include "AllSpikingNeurons.h"
#include "AllSpikingSynapses.h"
#ifdef __CUDACC__
#include <helper_cuda.h>
#endif

const BGFLOAT SYNAPSE_STRENGTH_ADJUSTMENT = 1.0e-8;

/*-----------------------------------------------------*\
|  Inline Functions for handling performance recording
\*-----------------------------------------------------*/
#if defined(PERFORMANCE_METRICS) && defined(__CUDACC__)
inline void cudaStartTimer(const ClusterInfo *clr_info) {
       	cudaEventRecord( clr_info->start, 0 );
};

//*! Increment elapsed time in seconds
inline void cudaLapTime(ClusterInfo *clr_info, double& t_event) {
       	cudaEventRecord( clr_info->stop, 0 );
       	cudaEventSynchronize( clr_info->stop );
       	cudaEventElapsedTime( &clr_info->g_time, clr_info->start, clr_info->stop );
	// The CUDA functions return time in milliseconds
       	t_event += clr_info->g_time/1000.0;
};
#endif // PERFORMANCE_METRICS

class AllSpikingSynapses;

class GPUSpikingCluster : public Cluster  {
	friend class GpuSInputPoisson;

public:
	GPUSpikingCluster(IAllNeurons *neurons, IAllSynapses *synapses);
	virtual ~GPUSpikingCluster();
 
        /**
         * Set up model state, if anym for a specific simulation run.
         *
         * @param sim_info - parameters defining the simulation to be run with the given collection of neurons.
         * @param clr_info - parameters defining the cluster to be run with the given collection of neurons.
         */
	virtual void setupCluster(SimulationInfo *sim_info, Layout *layout, ClusterInfo *clr_info);

        /**
         * Performs any finalization tasks on network following a simulation.
         *
         * @param sim_info - parameters defining the simulation to be run with the given collection of neurons.
         * @param clr_info - parameters defining the cluster to be run with the given collection of neurons.
         */
	virtual void cleanupCluster(SimulationInfo *sim_info, ClusterInfo *clr_info);

        /**
         *  Loads the simulation based on istream input.
         *
         *  @param  input       istream to read from.
         *  @param  sim_info    used as a reference to set info for neurons and synapses.
         *  @param  clr_info    used as a reference to set info for neurons and synapses.
         */
        virtual void deserialize(istream& input, const SimulationInfo *sim_info, const ClusterInfo *clr_info);

#if defined(VALIDATION)
        /**
         * Generates random numbers.
         *
         * @param sim_info - parameters defining the simulation to be run with the given collection of neurons.
         * @param clr_info - parameters defining the cluster to be run with the given collection of neurons.
         */
        virtual void genRandNumbers(const SimulationInfo *sim_info, ClusterInfo *clr_info);
#endif // VALIDATION

        /**
         * Advances neurons network state of the cluster one simulation step.
         *
         * @param sim_info - parameters defining the simulation to be run with
         *                   the given collection of neurons.
         * @param clr_info   ClusterInfo to refer.
         * @param iStepOffset  Offset from the current simulation step.
         */
        virtual void advanceNeurons(const SimulationInfo *sim_info, ClusterInfo *clr_info, int iStepOffset);

        /**
         * Process outgoing spiking data between clusters.
         *
         * @param  clr_info  ClusterInfo to refer.
         */
        virtual void processInterClustesOutgoingSpikes(ClusterInfo *clr_info);

        /**
         * Process incomng spiking data between clusters.
         *
         * @param  clr_info  ClusterInfo to refer.
         */
        virtual void processInterClustesIncomingSpikes(ClusterInfo *clr_info);

        /**
         * Advances synapses network state of the cluster one simulation step.
         *
         * @param sim_info - parameters defining the simulation to be run with
         *                   the given collection of neurons.
         * @param  clr_info  ClusterInfo to refer.
         * @param iStepOffset  Offset from the current simulation step.
         */
        virtual void advanceSynapses(const SimulationInfo *sim_info, ClusterInfo *clr_info, int iStepOffset);

        /**
         * Advances synapses spike event queue state of the cluster one simulation step.
         *
         * @param sim_info - parameters defining the simulation to be run with
         *                   the given collection of neurons.
         * @param clr_info - parameters defining the simulation to be run with
         *                   the given collection of neurons.
         * @param iStep    - simulation steps to advance.
         */
        virtual void advanceSpikeQueue(const SimulationInfo *sim_info, const ClusterInfo *clr_info, int iStep);

protected:
        /**
        * Allocates  and initializes memories on CUDA device.
        *
        * @param[out] allNeuronsDevice          Memory loation of the pointer to the neurons list on device memory.
        * @param[out] allSynapsesDevice         Memory loation of the pointer to the synapses list on device memory.
        * @param[in] sim_info                   Pointer to the simulation information.
        * @param[in] clr_info                   Pointer to the cluster information.
        */
	void allocDeviceStruct(void** allNeuronsDevice, void** allSynapsesDevice, SimulationInfo *sim_info, ClusterInfo *clr_info);

        /**
         * Copies device memories to host memories and deallocaes them.
         *
         * @param[out] allNeuronsDevice          Memory loation of the pointer to the neurons list on device memory.
         * @param[out] allSynapsesDevice         Memory loation of the pointer to the synapses list on device memory.
         * @param[in]  sim_info                  Pointer to the simulation information.
         * @param[in]  clr_info                  Pointer to the cluster information.
         */
	virtual void deleteDeviceStruct(void** allNeuronsDevice, void** allSynapsesDevice, SimulationInfo *sim_info, ClusterInfo *clr_info);

        /**
         * Add psr of all incoming synapses to summation points.
         * (parallel reduction base summation)
         *
         * @param[in] sim_info                   Pointer to the simulation information.
         * @param[in] clr_info                   Pointer to the cluster information.
         */
	virtual void calcSummationMap_1(const SimulationInfo *sim_info, const ClusterInfo *clr_info);

        /**
         * Add psr of all incoming synapses to summation points.
         * (sequential addtion base summation)
         *
         * @param[in] sim_info                   Pointer to the simulation information.
         * @param[in] clr_info                   Pointer to the cluster information.
         */
	virtual void calcSummationMap_2(const SimulationInfo *sim_info, const ClusterInfo *clr_info);

	/* ------------------*\
	|* # Helper Functions
	\* ------------------*/

	//! Pointer to device random noise array.
	float* randNoise_d;

	/*----------------------------------------------*\
	|  Member variables
	\*----------------------------------------------*/

public:
	//! Pointer to synapse index map in device memory.
	SynapseIndexMap* m_synapseIndexMapDevice;

	//! Synapse structures in device memory.
	AllSpikingSynapsesProperties* m_allSynapsesProperties;

	//! Neuron structure in device memory.
	AllSpikingNeuronsProps* m_allNeuronsProps;

        /**
         *  Copy SynapseIndexMap in host memory to SynapseIndexMap in device memory.
         *
         *  @param  clr_info    ClusterInfo to refer from.
         */
	void copySynapseIndexMapHostToDevice(const ClusterInfo *clr_info);

private: 
	/* ------------------*\
	|* # Helper Functions
	\* ------------------*/

        /**
         *  Allocate device memory for synapse inverse map.
         *  @param  count       The number of neurons.
         */
	void allocSynapseImap( int count );


        /**
         *  Deallocate device memory for synapse inverse map.
         */
	void deleteSynapseImap( );

	// # Load Memory
	// -------------

	// # Create All Neurons
	// --------------------

	// # Advance Network/Model
	// -----------------------

	// # Update Connections
	// --------------------

	// TODO
	void eraseSynapse(IAllSynapses &synapses, const int neuron_index, const int synapse_index);
	// TODO
	void addSynapse(IAllSynapses &synapses, synapseType type, const int src_neuron, const int dest_neuron, Coordinate &source, Coordinate &dest, BGFLOAT *sum_point, BGFLOAT deltaT);
	// TODO
	void createSynapse(IAllSynapses &synapses, const int neuron_index, const int synapse_index, Coordinate source, Coordinate dest, BGFLOAT* sp, BGFLOAT deltaT, synapseType type);

	/*----------------------------------------------*\
	|  Generic Functions for handling synapse types
	\*----------------------------------------------*/

#if defined(VALIDATION)
private:
        //! Buffer to save random numbers in host memory. 
        static float* m_randNoiseHost;
#endif // VALIDATION 
};

#if defined(__CUDACC__)
extern "C" {
void normalMTGPU(float * randNoise_d);
void initMTGPU(unsigned int seed, unsigned int blocks, unsigned int threads, unsigned int nPerRng, unsigned int mt_rng_count); 
}       
        
//! Calculate summation point (use parallel reduction method).
extern __global__ void calcSummationMapDevice_1(BGSIZE numTotalSynapses, AllSpikingNeuronsProps* allNeuronsProps, SynapseIndexMap* synapseIndexMapDevice, AllSpikingSynapsesProperties* allSynapsesProperties, int maxSynapsesPerNeuron, int clusterNeuronsBegin);

//! Helper kernel function for calcSummationMapDevice.
extern __global__ void reduceSummationMapKernel(BGSIZE numTotalSynapses, unsigned int s, AllSpikingSynapsesProperties* allSynapsesProperties, AllSpikingNeuronsProps* allNeuronsProps, BGSIZE* indexMap, BGSIZE* synapseCount, BGSIZE* synapseBegin, int clusterNeuronsBegin);

//! Calculate summation point.
extern __global__ void calcSummationMapDevice_2(int totalNeurons,
		    AllSpikingNeuronsProps* __restrict__ allNeurnsProps,
		    const SynapseIndexMap* __restrict__ synapseIndexMapDevice,
                    const AllSpikingSynapsesProperties* __restrict__ allSynapsesProperties );
#endif // __CUDACC__
