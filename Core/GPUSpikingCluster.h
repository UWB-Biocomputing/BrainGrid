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
extern float g_time;
extern cudaEvent_t start, stop;

inline void cudaStartTimer() {
       	cudaEventRecord( start, 0 );
};

//*! Increment elapsed time in seconds
inline void cudaLapTime(double& t_event) {
       	cudaEventRecord( stop, 0 );
       	cudaEventSynchronize( stop );
       	cudaEventElapsedTime( &g_time, start, stop );
	// The CUDA functions return time in milliseconds
       	t_event += g_time/1000.0;
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

        /**
         * Advances network state one simulation step.
         *
         * @param sim_info - parameters defining the simulation to be run with the given collection of neurons.
         * @param clr_info - parameters defining the cluster to be run with the given collection of neurons.
         */
	virtual void advance(const SimulationInfo *sim_info, const ClusterInfo *clr_info);

        /**
         * Modifies connections between neurons based on current state of the network and behavior
         * over the past epoch. Should be called once every epoch.
         *
         * @param sim_info - parameters defining the simulation to be run with the given collection of neurons.
         *  @param  layout      A class to define neurons' layout information in the network.
         *  @param  conns       A class to define neurons' connections information in the network.
         *  @param  clr_info    ClusterInfo class to read information from.
         */
	virtual void updateConnections(const SimulationInfo *sim_info, Connections *conns, Layout *layout, const ClusterInfo *clr_info);

        virtual void updateHistory(const SimulationInfo *sim_infos, const ClusterInfo *clr_info);

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
         *
         * @param[in] sim_info                   Pointer to the simulation information.
         * @param[in] clr_info                   Pointer to the cluster information.
         */
	virtual void calcSummationMap(const SimulationInfo *sim_info, const ClusterInfo *clr_info);

	/* ------------------*\
	|* # Helper Functions
	\* ------------------*/

	//! Pointer to device random noise array.
	float* randNoise_d;

	//! Pointer to synapse index map in device memory.
	SynapseIndexMap* synapseIndexMapDevice;

	/*----------------------------------------------*\
	|  Member variables
	\*----------------------------------------------*/

	//! Synapse structures in device memory.
	AllSpikingSynapsesDeviceProperties* m_allSynapsesDevice;

	//! Neuron structure in device memory.
	AllSpikingNeuronsDeviceProperties* m_allNeuronsDevice;

private: 
	/* ------------------*\
	|* # Helper Functions
	\* ------------------*/
	void allocSynapseImap( int count );
	void deleteSynapseImap( );
	void copySynapseIndexMapHostToDevice(SynapseIndexMap &synapseIndexMapHost, int neuron_count);

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
};

#if defined(__CUDACC__)
extern "C" {
void normalMTGPU(float * randNoise_d);
void initMTGPU(unsigned int seed, unsigned int blocks, unsigned int threads, unsigned int nPerRng, unsigned int mt_rng_count); 
}       
        
//! Calculate summation point.
extern __global__ void calcSummationMapDevice(int totalNeurons,
		    AllSpikingNeuronsDeviceProperties* __restrict__ allNeurnsDevice,
		    const SynapseIndexMap* __restrict__ synapseIndexMapDevice,
                    const AllSpikingSynapsesDeviceProperties* __restrict__ allSynapsesDevice );
#endif
