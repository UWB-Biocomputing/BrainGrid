/**
 *      @file GPUSpikingModel.h
 *
 *      @brief Implementation of Model for the spiking neunal networks.
 */

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - **\ 
 * @authors Aaron Oziel, Sean Blackbourn 
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
 * @class GPUSpikingModel GPUSpikingModel.h "GPUSpikingModel.h"
 *
 * \latexonly  \subsubsection*{Implementation} \endlatexonly
 * \htmlonly   <h3>Implementation</h3> \endhtmlonly
 *
 * The Model class maintains and manages classes of objects that make up
 * essential components of the spiking neunal networks.
 *    -# IAllNeurons: A class to define a list of partiular type of neurons.
 *    -# IAllSynapses: A class to define a list of partiular type of synapses.
 *    -# Connections: A class to define connections of the neunal network.
 *    -# Layout: A class to define neurons' layout information in the network.
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
#include "Book.h"
#endif

const BGFLOAT SYNAPSE_STRENGTH_ADJUSTMENT = 1.0e-8;

/*-----------------------------------------------------*\
|  Inline Functions for handling performance recording
\*-----------------------------------------------------*/
#if defined(PERFORMANCE_METRICS) && defined(__CUDACC__)
extern float g_time;
extern cudaEvent_t start, stop;

inline void startTimer() {
       	cudaEventRecord( start, 0 );
};

inline void lapTime(float& t_event) {
       	cudaEventRecord( stop, 0 );
       	cudaEventSynchronize( stop );
       	cudaEventElapsedTime( &g_time, start, stop );
       	t_event += g_time;
};
#endif // PERFORMANCE_METRICS

class AllSpikingSynapses;

class GPUSpikingModel : public Model  {
	friend class GpuSInputPoisson;

public:
	GPUSpikingModel(Connections *conns, IAllNeurons *neurons, IAllSynapses *synapses, Layout *layout);
	virtual ~GPUSpikingModel();
 
	virtual void setupSim(SimulationInfo *sim_info, IRecorder* simRecorder);
	virtual void cleanupSim(SimulationInfo *sim_info);
        virtual void loadMemory(istream& input, const SimulationInfo *sim_info);
	virtual void advance(const SimulationInfo *sim_info);
	virtual void updateConnections(const SimulationInfo *sim_info);

protected:
        /**
        * Allocates  and initializes memories on CUDA device.
        *
        * @param[out] allNeuronsDevice          Memory loation of the pointer to the neurons list on device memory.
        * @param[out] allSynapsesDevice         Memory loation of the pointer to the synapses list on device memory.
        * @param[in] sim_info                   Pointer to the simulation information.
        */
	void allocDeviceStruct(void** allNeuronsDevice, void** allSynapsesDevice, SimulationInfo *sim_info);

        /**
         * Copies device memories to host memories and deallocaes them.
         *
        * @param[out] allNeuronsDevice          Memory loation of the pointer to the neurons list on device memory.
        * @param[out] allSynapsesDevice         Memory loation of the pointer to the synapses list on device memory.
        * @param[in]  sim_info                  Pointer to the simulation information.
         */
	virtual void deleteDeviceStruct(void** allNeuronsDevice, void** allSynapsesDevice, SimulationInfo *sim_info);

        /**
         * Add psr of all incoming synapses to summation points.
         *
         * @param[in] sim_info                   Pointer to the simulation information.
         */
	virtual void calcSummationMap(const SimulationInfo *sim_info);

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
	AllSpikingSynapses* m_allSynapsesDevice;

	//! Neuron structure in device memory.
	AllSpikingNeurons* m_allNeuronsDevice;

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

        void updateHistory(const SimulationInfo *sim_infos, IRecorder* simRecorder);

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
        
extern __global__ void setSynapseSummationPointDevice(int num_neurons, AllSpikingNeurons* allNeuronsDevice, AllSpikingSynapses* allSynapsesDevice, int max_synapses, int width);
        
//! Calculate summation point.
extern __global__ void calcSummationMapDevice( int totalNeurons, SynapseIndexMap* synapseIndexMapDevice, AllSpikingSynapses* allSynapsesDevice );
#endif
