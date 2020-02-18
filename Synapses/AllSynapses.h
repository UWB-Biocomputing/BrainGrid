/**
 *      @file AllSynapses.h
 *
 *      @brief A container of all synapse data
 */

/** 
 * @authors Aaron Oziel, Sean Blackbourn
 * 
 * @class AllSynapses AllSynapses.h "AllSynapses.h"
 *
 * \latexonly  \subsubsection*{Implementation} \endlatexonly
 * \htmlonly   <h3>Implementation</h3> \endhtmlonly
 *
 *  The container holds synapse parameters of all synapses. 
 *  Each kind of synapse parameter is stored in a 2D array. Each item in the first 
 *  dimention of the array corresponds with each neuron, and each item in the second
 *  dimension of the array corresponds with a synapse parameter of each synapse of the neuron. 
 *  Bacause each neuron owns different number of synapses, the number of synapses 
 *  for each neuron is stored in a 1D array, synapse_counts.
 *
 *  For CUDA implementation, we used another structure, AllSynapsesDevice, where synapse
 *  parameters are stored in 1D arrays instead of 2D arrays, so that device functions
 *  can access these data less latency. When copying a synapse parameter, P[i][j],
 *  from host to device, it is stored in P[i * max_synapses_per_neuron + j] in 
 *  AllSynapsesDevice structure.
 *
 *  The latest implementation uses the identical data struture between host and CUDA;
 *  that is, synapse parameters are stored in a 1D array, so we don't need conversion 
 *  when copying data between host and device memory.
 *
 * \latexonly  \subsubsection*{Credits} \endlatexonly
 * \htmlonly   <h3>Credits</h3> \endhtmlonly
 *
 * Some models in this simulator is a rewrite of CSIM (2006) and other
 * work (Stiber and Kawasaki (2007?))
 */

#pragma once

#include "Global.h"
#include "SimulationInfo.h"
#include "IAllSynapses.h"
#include "AllSynapsesProps.h"

#ifdef _WIN32
typedef unsigned _int8 uint8_t;
#endif

class IAllNeurons;

class AllSynapses : public IAllSynapses
{
    public:
        CUDA_CALLABLE AllSynapses();
        CUDA_CALLABLE virtual ~AllSynapses();

        /**
         *  Assignment operator: copy synapses parameters.
         *
         *  @param  r_synapses  Synapses class object to copy from.
         */
        virtual IAllSynapses &operator=(const IAllSynapses &r_synapses);

        /**
         *  Setup the internal structure of the class (allocate memories and initialize them).
         *
         *  @param  sim_info  SimulationInfo class to read information from.
         *  @param  clr_info  ClusterInfo class to read information from.
         */
        virtual void setupSynapses(SimulationInfo *sim_info, ClusterInfo *clr_info);

        /**
         *  Setup the internal structure of the class (allocate memories and initialize them).
         *
         *  @param  num_neurons   Total number of neurons in the network.
         *  @param  max_synapses  Maximum number of synapses per neuron.
         *  @param  sim_info      SimulationInfo class to read information from.
         *  @param  clr_info      ClusterInfo class to read information from.
         */
        virtual void setupSynapses(const int num_neurons, const int max_synapses, SimulationInfo *sim_info, ClusterInfo *clr_info);

        /**
         *  Cleanup the class (deallocate memories).
         */
        virtual void cleanupSynapses();

        /**
         *  Checks the number of required parameters to read.
         *
         * @return true if all required parameters were successfully read, false otherwise.
         */
        virtual bool checkNumParameters();

        /**
         *  Attempts to read parameters from a XML file.
         *
         *  @param  element TiXmlElement to examine.
         *  @return true if successful, false otherwise.
         */
        virtual bool readParameters(const TiXmlElement& element);

        /**
         *  Prints out all parameters of the neurons to ostream.
         *
         *  @param  output  ostream to send output to.
         */
        virtual void printParameters(ostream &output) const;

        /**
         *  Adds a Synapse to the model, connecting two Neurons.
         *
         *  @param  iSyn        Index of the synapse to be added.
         *  @param  type        The type of the Synapse to add.
         *  @param  src_neuron  The Neuron that sends to this Synapse (layout index).
         *  @param  dest_neuron The Neuron that receives from the Synapse (layout index).
         *  @param  sum_point   Summation point address.
         *  @param  deltaT      Inner simulation step duration
         *  @param  iNeuron     Index of the destination neuron in the cluster.
         */
        CUDA_CALLABLE virtual void addSynapse(BGSIZE &iSyn, synapseType type, const int src_neuron, const int dest_neuron, BGFLOAT *sum_point, const BGFLOAT deltaT, int iNeuron);

        /**
         *  Get the sign of the synapseType.
         *
         *  @param    type    synapseType I to I, I to E, E to I, or E to E
         *  @return   1 or -1, or 0 if error
         */
        CUDA_CALLABLE int synSign(const synapseType type);

        /**
         *  Returns the type of synapse at the given coordinates
         *
         *  @param    neuron_type_map  The neuron type map (INH, EXC).
         *  @param    src_neuron  integer that points to a Neuron in the type map as a source.
         *  @param    dest_neuron integer that points to a Neuron in the type map as a destination.
         *  @return type of the synapse.
         */
        CUDA_CALLABLE virtual synapseType synType(neuronType* neuron_type_map, const int src_neuron, const int dest_neuron);

        /**
         *  Reset time varying state vars and recompute decay.
         *
         *  @param  iSyn     Index of the synapse to set.
         *  @param  deltaT   Inner simulation step duration
         */
        CUDA_CALLABLE virtual void resetSynapse(const BGSIZE iSyn, const BGFLOAT deltaT);
        
        /**
         *  Cereal serialization and deserialization method
         *  (Serializes/deserializes SynapseProps)
         */
        template<class Archive>
        void serialize(Archive & archive);

#if defined(USE_GPU)

    public:
        /**
         *  Advance all the Synapses in the simulation.
         *  Update the state of all synapses for a time step.
         *
         *  @param  allNeuronsProps        Reference to the allNeurons struct on device memory.
         *  @param  synapseIndexMapDevice  Reference to the SynapseIndexMap on device memory.
         *  @param  sim_info               SimulationInfo class to read information from.
         *  @param  clr_info               ClusterInfo to refer from.
         *  @param  iStepOffset            Offset from the current simulation step.
         *  @param  synapsesDevice         Pointer to the Synapses object in device memory.
         *  @param  neuronsDevice          Pointer to the Neurons object in device memory.
         */
        virtual void advanceSynapses(void* allNeuronsProps, void* synapseIndexMapDevice, const SimulationInfo *sim_info, const ClusterInfo *clr_info, int iStepOffset, IAllSynapses* synapsesDevice, IAllNeurons* neuronsDevice);

#else // USE_GPU

    public:
        /**
         *  Advance all the Synapses in the simulation.
         *  Update the state of all synapses for a time step.
         *
         *  @param  sim_info         SimulationInfo class to read information from.
         *  @param  neurons          The Neuron list to search from.
         *  @param  synapseIndexMap  Pointer to the synapse index map.
         *  @param  iStepOffset      Offset from the current simulation step.
         */
        virtual void advanceSynapses(const SimulationInfo *sim_info, IAllNeurons *neurons, SynapseIndexMap *synapseIndexMap, int iStepOffset);

#endif // !USE_GPU

    public:
        /**
         *  Remove a synapse from the network.
         *
         *  @param  neuron_index   Index of a neuron to remove from.
         *  @param  iSyn           Index of a synapse to remove.
         */
        CUDA_CALLABLE virtual void eraseSynapse(const int neuron_index, const BGSIZE iSyn);

#if defined(USE_GPU)
    public:
        /**
         *  Set neurons properties.
         *
         *  @param  pAllSynapsesProps  Pointer to the neurons properties.
         */
        CUDA_CALLABLE virtual void setSynapsesProps(void *pAllSynapsesProps);

        /**
         * Delete an Synapses class object in device
         *
         * @param pAllSynapses_d    Pointer to the AllSynapses object to be deleted in device.
         */
        virtual void deleteAllSynapsesInDevice(IAllSynapses* pAllSynapses_d);

#endif // defined(USE_GPU)

    public:
        // The factor to adjust overlapping area to synapse weight.
        static constexpr BGFLOAT SYNAPSE_STRENGTH_ADJUSTMENT = 1.0e-8;

        /**
         * Pointer to the synapses property data.
         */
        class AllSynapsesProps* m_pSynapsesProps;
};

#if defined(USE_GPU)

/* -------------------------------------*\
|* # CUDA Global Functions
\* -------------------------------------*/

__global__ void deleteAllSynapsesDevice(IAllSynapses *pAllSynapses);

/*
 *  CUDA code for advancing spiking synapses.
 *  Perform updating synapses for one time step.
 *
 *  @param[in] total_synapse_counts  Number of synapses.
 *  @param  synapseIndexMapDevice    Reference to the SynapseIndexMap on device memory.
 *  @param[in] simulationStep        The current simulation step.
 *  @param[in] deltaT                Inner simulation step duration.
 *  @param[in] maxSpikes             Maximum number of spikes per neuron per epoch.
 *  @param[in] iStepOffset           Offset from the current simulation step.
 *  @param[in] synapsesDevice        Pointer to the Synapses object in device memory.
 *  @param[in] neuronsDevice         Pointer to the Neurons object in device memory.
 *  @param[in] pINeuronsProps        Pointer to the neurons properties.
 */
__global__ void advanceSynapsesDevice ( int total_synapse_counts, SynapseIndexMap* synapseIndexMapDevice, uint64_t simulationStep, int maxSpikes, const BGFLOAT deltaT, int iStepOffset, IAllSynapses* synapsesDevice, IAllNeurons* neuronsDevice, IAllNeuronsProps* pINeuronsProps );

#endif // USE_GPU

/**
 *  Cereal serialization and deserialization method
 *  (Serializes/deserializes SynapseProps)
 */
template<class Archive>
void AllSynapses::serialize(Archive & archive) {
    archive(*m_pSynapsesProps);
}

