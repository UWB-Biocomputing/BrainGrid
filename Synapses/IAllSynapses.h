/**
 *      @file IAllSynapses.h
 *
 *      @brief An interface for synapse classes.
 */

#pragma once

#include "Global.h"
#include "SimulationInfo.h"
#include "ClusterInfo.h"
#include "IAllSynapsesProps.h"
#include "IAllNeuronsProps.h"

class IAllNeurons;
class IAllSynapses;
class SynapseIndexMap;

typedef void (*fpCreateSynapse_t)(void*, const int, const int, int, int, BGFLOAT*, const BGFLOAT, synapseType);

// enumerate all non-abstract synapse classes.
enum enumClassSynapses {classAllSpikingSynapses, classAllDSSynapses, classAllSTDPSynapses, classAllDynamicSTDPSynapses, undefClassSynapses};

class IAllSynapses
{
    public:
        CUDA_CALLABLE virtual ~IAllSynapses() {};

#if defined(BOOST_PYTHON)
        /*
         *  This function is called when Python variable that stores
         *  the synapses class object is destroyed.
         */
        virtual void destroy() = 0;
#endif // BOOST_PYTHON

        /**
         *  Create and setup synapses properties.
         */
        virtual void createSynapsesProps() = 0;

        /**
         *  Assignment operator: copy synapses parameters.
         *
         *  @param  r_synapses  Synapses class object to copy from.
         */
        virtual IAllSynapses &operator=(const IAllSynapses &r_synapses) = 0;

        /**
         *  Setup the internal structure of the class (allocate memories and initialize them).
         *
         *  @param  sim_info  SimulationInfo class to read information from.
         *  @param  clr_info  ClusterInfo class to read information from.
         */
        virtual void setupSynapses(SimulationInfo *sim_info, ClusterInfo *clr_info) = 0;

        /**
         *  Setup the internal structure of the class (allocate memories and initialize them).
         *
         *  @param  num_neurons   Total number of neurons in the network.
         *  @param  max_synapses  Maximum number of synapses per neuron.
         *  @param  sim_info      SimulationInfo class to read information from.
         *  @param  clr_info      ClusterInfo class to read information from.
         */
        virtual void setupSynapses(const int num_neurons, const int max_synapses, SimulationInfo *sim_info, ClusterInfo *clr_info) = 0;

        /**
         *  Cleanup the class (deallocate memories).
         */
        virtual void cleanupSynapses() = 0;

        /**
         *  Reset time varying state vars and recompute decay.
         *
         *  @param  iSyn     Index of the synapse to set.
         *  @param  deltaT   Inner simulation step duration
         */
        CUDA_CALLABLE virtual void resetSynapse(const BGSIZE iSyn, const BGFLOAT deltaT) = 0;

        /**
         *  Checks the number of required parameters to read.
         *
         * @return true if all required parameters were successfully read, false otherwise.
         */
        virtual bool checkNumParameters() = 0;

        /**
         *  Attempts to read parameters from a XML file.
         *
         *  @param  element TiXmlElement to examine.
         *  @return true if successful, false otherwise.
         */
        virtual bool readParameters(const TiXmlElement& element) = 0;

        /**
         *  Prints out all parameters of the neurons to ostream.
         *
         *  @param  output  ostream to send output to.
         */
        virtual void printParameters(ostream &output) const = 0;

        /**
         *  Sets the data for Synapses to input's data.
         *
         *  @param  input  istream to read from.
         *  @param clr_info  ClusterInfo class to read information from.
         */
        virtual void deserialize(istream& input, IAllNeurons &neurons, const ClusterInfo *clr_info) = 0;

        /**
         *  Write the synapses data to the stream.
         *
         *  @param  output  stream to print out to.
         *  @param  clr_info  ClusterInfo class to read information from.
         */
        virtual void serialize(ostream& output, const ClusterInfo *clr_info) = 0;

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
        CUDA_CALLABLE virtual void addSynapse(BGSIZE &iSyn, synapseType type, const int src_neuron, const int dest_neuron, BGFLOAT *sum_point, const BGFLOAT deltaT, int iNeuron) = 0;

        /**
         *  Create a Synapse and connect it to the model.
         *
         *  @param  synapses    The synapse list to reference.
         *  @param  iSyn        Index of the synapse to set.
         *  @param  source      Coordinates of the source Neuron.
         *  @param  dest        Coordinates of the destination Neuron.
         *  @param  sum_point   Summation point address.
         *  @param  deltaT      Inner simulation step duration.
         *  @param  type        Type of the Synapse to create.
         */
        CUDA_CALLABLE virtual void createSynapse(const BGSIZE iSyn, int source_index, int dest_index, BGFLOAT* sp, const BGFLOAT deltaT, synapseType type) = 0;

        /**
         *  Get the sign of the synapseType.
         *
         *  @param    type    synapseType I to I, I to E, E to I, or E to E
         *  @return   1 or -1, or 0 if error
         */
        CUDA_CALLABLE virtual int synSign(const synapseType type) = 0;

        /**
         *  Returns the type of synapse at the given coordinates
         *
         *  @param    neuron_type_map  The neuron type map (INH, EXC).
         *  @param    src_neuron  integer that points to a Neuron in the type map as a source.
         *  @param    dest_neuron integer that points to a Neuron in the type map as a destination.
         *  @return type of the synapse.
         */
        CUDA_CALLABLE virtual synapseType synType(neuronType* neuron_type_map, const int src_neuron, const int dest_neuron) = 0;

#if defined(USE_GPU)
    public:
        /**
         *  Set neurons properties.
         *
         *  @param  pAllSynapsesProps  Pointer to the neurons properties.
         */
        CUDA_CALLABLE virtual void setSynapsesProps(void *pAllSynapsesProps) = 0;

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
        virtual void advanceSynapses(void* allNeuronsProps, void* synapseIndexMapDevice, const SimulationInfo *sim_info, const ClusterInfo *clr_info, int iStepOffset, IAllSynapses* synapsesDevice, IAllNeurons* neuronsDevice) = 0;

        /**
         *  Create a AllSynapses class object in device
         *
         *  @param pAllSynapses_d      Device memory address to save the pointer of created AllSynapses object.
         *  @param pAllSynapsesProps_d  Pointer to the synapses properties in device memory.
         */
        virtual void createAllSynapsesInDevice(IAllSynapses** pAllSynapses_d, IAllSynapsesProps *pAllSynapsesProps_d) = 0;

        /**
         * Delete an Synapses class object in device
         *
         * @param pAllSynapses_d    Pointer to the AllSynapses object to be deleted in device.
         */
        virtual void deleteAllSynapsesInDevice(IAllSynapses* pAllSynapses_d) = 0;

#else // !defined(USE_GPU)
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
        virtual void advanceSynapses(const SimulationInfo *sim_info, IAllNeurons *neurons, SynapseIndexMap *synapseIndexMap, int iStepOffset) = 0;
#endif // defined(USE_GPU)

        /**
         *  Remove a synapse from the network.
         *
         *  @param  neuron_index   Index of a neuron to remove from.
         *  @param  iSyn           Index of a synapse to remove.
         */
        CUDA_CALLABLE virtual void eraseSynapse(const int neuron_index, const BGSIZE iSyn) = 0;

    public:
        /**
         *  Advance one specific Synapse.
         *
         *  @param  iSyn             Index of the Synapse to connect to.
         *  @param  deltaT           Inner simulation step duration.
         *  @param  neurons          The Neuron list to search from.
         *  @param  simulationStep   The current simulation step.
         *  @param  iStepOffset      Offset from the current simulation step.
         *  @param  maxSpikes        Maximum number of spikes per neuron per epoch.
         *  @param  pINeuronsProps   Pointer to the neurons properties.
         */
        CUDA_CALLABLE virtual void advanceSynapse(const BGSIZE iSyn, const BGFLOAT deltaT, IAllNeurons *neurons, uint64_t simulationStep, int iStepOffset, int maxSpikes, IAllNeuronsProps* pINeuronsProps) = 0;
};
