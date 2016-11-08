/**
 *      @file AllSTDPSynapses.h
 *
 *      @brief A container of all STDP synapse data
 */

/** 
 * @authors Aaron Oziel, Sean Blackbourn
 * 
 * @class AllSTDPSynapses AllSTDPSynapses.h "AllSTDPSynapses.h"
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
 *  For CUDA implementation, we used another structure, AllDSSynapsesDevice, where synapse
 *  parameters are stored in 1D arrays instead of 2D arrays, so that device functions
 *  can access these data less latency. When copying a synapse parameter, P[i][j],
 *  from host to device, it is stored in P[i * max_synapses_per_neuron + j] in 
 *  AllDSSynapsesDevice structure.
 *
 *  The latest implementation uses the identical data struture between host and CUDA;
 *  that is, synapse parameters are stored in a 1D array, so we don't need conversion
 *  when copying data between host and device memory.
 */

/** 
 *  Implements the basic weight update for a time difference \f$Delta =
 *  t_{post}-t_{pre}\f$ with presynaptic spike at time \f$t_{pre}\f$ and
 *  postsynaptic spike at time \f$t_{post}\f$. Then, the weight update is given by
 *  \f$dw =  Apos * exp(-Delta/taupos)\f$ for \f$Delta > 0\f$, and \f$dw =  Aneg *
 *  exp(-Delta/tauneg)\f$ for \f$Delta < 0\f$. (set \f$useFroemkeDanSTDP=0\f$ and
 *  \f$mupos=muneg=0\f$ for this basic update rule).
 *  
 *  It is also possible to use an
 *  extended multiplicative update by changing mupos and muneg. Then \f$dw =
 *  (Wex-W)^{mupos} * Apos * exp(-Delta/taupos)\f$ for \f$Delta > 0\f$ and \f$dw =
 *  W^{mupos} * Aneg * exp(Delta/tauneg)\f$ for \f$Delta < 0\f$. (see Guetig,
 *  Aharonov, Rotter and Sompolinsky (2003). Learning input correlations through
 *  non-linear asymmetric Hebbian plasticity. Journal of Neuroscience 23.
 *  pp.3697-3714.)
 *      
 *  Set \f$useFroemkeDanSTDP=1\f$ (this is the default value) and
 *  use \f$tauspost\f$ and \f$tauspre\f$ for the rule given in Froemke and Dan
 *  (2002). Spike-timing-dependent synaptic modification induced by natural spike
 *  trains. Nature 416 (3/2002). 
 *
 * \latexonly  \subsubsection*{Credits} \endlatexonly
 * \htmlonly   <h3>Credits</h3> \endhtmlonly
 *
 * Some models in this simulator is a rewrite of CSIM (2006) and other
 * work (Stiber and Kawasaki (2007?))
 */

#pragma once

#include "AllSpikingSynapses.h"

struct AllSTDPSynapsesDeviceProperties;

class AllSTDPSynapses : public AllSpikingSynapses
{
    public:
        AllSTDPSynapses();
        AllSTDPSynapses(const int num_neurons, const int max_synapses);
        virtual ~AllSTDPSynapses();

        static IAllSynapses* Create() { return new AllSTDPSynapses(); }
 
        /**
         *  Setup the internal structure of the class (allocate memories and initialize them).
         *
         *  @param  sim_info  SimulationInfo class to read information from.
         */
        virtual void setupSynapses(SimulationInfo *sim_info);

        /**
         *  Cleanup the class (deallocate memories).
         */
        virtual void cleanupSynapses();

        /**
         *  Reset time varying state vars and recompute decay.
         *
         *  @param  iSyn     Index of the synapse to set.
         *  @param  deltaT   Inner simulation step duration
         */
        virtual void resetSynapse(const BGSIZE iSyn, const BGFLOAT deltaT);

        /**
         *  Check if the back propagation (notify a spike event to the pre neuron)
         *  is allowed in the synapse class.
         *
         *  @retrun true if the back propagation is allowed.
         */
        virtual bool allowBackPropagation();

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
        virtual void createSynapse(const BGSIZE iSyn, int source_index, int dest_index, BGFLOAT* sp, const BGFLOAT deltaT, synapseType type);

    protected:
        /**
         *  Setup the internal structure of the class (allocate memories and initialize them).
         *
         *  @param  num_neurons   Total number of neurons in the network.
         *  @param  max_synapses  Maximum number of synapses per neuron.
         */
        virtual void setupSynapses(const int num_neurons, const int max_synapses);

        /**
         *  Sets the data for Synapse to input's data.
         *
         *  @param  input  istream to read from.
         *  @param  iSyn   Index of the synapse to set.
         */
        virtual void readSynapse(istream &input, const BGSIZE iSyn);

        /**
         *  Write the synapse data to the stream.
         *
         *  @param  output  stream to print out to.
         *  @param  iSyn    Index of the synapse to print out.
         */
        virtual void writeSynapse(ostream& output, const BGSIZE iSyn) const;

        /**
         *  Initializes the queues for the Synapse.
         *
         *  @param  iSyn   index of the synapse to set.
         */
        virtual void initSpikeQueue(const BGSIZE iSyn);

#if defined(USE_GPU)
    public:
        /**
         *  Allocate GPU memories to store all synapses' states,
         *  and copy them from host to GPU memory.
         *
         *  @param  allSynapsesDevice  Reference to the allSynapses struct on device memory.
         *  @param  sim_info           SimulationInfo to refer from.
         */
        virtual void allocSynapseDeviceStruct( void** allSynapsesDevice, const SimulationInfo *sim_info );

        /**
         *  Allocate GPU memories to store all synapses' states,
         *  and copy them from host to GPU memory.
         *
         *  @param  allSynapsesDevice     Reference to the allSynapses struct on device memory.
         *  @param  num_neurons           Number of neurons.
         *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
         */
        virtual void allocSynapseDeviceStruct( void** allSynapsesDevice, int num_neurons, int maxSynapsesPerNeuron );

        /**
         *  Delete GPU memories.
         *
         *  @param  allSynapsesDevice  Reference to the allSynapses struct on device memory.
         *  @param  sim_info           SimulationInfo to refer from.
         */
        virtual void deleteSynapseDeviceStruct( void* allSynapsesDevice );

        /**
         *  Copy all synapses' data from host to device.
         *
         *  @param  allSynapsesDevice  Reference to the allSynapses struct on device memory.
         *  @param  sim_info           SimulationInfo to refer from.
         */
        virtual void copySynapseHostToDevice( void* allSynapsesDevice, const SimulationInfo *sim_info );

        /**
         *  Copy all synapses' data from host to device.
         *
         *  @param  allSynapsesDevice  Reference to the allSynapses struct on device memory.
         *  @param  num_neurons           Number of neurons.
         *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
         */
        virtual void copySynapseHostToDevice( void* allSynapsesDevice, int num_neurons, int maxSynapsesPerNeuron );

        /**
         *  Copy all synapses' data from device to host.
         *
         *  @param  allSynapsesDevice  Reference to the allSynapses struct on device memory.
         *  @param  sim_info           SimulationInfo to refer from.
         */
        virtual void copySynapseDeviceToHost( void* allSynapsesDevice, const SimulationInfo *sim_info );

        /**
         *  Advance all the Synapses in the simulation.
         *  Update the state of all synapses for a time step.
         *
         *  @param  allSynapsesDevice      Reference to the allSynapses struct on device memory.
         *  @param  allNeuronsDevice       Reference to the allNeurons struct on device memory.
         *  @param  synapseIndexMapDevice  Reference to the SynapseIndexMap on device memory.
         *  @param  sim_info               SimulationInfo class to read information from.
         */
        virtual void advanceSynapses(void* allSynapsesDevice, void* allNeuronsDevice, void* synapseIndexMapDevice, const SimulationInfo *sim_info);

        /**
         *  Set synapse class ID defined by enumClassSynapses for the caller's Synapse class.
         *  The class ID will be set to classSynapses_d in device memory,
         *  and the classSynapses_d will be referred to call a device function for the
         *  particular synapse class.         
         *  Because we cannot use virtual function (Polymorphism) in device functions,
         *  we use this scheme.         
         *  Note: we used to use a function pointer; however, it caused the growth_cuda crash
         *  (see issue#137).
         */
        virtual void setSynapseClassID();

    protected:
        /**
         *  Allocate GPU memories to store all synapses' states,
         *  and copy them from host to GPU memory.
         *  (Helper function of allocSynapseDeviceStruct)
         *
         *  @param  allSynapsesDevice  Reference to the allSynapses struct on device memory.
         *  @param  num_neurons           Number of neurons.
         *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
         */
        void allocDeviceStruct( AllSTDPSynapsesDeviceProperties &allSynapses, int num_neurons, int maxSynapsesPerNeuron );

        /**
         *  Delete GPU memories.
         *  (Helper function of deleteSynapseDeviceStruct)
         *
         *  @param  allSynapsesDevice  Reference to the allSynapses struct on device memory.
         */
        void deleteDeviceStruct( AllSTDPSynapsesDeviceProperties& allSynapses );

        /**
         *  Copy all synapses' data from host to device.
         *  (Helper function of copySynapseHostToDevice)
         *
         *  @param  allSynapsesDevice  Reference to the allSynapses struct on device memory.
         *  @param  num_neurons           Number of neurons.
         *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
         */
        void copyHostToDevice( void* allSynapsesDevice, AllSTDPSynapsesDeviceProperties& allSynapses, int num_neurons, int maxSynapsesPerNeuron );

        /**
         *  Copy all synapses' data from device to host.
         *  (Helper function of copySynapseDeviceToHost)
         *
         *  @param  allSynapsesDevice  Reference to the allSynapses struct on device memory.
         *  @param  num_neurons           Number of neurons.
         *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
         */
        void copyDeviceToHost( AllSTDPSynapsesDeviceProperties& allSynapses, const SimulationInfo *sim_info );
#else // !defined(USE_GPU)
    public:
        /**
         *  Advance one specific Synapse.
         *  Update the state of synapse for a time step
         *
         *  @param  iSyn      Index of the Synapse to connect to.
         *  @param  sim_info  SimulationInfo class to read information from.
         *  @param  neurons   The Neuron list to search from.
         */
        virtual void advanceSynapse(const BGSIZE iSyn, const SimulationInfo *sim_info, IAllNeurons *neurons);

        /**
         *  Prepares Synapse for a spike hit (for back propagation).
         *
         *  @param  iSyn   Index of the Synapse to connect to.
         */
        virtual void postSpikeHit(const BGSIZE iSyn);

    protected:
        /**
         *  Checks if there is an input spike in the queue (for back propagation).
         *
         *  @param  iSyn   Index of the Synapse to connect to.
         *  @return true if there is an input spike event.
         */
        bool isSpikeQueuePost(const BGSIZE iSyn);

    private:
        /**
         *  Adjust synapse weight according to the Spike-timing-dependent synaptic modification
         *  induced by natural spike trains
         *
         *  @param  iSyn        Index of the synapse to set.
         *  @param  delta       Pre/post synaptic spike interval.
         *  @param  epost       Params for the rule given in Froemke and Dan (2002).
         *  @param  epre        Params for the rule given in Froemke and Dan (2002).
         */
        void stdpLearning(const BGSIZE iSyn,double delta, double epost, double epre);

#endif
    public:
        /**
         *  The synaptic transmission delay (delay of dendritic backpropagating spike), 
         *  descretized into time steps.
         */
        int *total_delayPost;

        /**
         *  Pointer to the delayed queue
         */
        uint32_t *delayQueuePost;

        /**
         *  The index indicating the current time slot in the delayed queue.
         */
        int *delayIdxPost;

        /**
         *  Length of the delayed queue.
         */
        int *ldelayQueuePost;

        /**
         *  Used for extended rule by Froemke and Dan. See Froemke and Dan (2002). 
         *  Spike-timing-dependent synaptic modification induced by natural spike trains. 
         *  Nature 416 (3/2002).
         */
        BGFLOAT *tauspost;

        /**
         *  sed for extended rule by Froemke and Dan.
         */
        BGFLOAT *tauspre;

        /**
         *  Timeconstant of exponential decay of positive learning window for STDP.
         */
        BGFLOAT *taupos;

        /**
         *  Timeconstant of exponential decay of negative learning window for STDP.
         */
        BGFLOAT *tauneg;

        /**
         *  No learning is performed if \f$|Delta| = |t_{post}-t_{pre}| < STDPgap\f$
         */
        BGFLOAT *STDPgap;

        /**
         *  The maximal/minimal weight of the synapse [readwrite; units=;]
         */
        BGFLOAT *Wex;

        /**
         *  Defines the peak of the negative exponential learning window.
         */
        BGFLOAT *Aneg;

        /**
         *  Defines the peak of the positive exponential learning window.
         */
        BGFLOAT *Apos;

        /**
         *  Extended multiplicative positive update: 
         *  \f$dw = (Wex-W)^{mupos} * Apos * exp(-Delta/taupos)\f$. 
         *  Set to 0 for basic update. See Guetig, Aharonov, Rotter and Sompolinsky (2003). 
         *  Learning input correlations through non-linear asymmetric Hebbian plasticity. 
         *  Journal of Neuroscience 23. pp.3697-3714.
         */
        BGFLOAT *mupos;

        /**
         *  Extended multiplicative negative update: 
         *  \f$dw = W^{mupos} * Aneg * exp(Delta/tauneg)\f$. Set to 0 for basic update.
         */
        BGFLOAT *muneg;
  
        /**
         *  True if use the rule given in Froemke and Dan (2002).
         */
        bool *useFroemkeDanSTDP;
};

#if defined(USE_GPU)
struct AllSTDPSynapsesDeviceProperties : public AllSpikingSynapsesDeviceProperties
{
        /**
         *  The synaptic transmission delay (delay of dendritic backpropagating spike), 
         *  descretized into time steps.
         */
        int *total_delayPost;

        /**
         *  Pointer to the delayed queue
         */
        uint32_t *delayQueuePost;

        /**
         *  The index indicating the current time slot in the delayed queue.
         */
        int *delayIdxPost;

        /**
         *  Length of the delayed queue.
         */
        int *ldelayQueuePost;

        /**
         *  Used for extended rule by Froemke and Dan. See Froemke and Dan (2002). 
         *  Spike-timing-dependent synaptic modification induced by natural spike trains. 
         *  Nature 416 (3/2002).
         */
        BGFLOAT *tauspost;

        /**
         *  sed for extended rule by Froemke and Dan.
         */
        BGFLOAT *tauspre;

        /**
         *  Timeconstant of exponential decay of positive learning window for STDP.
         */
        BGFLOAT *taupos;

        /**
         *  Timeconstant of exponential decay of negative learning window for STDP.
         */
        BGFLOAT *tauneg;

        /**
         *  No learning is performed if \f$|Delta| = |t_{post}-t_{pre}| < STDPgap\f$
         */
        BGFLOAT *STDPgap;

        /**
         *  The maximal/minimal weight of the synapse [readwrite; units=;]
         */
        BGFLOAT *Wex;

        /**
         *  Defines the peak of the negative exponential learning window.
         */
        BGFLOAT *Aneg;

        /**
         *  Defines the peak of the positive exponential learning window.
         */
        BGFLOAT *Apos;

        /**
         *  Extended multiplicative positive update: 
         *  \f$dw = (Wex-W)^{mupos} * Apos * exp(-Delta/taupos)\f$. 
         *  Set to 0 for basic update. See Guetig, Aharonov, Rotter and Sompolinsky (2003). 
         *  Learning input correlations through non-linear asymmetric Hebbian plasticity. 
         *  Journal of Neuroscience 23. pp.3697-3714.
         */
        BGFLOAT *mupos;

        /**
         *  Extended multiplicative negative update: 
         *  \f$dw = W^{mupos} * Aneg * exp(Delta/tauneg)\f$. Set to 0 for basic update.
         */
        BGFLOAT *muneg;
  
        /**
         *  True if use the rule given in Froemke and Dan (2002).
         */
        bool *useFroemkeDanSTDP;
};
#endif // defined(USE_GPU)
