/**
 *      @file AllIZHNeurons.h
 *
 *      @brief A container of all Izhikevich neuron data
 */

/** 
 * 
 * @class AllIZHNeurons AllIZHNeurons.h "AllIZHNeurons.h"
 *
 * \latexonly  \subsubsection*{Implementation} \endlatexonly
 * \htmlonly   <h3>Implementation</h3> \endhtmlonly
 * A container of all spiking neuron data.
 * This is the base class of all spiking neuron classes.
 *
 * The class uses a data-centric structure, which utilizes a structure as the containers of
 * all neuron.
 *
 * The container holds neuron parameters of all neurons.
 * Each kind of neuron parameter is stored in a 1D array, of which length
 * is number of all neurons. Each array of a neuron parameter is pointed by a
 * corresponding member variable of the neuron parameter in the class.
 *
 * This structure was originally designed for the GPU implementation of the
 * simulator, and this refactored version of the simulator simply uses that design for
 * all other implementations as well. This is to simplify transitioning from
 * single-threaded to multi-threaded.
 *
 * The Izhikevich neuron model uses the quadratic integrate-and-fire model 
 * for ordinary differential equations of the form:
 * \f[
 *  \frac{d v}{dt} = 0.04v^2 + 5v + 140 - u + (I_{syn}(t) + I_{inject} + I_{noise})
 * \f]
 * \f[
 *  \frac{d u}{dt} = a \cdot (bv - u)
 * \f]
 * with the auxiliary after-spike resetting: if \f$v\ge30\f$ mv, then \f$v=c,u=u+d\f$.
 *
 * where \f$v\f$ and \f$u\f$ are dimensionless variable, and \f$a,b,c\f$, and \f$d\f$ are dimensioless parameters. 
 * The variable \f$v\f$ represents the membrane potential of the neuron and \f$u\f$ represents a membrane 
 * recovery variable, which accounts for the activation of \f$K^+\f$ ionic currents and 
 * inactivation of \f$Na^+\f$ ionic currents, and it provides negative feedback to \f$v\f$. 
 * \f$I_{syn}(t)\f$ is the current supplied by the synapses, \f$I_{inject}\f$ is a non-specific 
 * background current and Inoise is a Gaussian random variable with zero mean and 
 * a given variance noise (Izhikevich. 2003).
 *
 * The simple Euler method combined with the exponential Euler method is used for 
 * numerical integration. The main idea behind the exponential Euler rule is 
 * that many biological processes are governed by an exponential decay function. 
 * For an equation of the form:
 * \f[
 *  \frac{d y}{dt} = A - By
 * \f]
 * its scheme is given by:
 * \f[
 *  y(t+\Delta t) = y(t) \cdot \mathrm{e}^{-B \Delta t} + \frac{A}{B} \cdot (1 - \mathrm{e}^{-B \Delta t}) 
 * \f]
 * After appropriate substituting all variables, we obtain the Euler step:
 * \f[
 *  v(t+\Delta t)=v(t)+\Delta t \cdot (0.04v(t)^2+5v(t)+140-u(t))+R_{m} \cdot (I_{syn}(t)+I_{inject}+I_{noise}+\frac{V_{resting}}{R_{m}}) \cdot (1-\mathrm{e}^{-\frac{\Delta t}{\tau_{m}}})
 * \f]
 * \f[
 *  u(t+ \Delta t)=u(t) + \Delta t \cdot a \cdot (bv(t)-u(t))
 * \f]
 * where \f$\tau_{m}=C_{m} \cdot R_{m}\f$ is the membrane time constant, \f$R_{m}\f$ is the membrane resistance.
 *
 * \latexonly  \subsubsection*{Credits} \endlatexonly
 * \htmlonly   <h3>Credits</h3> \endhtmlonly
 *
 * This model is a rewrite of work by Stiber, Kawasaki, Allan Ortiz, and Cory Mayberry
 *
 */
#pragma once

#include "Global.h"
#include "AllIFNeurons.h"

// Class to hold all data necessary for all the Neurons.
class AllIZHNeurons : public AllIFNeurons
{
    public:
        AllIZHNeurons();
        virtual ~AllIZHNeurons();

        static AllNeurons* Create() { return new AllIZHNeurons(); }

        /**
         *  A constant (0.02, 01) describing the coupling of variable u to Vm.
         */
        BGFLOAT *Aconst;

        /**
         *  A constant controlling sensitivity of u.
         */
        BGFLOAT *Bconst;

        /**
         *  A constant controlling reset of Vm. 
         */
        BGFLOAT *Cconst;

        /**
         *  A constant controlling reset of u.
         */
        BGFLOAT *Dconst;

        /**
         *  internal variable.
         */
        BGFLOAT *u;

        /**
         *
         */ 
        BGFLOAT *C3;

        /**
         *  Setup the internal structure of the class.
         *  Allocate memories to store all neurons' state.
         *
         *  @param  sim_info  SimulationInfo class to read information from.
         */
        virtual void setupNeurons(SimulationInfo *sim_info);

        /**
         *  Cleanup the class.
         *  Deallocate memories.
         */
        virtual void cleanupNeurons();  

        /**
         *  Returns the number of required parameters to read.
         */
        virtual int numParameters();

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
         *  Creates all the Neurons and assigns initial data for them.
         *
         *  @param  sim_info    SimulationInfo class to read information from.
         *  @param  layout      Layout information of the neunal network.
         */
        virtual void createAllNeurons(SimulationInfo *sim_info, Layout *layout);

        /**
         *  Outputs state of the neuron chosen as a string.
         *
         *  @param  i   index of the neuron (in neurons) to output info from.
         *  @return the complete state of the neuron.
         */
        virtual string toString(const int i) const;

        /**
         *  Reads and sets the data for all neurons from input stream.
         *
         *  @param  input       istream to read from.
         *  @param  sim_info    used as a reference to set info for neuronss.
         */
        virtual void readNeurons(istream &input, const SimulationInfo *sim_info);

        /**
         *  Writes out the data in all neurons to output stream.
         *
         *  @param  output      stream to write out to.
         *  @param  sim_info    used as a reference to set info for neuronss.
         */
        virtual void writeNeurons(ostream& output, const SimulationInfo *sim_info) const;

#if defined(USE_GPU)
        /**
         *  Allocate GPU memories to store all neurons' states,
         *  and copy them from host to GPU memory.
         *
         *  @param  allNeuronsDevice   Reference to the allNeurons struct on device memory.
         *  @param  sim_info           SimulationInfo to refer from.
         */
        virtual void allocNeuronDeviceStruct( void** allNeuronsDevice, SimulationInfo *sim_info );

        /**
         *  Delete GPU memories.
         *
         *  @param  allNeuronsDevice   Reference to the allNeurons struct on device memory.
         *  @param  sim_info           SimulationInfo to refer from.
         */
        virtual void deleteNeuronDeviceStruct( void* allNeuronsDevice, const SimulationInfo *sim_info );

        /**
         *  Copy all neurons' data from host to device.
         *
         *  @param  allNeuronsDevice   Reference to the allNeurons struct on device memory.
         *  @param  sim_info           SimulationInfo to refer from.
         */
        virtual void copyNeuronHostToDevice( void* allNeuronsDevice, const SimulationInfo *sim_info );

        /**
         *  Copy all neurons' data from device to host.
         *
         *  @param  allNeuronsDevice   Reference to the allNeurons struct on device memory.
         *  @param  sim_info           SimulationInfo to refer from.
         */
        virtual void copyNeuronDeviceToHost( void* allNeuronsDevice, const SimulationInfo *sim_info );

        /**
         *  Copy spike history data stored in device memory to host.
         *
         *  @param  allNeuronsDevice   Reference to the allNeurons struct on device memory.
         *  @param  sim_info           SimulationInfo to refer from.
         */
        virtual void copyNeuronDeviceSpikeHistoryToHost( void* allNeuronsDevice, const SimulationInfo *sim_info );

        /**
         *  Copy spike counts data stored in device memory to host.
         *
         *  @param  allNeuronsDevice   Reference to the allNeurons struct on device memory.
         *  @param  sim_info           SimulationInfo to refer from.
         */
        virtual void copyNeuronDeviceSpikeCountsToHost( void* allNeuronsDevice, const SimulationInfo *sim_info );

        /**
         *  Clear the spike counts out of all neurons.
         *
         *  @param  allNeuronsDevice   Reference to the allNeurons struct on device memory.
         *  @param  sim_info           SimulationInfo to refer from.
         */
        virtual void clearNeuronSpikeCounts( void* allNeuronsDevice, const SimulationInfo *sim_info );

    protected:
        void allocDeviceStruct( AllIZHNeurons &allNeurons, SimulationInfo *sim_info );
        void deleteDeviceStruct( AllIZHNeurons& allNeurons, const SimulationInfo *sim_info );
        void copyHostToDevice( AllIZHNeurons& allNeurons, const SimulationInfo *sim_info );
        void copyDeviceToHost( AllIZHNeurons& allNeurons, const SimulationInfo *sim_info );
#endif

        void createNeuron(SimulationInfo *sim_info, int neuron_index, Layout *layout);
        void setNeuronDefaults(const int index);
        virtual void initNeuronConstsFromParamValues(int neuron_index, const BGFLOAT deltaT);
        void readNeuron(istream &input, const SimulationInfo *sim_info, int i);
        void writeNeuron(ostream& output, const SimulationInfo *sim_info, int i) const;

#if defined(USE_GPU)
        /**
         *  Update the state of all neurons for a time step
         *  Notify outgoing synapses if neuron has fired.
         *
         *  @param  synapses               Reference to the allSynapses struct on host memory.
         *  @param  allNeuronsDevice       Reference to the allNeurons struct on device memory.
         *  @param  allSynapsesDevice      Reference to the allSynapses struct on device memory.
         *  @param  sim_info               SimulationInfo to refer from.
         *  @param  randNoise              Reference to the random noise array.
         *  @param  synapseIndexMapDevice  Reference to the SynapseIndexMap on device memory.
         */
        virtual void advanceNeurons(AllSynapses &synapses, AllNeurons* allNeuronsDevice, AllSynapses* allSynapsesDevice, const SimulationInfo *sim_info, float* randNoise, SynapseIndexMap* synapseIndexMapDevice);
#else
        /**
         *  Helper for #advanceNeuron. Updates state of a single neuron.
         *
         *  @param  index            Index of the neuron to update.
         *  @param  sim_info         SimulationInfo class to read information from.
         */
        virtual void advanceNeuron(const int index, const SimulationInfo *sim_info);

        /**
         *  Initiates a firing of a neuron to connected neurons.
         *
         *  @param  index            Index of the neuron to fire.
         *  @param  sim_info         SimulationInfo class to read information from.
         */
        virtual void fire(const int index, const SimulationInfo *sim_info) const;
#endif

    private:
        static const BGFLOAT DEFAULT_a = 0.0035;
        static const BGFLOAT DEFAULT_b = 0.2;
        static const BGFLOAT DEFAULT_c = -50;
        static const BGFLOAT DEFAULT_d = 2;

        // TODO
        BGFLOAT m_Aconst[2];
        // TODO
        BGFLOAT m_Bconst[2];
        // TODO
        BGFLOAT m_Cconst[2];
        // TODO
        BGFLOAT m_Dconst[2];

        void freeResources();  
};
