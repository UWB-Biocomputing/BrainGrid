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
 * numerical integration. 
 * 
 * One step of the simple Euler method from \f$y(t)\f$ to \f$y(t + \Delta t)\f$ is:
 *  \f$y(t + \Delta t) = y(t) + \Delta t \cdot y(t)\f$
 *
 * The main idea behind the exponential Euler rule is 
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
 *  v(t+\Delta t)=v(t)+ C3 \cdot (0.04v(t)^2+5v(t)+140-u(t))+C2 \cdot (I_{syn}(t)+I_{inject}+I_{noise}+\frac{V_{resting}}{R_{m}})
 * \f]
 * \f[
 *  u(t+ \Delta t)=u(t) + C3 \cdot a \cdot (bv(t)-u(t))
 * \f]
 * where \f$\tau_{m}=C_{m} \cdot R_{m}\f$ is the membrane time constant, \f$R_{m}\f$ is the membrane resistance,
 * \f$C2 = R_m \cdot (1 - \mathrm{e}^{-\frac{\Delta t}{\tau_m}})\f$,
 * and \f$C3 = \Delta t\f$.
 *
 * Because the time scale \f$t\f$ of the Izhikevich model is \f$ms\f$ scale, 
 *  so \f$C3\f$ is : \f$C3 = \Delta t = 1000 \cdot deltaT\f$ (\f$deltaT\f$ is the simulation time step in second) \f$ms\f$.
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

struct AllIZHNeuronsDeviceProperties;

// Class to hold all data necessary for all the Neurons.
class AllIZHNeurons : public AllIFNeurons
{
    public:
        AllIZHNeurons();
        virtual ~AllIZHNeurons();

        static IAllNeurons* Create() { return new AllIZHNeurons(); }

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
        virtual void deserialize(istream &input, const SimulationInfo *sim_info);

        /**
         *  Writes out the data in all neurons to output stream.
         *
         *  @param  output      stream to write out to.
         *  @param  sim_info    used as a reference to set info for neuronss.
         */
        virtual void serialize(ostream& output, const SimulationInfo *sim_info) const;

#if defined(USE_GPU)
    public:
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
        virtual void advanceNeurons(IAllSynapses &synapses, void* allNeuronsDevice, void* allSynapsesDevice, const SimulationInfo *sim_info, float* randNoise, SynapseIndexMap* synapseIndexMapDevice);

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
        /**
         *  Allocate GPU memories to store all neurons' states.
         *  (Helper function of allocNeuronDeviceStruct)
         *
         *  @param  allNeurons         Reference to the AllIZHNeuronsDeviceProperties struct.
         *  @param  sim_info           SimulationInfo to refer from.
         */
        void allocDeviceStruct( AllIZHNeuronsDeviceProperties &allNeurons, SimulationInfo *sim_info );

        /**
         *  Delete GPU memories.
         *  (Helper function of deleteNeuronDeviceStruct)
         *
         *  @param  allNeurons         Reference to the AllIZHNeuronsDeviceProperties struct.
         *  @param  sim_info           SimulationInfo to refer from.
         */
        void deleteDeviceStruct( AllIZHNeuronsDeviceProperties& allNeurons, const SimulationInfo *sim_info );

        /**
         *  Copy all neurons' data from host to device.
         *  (Helper function of copyNeuronHostToDevice)
         *
         *  @param  allNeurons         Reference to the AllIZHNeuronsDeviceProperties struct.
         *  @param  sim_info           SimulationInfo to refer from.
         */
        void copyHostToDevice( AllIZHNeuronsDeviceProperties& allNeurons, const SimulationInfo *sim_info );

        /**
         *  Copy all neurons' data from device to host.
         *  (Helper function of copyNeuronDeviceToHost)
         *
         *  @param  allNeurons         Reference to the AllIZHNeuronsDeviceProperties struct.
         *  @param  sim_info           SimulationInfo to refer from.
         */
        void copyDeviceToHost( AllIZHNeuronsDeviceProperties& allNeurons, const SimulationInfo *sim_info );

#else  // !defined(USE_GPU)

    protected:
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
#endif  // defined(USE_GPU)

    protected:
        /**
         *  Creates a single Neuron and generates data for it.
         *
         *  @param  sim_info     SimulationInfo class to read information from.
         *  @param  neuron_index Index of the neuron to create.
         *  @param  layout       Layout information of the neunal network.
         */
        void createNeuron(SimulationInfo *sim_info, int neuron_index, Layout *layout);

        /**
         *  Set the Neuron at the indexed location to default values.
         *
         *  @param  neuron_index    Index of the Neuron that the synapse belongs to.
         */
        void setNeuronDefaults(const int index);

        /**
         *  Initializes the Neuron constants at the indexed location.
         *
         *  @param  neuron_index    Index of the Neuron.
         *  @param  deltaT          Inner simulation step duration
         */
        virtual void initNeuronConstsFromParamValues(int neuron_index, const BGFLOAT deltaT);

        /**
         *  Sets the data for Neuron #index to input's data.
         *
         *  @param  input       istream to read from.
         *  @param  sim_info    used as a reference to set info for neurons.
         *  @param  i           index of the neuron (in neurons).
         */
        void readNeuron(istream &input, const SimulationInfo *sim_info, int i);

        /**
         *  Writes out the data in the selected Neuron.
         *
         *  @param  output      stream to write out to.
         *  @param  sim_info    used as a reference to set info for neuronss.
         *  @param  i           index of the neuron (in neurons).
         */
        void writeNeuron(ostream& output, const SimulationInfo *sim_info, int i) const;

    private:
        /**
         *  Deallocate all resources
         */
        void freeResources();  

    public:
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
         *  Internal constant for the exponential Euler integration.
         */ 
        BGFLOAT *C3;

    private:
        /**
         *  Default value of Aconst.
         */
        static const BGFLOAT DEFAULT_a = 0.0035;

        /**
         *  Default value of Bconst.
         */
        static const BGFLOAT DEFAULT_b = 0.2;

        /**
         *  Default value of Cconst.
         */
        static const BGFLOAT DEFAULT_c = -50;

        /**
         *  Default value of Dconst.
         */
        static const BGFLOAT DEFAULT_d = 2;

        /**
         *  Min/max values of Aconst for excitatory neurons.
         */
        BGFLOAT m_excAconst[2];

        /**
         *  Min/max values of Aconst for inhibitory neurons.
         */
        BGFLOAT m_inhAconst[2];

        /**
         *  Min/max values of Bconst for excitatory neurons.
         */
        BGFLOAT m_excBconst[2];

        /**
         *  Min/max values of Bconst for inhibitory neurons.
         */
        BGFLOAT m_inhBconst[2];

        /**
         *  Min/max values of Cconst for excitatory neurons.
         */
        BGFLOAT m_excCconst[2];

        /**
         *  Min/max values of Cconst for inhibitory neurons.
         */
        BGFLOAT m_inhCconst[2];

        /**
         *  Min/max values of Dconst for excitatory neurons.
         */
        BGFLOAT m_excDconst[2];

        /**
         *  Min/max values of Dconst for inhibitory neurons.
         */
        BGFLOAT m_inhDconst[2];
};

#if defined(USE_GPU)
struct AllIZHNeuronsDeviceProperties : public AllIFNeuronsDeviceProperties
{
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
         *  Internal constant for the exponential Euler integration.
         */ 
        BGFLOAT *C3;
};
#endif // defined(USE_GPU)
