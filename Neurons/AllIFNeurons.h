/**
 *      @file AllIFNeurons.h
 *
 *      @brief A container of all Integate and Fire (IF) neuron data
 */

/** 
 ** @authors Aaron Oziel, Sean Blackbourn
 **
 ** @class AllIFNeurons AllIFNeurons.h "AllIFNeurons.h"
 **
 ** \latexonly  \subsubsection*{Implementation} \endlatexonly
 ** \htmlonly   <h3>Implementation</h3> \endhtmlonly
 **
 ** A container of all Integate and Fire (IF) neuron data.
 ** This is the base class of all Integate and Fire (IF) neuron classes.
 **
 ** The class uses a data-centric structure, which utilizes a structure as the containers of
 ** all neuron.
 **
 ** The container holds neuron parameters of all neurons.
 ** Each kind of neuron parameter is stored in a 1D array, of which length
 ** is number of all neurons. Each array of a neuron parameter is pointed by a
 ** corresponding member variable of the neuron parameter in the class.
 **
 ** This structure was originally designed for the GPU implementation of the
 ** simulator, and this refactored version of the simulator simply uses that design for
 ** all other implementations as well. This is to simplify transitioning from
 ** single-threaded to multi-threaded.
 **
 ** \latexonly  \subsubsection*{Credits} \endlatexonly
 ** \htmlonly   <h3>Credits</h3> \endhtmlonly
 **
 ** Some models in this simulator is a rewrite of CSIM (2006) and other
 ** work (Stiber and Kawasaki (2007?))
 **
 **/
#pragma once

#include "Global.h"
#include "AllSpikingNeurons.h"

struct AllIFNeuronsDeviceProperties;

class AllIFNeurons : public AllSpikingNeurons
{
    public:
        AllIFNeurons();
        virtual ~AllIFNeurons();

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
         *  @param  allNeurons         Reference to the AllIFNeuronsDeviceProperties struct.
         *  @param  sim_info           SimulationInfo to refer from.
         */
        void allocDeviceStruct( AllIFNeuronsDeviceProperties &allNeurons, SimulationInfo *sim_info );

        /**
         *  Delete GPU memories.
         *  (Helper function of deleteNeuronDeviceStruct)
         *
         *  @param  allNeurons         Reference to the AllIFNeuronsDeviceProperties struct.
         *  @param  sim_info           SimulationInfo to refer from.
         */
        void deleteDeviceStruct( AllIFNeuronsDeviceProperties& allNeurons, const SimulationInfo *sim_info );

        /**
         *  Copy all neurons' data from host to device.
         *  (Helper function of copyNeuronHostToDevice)
         *
         *  @param  allNeurons         Reference to the AllIFNeuronsDeviceProperties struct.
         *  @param  sim_info           SimulationInfo to refer from.
         */
	void copyHostToDevice( AllIFNeuronsDeviceProperties& allNeurons, const SimulationInfo *sim_info );

        /**
         *  Copy all neurons' data from device to host.
         *  (Helper function of copyNeuronDeviceToHost)
         *
         *  @param  allNeurons         Reference to the AllIFNeuronsDeviceProperties struct.
         *  @param  sim_info           SimulationInfo to refer from.
         */
	void copyDeviceToHost( AllIFNeuronsDeviceProperties& allNeurons, const SimulationInfo *sim_info );

#endif // defined(USE_GPU)

    protected:
        /**
         *  Creates a single Neuron and generates data for it.
         *
         *  @param  sim_info     SimulationInfo class to read information from.
         *  @param  neuron_index Index of the neuron to create.
         *  @param  layout       Layout information of the neunal network.
         */
        void createNeuron(SimulationInfo *sim_info, int neuron_index, Layout *layoug);

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
         *  The length of the absolute refractory period. [units=sec; range=(0,1);]
         */
        BGFLOAT *Trefract;

        /**
         *  If \f$V_m\f$ exceeds \f$V_{thresh}\f$ a spike is emmited. [units=V; range=(-10,100);]
         */
        BGFLOAT *Vthresh;

        /**
         *  The resting membrane voltage. [units=V; range=(-1,1);]
         */
        BGFLOAT *Vrest;

        /**
         *  The voltage to reset \f$V_m\f$ to after a spike. [units=V; range=(-1,1);]
         */
        BGFLOAT *Vreset;

        /**
         *  The initial condition for \f$V_m\f$ at time \f$t=0\f$. [units=V; range=(-1,1);]
         */
        BGFLOAT *Vinit;

        /**
         *  The membrane capacitance \f$C_m\f$ [range=(0,1); units=F;]
         *  Used to initialize Tau (no use after that)
         */
        BGFLOAT *Cm;

        /**
         *  The membrane resistance \f$R_m\f$ [units=Ohm; range=(0,1e30)]
         */
        BGFLOAT *Rm;

        /**
         * The standard deviation of the noise to be added each integration time constant. [range=(0,1); units=A;]
         */
        BGFLOAT *Inoise;

        /**
         *  A constant current to be injected into the LIF neuron. [units=A; range=(-1,1);]
         */
        BGFLOAT *Iinject;

        /**
         * What the hell is this used for???
         *  It does not seem to be used; seems to be a candidate for deletion.
         *  Possibly from the old code before using a separate summation point
         *  The synaptic input current.
         */
        BGFLOAT *Isyn;

        /**
         * The remaining number of time steps for the absolute refractory period.
         */
        int *nStepsInRefr;

        /**
         * Internal constant for the exponential Euler integration of f$V_m\f$.
         */
        BGFLOAT *C1;

        /**
         * Internal constant for the exponential Euler integration of \f$V_m\f$.
         */
        BGFLOAT *C2;

        /**
         * Internal constant for the exponential Euler integration of \f$V_m\f$.
         */
        BGFLOAT *I0;

        /**
         * The membrane voltage \f$V_m\f$ [readonly; units=V;]
         */
        BGFLOAT *Vm;

        /**
         * The membrane time constant \f$(R_m \cdot C_m)\f$
         */
        BGFLOAT *Tau;

    private:
        /**
         * Min/max values of Iinject.
         */
        BGFLOAT m_Iinject[2];

        /**
         * Min/max values of Inoise.
         */
        BGFLOAT m_Inoise[2];

        /**
         * Min/max values of Vthresh.
         */
        BGFLOAT m_Vthresh[2];

        /**
         * Min/max values of Vresting.
         */
        BGFLOAT m_Vresting[2];

        /**
         * Min/max values of Vreset.
         */
        BGFLOAT m_Vreset[2];

        /**
         * Min/max values of Vinit.
         */
        BGFLOAT m_Vinit[2];

        /**
         * Min/max values of Vthresh.
         */
        BGFLOAT m_starter_Vthresh[2];

        /**
         * Min/max values of Vreset.
         */
        BGFLOAT m_starter_Vreset[2];
};

#if defined(USE_GPU)
struct AllIFNeuronsDeviceProperties : public AllSpikingNeuronsDeviceProperties
{
        /**
         *  The length of the absolute refractory period. [units=sec; range=(0,1);]
         */
        BGFLOAT *Trefract;

        /**
         *  If \f$V_m\f$ exceeds \f$V_{thresh}\f$ a spike is emmited. [units=V; range=(-10,100);]
         */
        BGFLOAT *Vthresh;

        /**
         *  The resting membrane voltage. [units=V; range=(-1,1);]
         */
        BGFLOAT *Vrest;

        /**
         *  The voltage to reset \f$V_m\f$ to after a spike. [units=V; range=(-1,1);]
         */
        BGFLOAT *Vreset;

        /**
         *  The initial condition for \f$V_m\f$ at time \f$t=0\f$. [units=V; range=(-1,1);]
         */
        BGFLOAT *Vinit;

        /**
         *  The membrane capacitance \f$C_m\f$ [range=(0,1); units=F;]
         *  Used to initialize Tau (no use after that)
         */
        BGFLOAT *Cm;

        /**
         *  The membrane resistance \f$R_m\f$ [units=Ohm; range=(0,1e30)]
         */
        BGFLOAT *Rm;

        /**
         * The standard deviation of the noise to be added each integration time constant. [range=(0,1); units=A;]
         */
        BGFLOAT *Inoise;

        /**
         *  A constant current to be injected into the LIF neuron. [units=A; range=(-1,1);]
         */
        BGFLOAT *Iinject;

        /**
         * What the hell is this used for???
         *  It does not seem to be used; seems to be a candidate for deletion.
         *  Possibly from the old code before using a separate summation point
         *  The synaptic input current.
         */
        BGFLOAT *Isyn;

        /**
         * The remaining number of time steps for the absolute refractory period.
         */
        int *nStepsInRefr;

        /**
         * Internal constant for the exponential Euler integration of f$V_m\f$.
         */
        BGFLOAT *C1;

        /**
         * Internal constant for the exponential Euler integration of \f$V_m\f$.
         */
        BGFLOAT *C2;

        /**
         * Internal constant for the exponential Euler integration of \f$V_m\f$.
         */
        BGFLOAT *I0;

        /**
         * The membrane voltage \f$V_m\f$ [readonly; units=V;]
         */
        BGFLOAT *Vm;

        /**
         * The membrane time constant \f$(R_m \cdot C_m)\f$
         */
        BGFLOAT *Tau;
};
#endif // defined(USE_GPU)
