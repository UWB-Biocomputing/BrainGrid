/** 
 * @authors Aaron Oziel, Sean Blackbourn
 * 
 * @class AllIFNeurons AllIFNeurons.h "AllIFNeurons.h"
 * @brief A container of all LIF neuron data
 *
 *  The container holds neuron parameters of all neurons. 
 *  Each kind of neuron parameter is stored in a 1D array, of which length
 *  is number of all neurons. Each array of a neuron parameter is pointed by a 
 *  corresponding member variable of the neuron parameter in the class.
 *
 *  In this file you will find usage statistics for every variable in the BrainGrid 
 *  project as we find them. These statistics can be used to help 
 *  determine if a variable is being used, where it is being used, and how it
 *  is being used in each class::function()
 *  
 *  For Example
 *
 *  Usage:
 *  - LOCAL VARIABLE -- a variable for individual neuron
 *  - LOCAL CONSTANT --  a constant for individual neuron
 *  - GLOBAL VARIABLE -- a variable for all neurons
 *  - GLOBAL CONSTANT -- a constant for all neurons
 *
 *  Class::function(): --- Initialized, Modified OR Accessed
 *
 *  OtherClass::function(): --- Accessed   
 *
 *  Note: All GLOBAL parameters can be scalars. Also some LOCAL CONSTANT can be categorized 
 *  depending on neuron types. 
 */
#pragma once

#include "Global.h"
#include "AllSpikingNeurons.h"

// Class to hold all data necessary for all the Neurons.
class AllIFNeurons : public AllSpikingNeurons
{
    public:

        /*! The length of the absolute refractory period. [units=sec; range=(0,1);]
         *  
         *  Usage: LOCAL CONSTANT depending on a type of neuron
         *  - LIFModel::createAllNeurons() --- Initialized
         *  - LIFModel::setNeuronDefaults() --- Initialized
         *  - LIFModel::readNeuron() --- Modified
         *  - LIFModel::writeNeuron() --- Accessed
         *  - SingleThreadedSpikingModel::fire() --- Accessed
         *  - GpuSim_struct.cu::advanceNeuronsDevice() --- Accessed
         */
        BGFLOAT *Trefract;

        /*! If \f$V_m\f$ exceeds \f$V_{thresh}\f$ a spike is emmited. [units=V; range=(-10,100);]
         *  
         *  Usage: LOCAL CONSTANT
         *  - LIFModel::createAllNeurons() --- Initialized
         *  - LIFModel::setNeuronDefaults() --- Initialized
         *  - LIFModel::readNeuron() --- Modified
         *  - LIFModel::writeNeuron() --- Accessed
         *  - SingleThreadedSpikingModel::advanceNeuron() --- Accessed
         *  - GpuSim_struct.cu::advanceNeuronsDevice() --- Accessed
         *  - Hdf5Recorder::saveSimState() --- Accessed
         *  - XmlRecorder::saveSimState() --- Accessed
         */
        BGFLOAT *Vthresh;

        /*! The resting membrane voltage. [units=V; range=(-1,1);]
         *  
         *  Usage: LOCAL CONSTANT
         *  - LIFModel::createAllNeurons() --- Initialized
         *  - LIFModel::setNeuronDefaults() --- Initialized
         *  - LIFModel::readNeuron() --- Modified
         *  - LIFModel::writeNeuron() --- Accessed
         *  - LIFModel::initNeuronConstsFromParamValues() --- Accessed
         */
        BGFLOAT *Vrest;

        /*! The voltage to reset \f$V_m\f$ to after a spike. [units=V; range=(-1,1);]
         *  
         *  Usage: LOCAL CONSTANT
         *  - LIFModel::createAllNeurons() --- Initialized
         *  - LIFModel::setNeuronDefaults() --- Initialized
         *  - LIFModel::readNeuron() --- Modified
         *  - LIFModel::writeNeuron() --- Accessed
         *  - SingleThreadedSpikingModel::fire() --- Accessed
         *  - GpuSim_struct.cu::advanceNeuronsDevice() --- Accessed
         */
        BGFLOAT *Vreset;

        /*! The initial condition for \f$V_m\f$ at time \f$t=0\f$. [units=V; range=(-1,1);]
         *
         *  Usage: LOCAL CONSTANT
         *  - LIFModel::createAllNeurons() --- Initialized & Accessed
         *  - LIFModel::setNeuronDefaults() --- Initialized
         *  - LIFModel::readNeuron() --- Modified
         *  - LIFModel::writeNeuron() --- Accessed
         */
        BGFLOAT *Vinit;

        /*! The membrane capacitance \f$C_m\f$ [range=(0,1); units=F;]
         *  Used to initialize Tau (no use after that)
         *
         *  Usage: GLOBAL CONSTANT
         *  - LIFModel::setNeuronDefaults() --- Initialized and accessed
         *  - LIFModel::readNeuron() --- Modified
         *  - LIFModel::writeNeuron() --- Accessed
         */
        BGFLOAT *Cm;

        /*! The membrane resistance \f$R_m\f$ [units=Ohm; range=(0,1e30)]
         *  
         *  Usage: GLOBAL CONSTANT
         *  - LIFModel::setNeuronDefaults() --- Initialized
         *  - LIFModel::initNeuronConstsFromParamValues() --- Accessed
         *  - LIFModel::readNeuron() --- Modified
         *  - LIFModel::writeNeuron() --- Accessed
         */
        BGFLOAT *Rm;

        /*! The standard deviation of the noise to be added each integration time constant. [range=(0,1); units=A;]
         *  
         *  Usage: LOCAL CONSTANT
         *  - LIFModel::createAllNeurons() --- Initialized
         *  - LIFModel::setNeuronDefaults() --- Initialized
         *  - LIFModel::readNeuron() --- Modified
         *  - LIFModel::writeNeuron() --- Accessed
         *  - SingleThreadedSpikingModel::advanceNeuron() --- Accessed
         *  - GpuSim_struct.cu::advanceNeuronsDevice() --- Accessed
         */
        BGFLOAT *Inoise;

        /*! A constant current to be injected into the LIF neuron. [units=A; range=(-1,1);]
         *  
         *  Usage: LOCAL CONSTANT
         *  - LIFModel::createAllNeurons() --- Initialized
         *  - LIFModel::setNeuronDefaults() --- Initialized
         *  - LIFModel::readNeuron() --- Modified
         *  - LIFModel::writeNeuron() --- Accessed
         *  - LIFModel::initNeuronConstsFromParamValues() --- Accessed
         */
        BGFLOAT *Iinject;

        /*! What the hell is this used for???
         *  It does not seem to be used; seems to be a candidate for deletion.
         *  Possibly from the old code before using a separate summation point
         *  The synaptic input current.
         *  
         *  Usage: NOT USED ANYWHERE
         *  - LIFModel::readNeuron() --- Modified
         *  - LIFModel::writeNeuron() --- Accessed
         */
        BGFLOAT *Isyn;

        /*! The remaining number of time steps for the absolute refractory period.
         *  
         *  Usage: LOCAL VARIABLE
         *  - AllIFNeurons::AllIFNeurons() --- Initialized
         *  - LIFModel::readNeuron() --- Modified
         *  - LIFModel::writeNeuron() --- Accessed
         *  - SingleThreadedSpikingModel::advanceNeuron() --- Accessed & Modified
         *  - SingleThreadedSpikingModel::fire() --- Modified
         *  - GpuSim_struct.cu::advanceNeuronsDevice() --- Accessed & Modified
         */
        int *nStepsInRefr;

        /*! Internal constant for the exponential Euler integration of f$V_m\f$.
         *  
         *  Usage: GLOBAL CONSTANT
         *  - LIFModel::initNeuronConstsFromParamValues() --- Initialized
         *  - LIFModel::readNeuron() --- Modified
         *  - LIFModel::writeNeuron() --- Accessed
         *  - SingleThreadedSpikingModel::advanceNeuron() --- Accessed
         *  - GpuSim_struct.cu::advanceNeuronsDevice() --- Accessed
         */
        BGFLOAT *C1;

        /*! Internal constant for the exponential Euler integration of \f$V_m\f$.
         *  
         *  Usage: GLOBAL CONSTANT
         *  - LIFModel::initNeuronConstsFromParamValues() --- Initialized
         *  - LIFModel::readNeuron() --- Modified
         *  - LIFModel::writeNeuron() --- Accessed
         *  - SingleThreadedSpikingModel::advanceNeuron() --- Accessed
         *  - GpuSim_struct.cu::advanceNeuronsDevice() --- Accessed
         */
        BGFLOAT *C2;

        /*! Internal constant for the exponential Euler integration of \f$V_m\f$.
         *  
         *  Usage: LOCAL CONSTANT
         *  - LIFModel::initNeuronConstsFromParamValues() --- Initialized
         *  - LIFModel::readNeuron() --- Modified
         *  - LIFModel::writeNeuron() --- Accessed
         *  - SingleThreadedSpikingModel::advanceNeuron() --- Accessed
         *  - GpuSim_struct.cu::advanceNeuronsDevice() --- Accessed
         */
        BGFLOAT *I0;

        /*! The membrane voltage \f$V_m\f$ [readonly; units=V;]
         *  
         *  Usage: LOCAL VARIABLE
         *  - LIFModel::createAllNeurons() --- Initialized
         *  - LIFModel::readNeuron() --- Modified
         *  - LIFModel::writeNeuron() --- Accessed
         *  - SingleThreadedSpikingModel::advanceNeuron() --- Accessed & Modified
         *  - SingleThreadedSpikingModel::fire() --- Modified
         *  - GpuSim_struct.cu::advanceNeuronsDevice() --- Accessed & Modified
         */
        BGFLOAT *Vm;

        /*! The membrane time constant \f$(R_m \cdot C_m)\f$
         *  
         *  Usage: GLOBAL CONSTANT
         *  - LIFModel::setNeuronDefaults() --- Initialized
         *  - LIFModel::readNeuron() --- Modified
         *  - LIFModel::writeNeuron() --- Accessed
         *  - LIFModel::initNeuronConstsFromParamValues() --- Accessed
         */
        BGFLOAT *Tau;

        AllIFNeurons();
        virtual ~AllIFNeurons();

        virtual void setupNeurons(SimulationInfo *sim_info);
        virtual void cleanupNeurons();  
        virtual int numParameters();
        virtual int readParameters(const TiXmlElement& element);
        virtual void printParameters(ostream &output) const;
        virtual void createAllNeurons(SimulationInfo *sim_info);
        virtual string toString(const int i) const;
        virtual void readNeurons(istream &input, const SimulationInfo *sim_info);
        virtual void writeNeurons(ostream& output, const SimulationInfo *sim_info) const;

#if defined(USE_GPU)
        virtual void allocNeuronDeviceStruct( void** allNeuronsDevice, SimulationInfo *sim_info );
        virtual void deleteNeuronDeviceStruct( void* allNeuronsDevice, const SimulationInfo *sim_info );
        virtual void copyNeuronHostToDevice( void* allNeuronsDevice, const SimulationInfo *sim_info );
        virtual void copyNeuronDeviceToHost( void* allNeuronsDevice, const SimulationInfo *sim_info );
#endif

    protected:
        void allocDeviceStruct( AllIFNeurons &allNeurons, SimulationInfo *sim_info );
        void deleteDeviceStruct( AllIFNeurons& allNeurons, const SimulationInfo *sim_info );
	void copyHostToDevice( AllIFNeurons& allNeurons, const SimulationInfo *sim_info );
	void copyDeviceToHost( AllIFNeurons& allNeurons, const SimulationInfo *sim_info );

        void createNeuron(SimulationInfo *sim_info, int neuron_index);
        void setNeuronDefaults(const int index);
        virtual void initNeuronConstsFromParamValues(int neuron_index, const BGFLOAT deltaT);
        void readNeuron(istream &input, const SimulationInfo *sim_info, int i);
        void writeNeuron(ostream& output, const SimulationInfo *sim_info, int i) const;

    private:
        // TODO
        BGFLOAT m_Iinject[2];
        // TODO
        BGFLOAT m_Inoise[2];
        // TODO
        BGFLOAT m_Vthresh[2];
        // TODO
        BGFLOAT m_Vresting[2];
        // TODO
        BGFLOAT m_Vreset[2];
        // TODO
        BGFLOAT m_Vinit[2];
        // TODO
        BGFLOAT m_starter_Vthresh[2];
        // TODO
        BGFLOAT m_starter_Vreset[2];

        void freeResources();  
};
