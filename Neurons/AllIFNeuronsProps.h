/**
 *      @file AllIFNeuronsProps.h
 *
 *      @brief A container of the base class of all neuron data
 */

#pragma once 

#include "AllSpikingNeuronsProps.h"

class AllIFNeuronsProps : public AllSpikingNeuronsProps
{
    public:
        AllIFNeuronsProps();
        virtual ~AllIFNeuronsProps();

        /**
         *  Setup the internal structure of the class.
         *  Allocate memories to store all neurons' state.
         *
         *  @param  sim_info  SimulationInfo class to read information from.
         *  @param  clr_info  ClusterInfo class to read information from.
         */
        virtual void setupNeuronsProps(SimulationInfo *sim_info, ClusterInfo *clr_info);

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
         *  Copy neurons parameters.
         *
         *  @param  r_neurons  Neurons properties class object to copy from.
         */
        virtual void copyParameters(const AllNeuronsProps *r_neuronsProps);

        /**
         *  Sets the data for Neuron #index to input's data.
         *
         *  @param  input       istream to read from.
         *  @param  i           index of the neuron (in neurons).
         */
        virtual void readNeuronProps(istream &input, int i);

        /**
         *  Writes out the data in the selected Neuron.
         *
         *  @param  output      stream to write out to.
         *  @param  i           index of the neuron (in neurons).
         */
        virtual void writeNeuronProps(ostream& output, int i) const;

        /**
         *  Creates a single Neuron and generates data for it.
         *
         *  @param  sim_info     SimulationInfo class to read information from.
         *  @param  neuron_index Index of the neuron to create.
         *  @param  layout       Layout information of the neunal network.
         *  @param  clr_info     ClusterInfo class to read information from.
         */
        virtual void setNeuronPropValues(SimulationInfo *sim_info, int neuron_index, Layout *layoug, ClusterInfo *clr_info);

        /**
         *  Set the Neuron at the indexed location to default values.
         *
         *  @param  neuron_index    Index of the Neuron that the synapse belongs to.
         */
        virtual void setNeuronPropDefaults(const int index);

    protected:
        /**
         *  Initializes the Neuron constants at the indexed location.
         *
         *  @param  neuron_index    Index of the Neuron.
         *  @param  deltaT          Inner simulation step duration
         */
        void initNeuronPropConstsFromParamValues(int neuron_index, const BGFLOAT deltaT);

    private:
        /**
         *  Cleanup the class.
         *  Deallocate memories.
         */
        void cleanupNeuronsProps();


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
