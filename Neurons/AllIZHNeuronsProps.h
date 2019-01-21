/**
 *      @file AllIZHNeuronsProps.h
 *
 *      @brief A container of the base class of all neuron data
 */

#pragma once 

#include "AllIFNeuronsProps.h"

class AllIZHNeuronsProps : public AllIFNeuronsProps
{
    public:
        AllIZHNeuronsProps();
        virtual ~AllIZHNeuronsProps();

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
         *  Copy neurons parameters.
         *
         *  @param  r_neurons  Neurons class object to copy from.
         */
        virtual void copyParameters(const AllNeuronsProps *r_neuronsProps);

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
        static constexpr BGFLOAT DEFAULT_a = 0.0035;

        /**
         *  Default value of Bconst.
         */
        static constexpr BGFLOAT DEFAULT_b = 0.2;

        /**
         *  Default value of Cconst.
         */
        static constexpr BGFLOAT DEFAULT_c = -50;

        /**
         *  Default value of Dconst.
         */
        static constexpr BGFLOAT DEFAULT_d = 2;

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
