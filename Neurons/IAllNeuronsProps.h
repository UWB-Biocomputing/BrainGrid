/**
 *      @file IAllNeuronsProps.h
 *
 *      @brief An interface for neuron properties class.
 */

#pragma once

#include "SimulationInfo.h"
#include "ClusterInfo.h"
#include "Layout.h"

class AllNeuronsProps;

class IAllNeuronsProps
{
    public:
        virtual ~IAllNeuronsProps() {};

        /**
         *  Setup the internal structure of the class.
         *  Allocate memories to store all neurons' state.
         *
         *  @param  sim_info  SimulationInfo class to read information from.
         *  @param  clr_info  ClusterInfo class to read information from.
         */
        virtual void setupNeuronsProps(SimulationInfo *sim_info, ClusterInfo *clr_info) = 0;

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
         *  Sets the data for Neuron #index to input's data.
         *
         *  @param  input       istream to read from.
         *  @param  i           index of the neuron (in neurons).
         */
        virtual void readNeuronProps(istream &input, int i) = 0;

        /**
         *  Writes out the data in the selected Neuron.
         *
         *  @param  output      stream to write out to.
         *  @param  i           index of the neuron (in neurons).
         */
        virtual void writeNeuronProps(ostream& output, int i) const = 0;

        /**
         *  Copy neurons parameters.
         *
         *  @param  r_neurons  Neurons class object to copy from.
         */
        virtual void copyParameters(const AllNeuronsProps *r_neuronsProps) = 0;

        /**
         *  Creates a single Neuron and generates data for it.
         *
         *  @param  sim_info     SimulationInfo class to read information from.
         *  @param  neuron_index Index of the neuron to create.
         *  @param  layout       Layout information of the neunal network.
         *  @param  clr_info     ClusterInfo class to read information from.
         */
        virtual void setNeuronPropValues(SimulationInfo *sim_info, int neuron_index, Layout *layoug, ClusterInfo *clr_info) = 0;

        /**
         *  Set the Neuron at the indexed location to default values.
         *
         *  @param  neuron_index    Index of the Neuron that the synapse belongs to.
         */
        virtual void setNeuronPropDefaults(const int index) = 0;
};
