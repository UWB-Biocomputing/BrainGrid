/**
 *      @file AllNeuronsProps.h
 *
 *      @brief A container of the base class of all neuron data
 */

#pragma once

#include "IAllNeuronsProps.h"

class AllNeuronsProps : public IAllNeuronsProps
{
    public:
        AllNeuronsProps();
        virtual ~AllNeuronsProps();

        /**
         *  Setup the internal structure of the class.
         *  Allocate memories to store all neurons' state.
         *
         *  @param  sim_info  SimulationInfo class to read information from.
         *  @param  clr_info  ClusterInfo class to read information from.
         */
        virtual void setupNeuronsProps(SimulationInfo *sim_info, ClusterInfo *clr_info);

#if defined(USE_GPU)
        /**
         *  Allocate GPU memories to store all neurons' states,
         *  and copy them from host to GPU memory.
         *
         *  @param  allNeuronsDeviceProps   Reference to the AllIZHNeuronsProps class on device memory.
         *  @param  sim_info           SimulationInfo to refer from.
         *  @param  clr_info           ClusterInfo to refer from.
         */
        virtual void setupNeuronsDeviceProps(void** allNeuronsDeviceProps, SimulationInfo *sim_info, ClusterInfo *clr_info);

        /**
         *  Delete GPU memories.
         *
         *  @param  allNeuronsDeviceProps   Reference to the AllIZHNeuronsProps class on device memory.
         *  @param  clr_info           ClusterInfo to refer from.
         */
        virtual void cleanupNeuronsDeviceProps(void *allNeuronsDeviceProps, ClusterInfo *clr_info);

        /**
         *  Copy all neurons' data from host to device.
         *
         *  @param  allNeuronsDeviceProps   Reference to the AllIFNeuronsProps class on device memory.
         *  @param  sim_info                SimulationInfo to refer from.
         *  @param  clr_info                ClusterInfo to refer from.
         */
        virtual void copyNeuronHostToDeviceProps( void* allNeuronsDeviceProps, const SimulationInfo *sim_info, const ClusterInfo *clr_info );

        /*
         *  Copy all neurons' data from device to host.
         *
         *  @param  allNeuronsDeviceProps   Reference to the AllIFNeuronsProps class on device memory.
         *  @param  sim_info                SimulationInfo to refer from.
         *  @param  clr_info                ClusterInfo to refer from.
         */
        virtual void copyNeuronDeviceToHostProps( void* allNeuronsDeviceProps, const SimulationInfo *sim_info, const ClusterInfo *clr_info );

    protected:
        /**
         *  Allocate GPU memories to store all neurons' states.
         *
         *  @param  allNeuronsProps   Reference to the AllNeuronsProps struct.
         *  @param  sim_info  SimulationInfo class to read information from.
         *  @param  clr_info               ClusterInfo to refer from.
         */
        void allocNeuronsDeviceProps(AllNeuronsProps &allNeuronsProps, SimulationInfo *sim_info, ClusterInfo *clr_info);

        /**
         *  Delete GPU memories.
         *
         *  @param  allNeuronsProps   Reference to the AllNeuronsProps class.
         *  @param  clr_info               ClusterInfo to refer from.
         */
        void deleteNeuronsDeviceProps(AllNeuronsProps &allNeuronsProps, ClusterInfo *clr_info);
#endif // USE_GPU
    public:
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

    private:
        /**
         *  Cleanup the class.
         *  Deallocate memories.
         */
        void cleanupNeuronsProps();

    protected:
        /**
         *  Total number of neurons.
         */
        int size;

        /**
         *  Number of parameters read.
         */
        int nParams;

    public:
        /**
         *  The summation point for each neuron.
         *  Summation points are places where the synapses connected to the neuron
         *  apply (summed up) their PSRs (Post-Synaptic-Response).
         *  On the next advance cycle, neurons add the values stored in their corresponding
         *  summation points to their Vm and resets the summation points to zero
         */
        BGFLOAT *summation_map;
};
