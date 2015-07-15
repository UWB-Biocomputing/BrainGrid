#pragma once

#include "Global.h"
#include "SimulationInfo.h"
#include <vector>
#include <iostream>

using namespace std;

class Layout
{
    public:
        // TODO
        Layout();
        virtual ~Layout();

        virtual void setupLayout(const SimulationInfo *sim_info);
        virtual bool readParameters(const TiXmlElement& element);
        virtual void printParameters(ostream &output) const;
        virtual void generateNeuronTypeMap(int num_neurons);
        virtual void initStarterMap(const int num_neurons);

        // Determines the type of synapse for a synapse between two neurons.
        synapseType synType(const int src_neuron, const int dest_neuron);

        // TODO
        VectorMatrix *xloc;
        // TODO
        VectorMatrix *yloc;
        //! Inter-neuron distance squared
        CompleteMatrix *dist2;
        //! the true inter-neuron distance
        CompleteMatrix *dist;

        // TODO
        vector<int> m_probed_neuron_list;

        // TODO
        BGFLOAT m_frac_starter_neurons;

        /*! The neuron type map (INH, EXC).
         *  
         *  Usage: LOCAL CONSTANT
         *  - LIFModel::generateNeuronTypeMap --- Initialized
         *  - LIFModel::logSimStep() --- Accessed
         *  - SingleThreadedSpikingModel::synType() --- Accessed
         *  - GpuSim_struct.cu::synType() --- Accessed
         *  - Hdf5Recorder::saveSimState() --- Accessed
         */
        neuronType *neuron_type_map;

        /*! The starter existence map (T/F).
         *  
         *  Usage: LOCAL CONSTANT
         *  - LIFModel::initStarterMap() --- Initialized
         *  - LIFModel::createAllNeurons() --- Accessed
         *  - LIFModel::logSimStep() --- Accessed
         *  - LIFModel::getStarterNeuronMatrix() --- Accessed
         *  - Hdf5Recorder::saveSimState() --- Accessed
         *  - XmlRecorder::saveSimState() --- Accessed
         */
        bool *starter_map;

    protected:
        virtual void initNeuronsLocs(const SimulationInfo *sim_info);

        static const bool STARTER_FLAG; // = true; // true = use endogenously active neurons in simulation

    private:
        //! True if a fixed layout has been provided
        bool m_fixed_layout;
        // TODO
        bool m_grid_layout;
        // TODO
        vector<int> m_endogenously_active_neuron_list;
        // TODO
        vector<int> m_inhibitory_neuron_layout;
        // TODO
        BGFLOAT m_frac_excititory_neurons;
};

