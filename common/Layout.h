#pragma once

#include "Global.h"
#include "SimulationInfo.h"
#include <vector>
#include <iostream>

using namespace std;

class XmlGrowthRecorder;
#ifdef USE_HDF5
class Hdf5GrowthRecorder;
#endif // USE_HDF5

class Layout
{
        friend XmlGrowthRecorder;
#ifdef USE_HDF5
        friend Hdf5GrowthRecorder;
#endif // USE_HDF5

    public:
        // TODO
        Layout();
        virtual ~Layout();

        virtual bool readParameters(const TiXmlElement& element);
        virtual void printParameters(ostream &output) const;
        virtual void generateNeuronTypeMap(neuronType neuron_types[], int num_neurons);
        virtual void initStarterMap(bool *starter_map, const int num_neurons, const neuronType neuron_type_map[]);

    protected:
        static const bool STARTER_FLAG; // = true; // true = use endogenously active neurons in simulation

        //! True if a fixed layout has been provided
        bool m_fixed_layout;

        // TODO
        vector<int> m_endogenously_active_neuron_list;
        // TODO
        vector<int> m_inhibitory_neuron_layout;
        // TODO
        vector<int> m_probed_neuron_list;

        // TODO
        BGFLOAT m_frac_starter_neurons;
        // TODO
        BGFLOAT m_frac_excititory_neurons;
};

