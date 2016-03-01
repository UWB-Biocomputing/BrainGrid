/**
 *      @file Layout.h
 *
 *      @brief The Layout class defines the layout of neurons in neunal networks
 */

/**
 *
 * @class Layout Layout.h "Layout.h"
 *
 * \latexonly  \subsubsection*{Implementation} \endlatexonly
 * \htmlonly   <h3>Implementation</h3> \endhtmlonly
 *
 * The Layout class maintains neurons locations (x, y coordinates), distance of every couple neurons,
 * neurons type map (distribution of excitatory and inhibitory neurons), and starter neurons map
 * (distribution of endogenously active neurons).  
 *
 */

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

        static Layout* Create() { return new Layout(); }

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

        /** 
         * The neuron type map (INH, EXC).
         */
        neuronType *neuron_type_map;

        /** 
         * The starter existence map (T/F).
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

