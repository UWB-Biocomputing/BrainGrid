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
        Layout();
        virtual ~Layout();

        /**
         *  Setup the internal structure of the class. 
         *  Allocate memories to store all layout state.
         *
         *  @param  sim_info  SimulationInfo class to read information from.
         */
        virtual void setupLayout(const SimulationInfo *sim_info);

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
        virtual bool readParameters(const TiXmlElement& element);

        /**
         *  Prints out all parameters of the neurons to ostream.
         *
         *  @param  output  ostream to send output to.
         */
        virtual void printParameters(ostream &output) const;

        /**
         *  Creates a neurons type map.
         *
         *  @param  num_neurons number of the neurons to have in the type map.
         */
        virtual void generateNeuronTypeMap(int num_neurons);

        /**
         *  Populates the starter map.
         *  Selects num_endogenously_active_neurons excitory neurons 
         *  and converts them into starter neurons.
         *
         *  @param  num_neurons number of neurons to have in the map.
         */
        virtual void initStarterMap(const int num_neurons);

        /**
         *  Returns the type of synapse at the given coordinates
         *
         *  @param    src_neuron  integer that points to a Neuron in the type map as a source.
         *  @param    dest_neuron integer that points to a Neuron in the type map as a destination.
         *  @return type of the synapse.
         */
        synapseType synType(const int src_neuron, const int dest_neuron);

        //! Store neuron i's x location.
        VectorMatrix *xloc;

        //! Store neuron i's y location.
        VectorMatrix *yloc;

        // Inter-neuron distance squared.
        CompleteMatrix *dist2;

        //! The true inter-neuron distance.
        CompleteMatrix *dist;

        //! Probed neurons list.
        vector<int> m_probed_neuron_list;

        //! The neuron type map (INH, EXC).
        neuronType *neuron_type_map;

        //! The starter existence map (T/F).
        bool *starter_map;

        //! Number of endogenously active neurons.
        BGSIZE num_endogenously_active_neurons;

    protected:
        //! Number of parameters read.
        int nParams;

        //! Endogenously active neurons list. 
        vector<int> m_endogenously_active_neuron_list;

        //! Inhibitory neurons list.
        vector<int> m_inhibitory_neuron_layout;

    private:
        /*
         *  Initialize the location maps (xloc and yloc).
         *
         *  @param sim_info   SimulationInfo class to read information from.
         */
        void initNeuronsLocs(const SimulationInfo *sim_info);

        // True if grid layout.
        bool m_grid_layout;
};

