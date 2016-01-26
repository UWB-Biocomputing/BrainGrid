/**
 *      @file Model.h
 *
 *      @brief Implementation of Model for the spiking neunal networks.
 */

/**
 *
 * @class Model Model.h "Model.h"
 *
 * \latexonly  \subsubsection*{Implementation} \endlatexonly
 * \htmlonly   <h3>Implementation</h3> \endhtmlonly
 *
 * The Model class maintains and manages classes of objects that make up
 * essential components of the spiking neunal networks.
 *    -# IAllNeurons: A class to define a list of partiular type of neurons.
 *    -# IAllSynapses: A class to define a list of partiular type of synapses.
 *    -# Connections: A class to define connections of the neunal network.
 *    -# Layout: A class to define neurons' layout information in the network.
 *
 * \latexonly  \subsubsection*{Credits} \endlatexonly
 * \htmlonly   <h3>Credits</h3> \endhtmlonly
 *
 * Some models in this simulator is a rewrite of CSIM (2006) and other
 * work (Stiber and Kawasaki (2007?))
 */

#pragma once

#include "IModel.h"
#include "Coordinate.h"
#include "Layout.h"
#include "SynapseIndexMap.h"

#include <vector>
#include <iostream>

using namespace std;

class Model : public IModel, TiXmlVisitor
{
    public:
        Model(Connections *conns, IAllNeurons *neurons, IAllSynapses *synapses, Layout *layout);
        virtual ~Model();

        /*
         * Declarations of concrete implementations of Model interface for an 
         * Leaky-Integrate-and-Fire * model.
         *
         * @see Model.h
         */
        virtual void loadMemory(istream& input, const SimulationInfo *sim_info);
        virtual void saveMemory(ostream& output, const SimulationInfo *sim_info);
        virtual void saveState(IRecorder* simRecorder);
        virtual void setupSim(SimulationInfo *sim_info, IRecorder* simRecorder);
        virtual void cleanupSim(SimulationInfo *sim_info);
        virtual IAllNeurons* getNeurons();
        virtual Connections* getConnections();
        virtual Layout* getLayout();
        virtual void updateHistory(const SimulationInfo *sim_info, IRecorder* simRecorder);

    protected:

        /* -----------------------------------------------------------------------------------------
         * # Helper Functions
         * ------------------
         */

        // # Print Parameters
        // ------------------

        // # Save State
        // ------------
	void logSimStep(const SimulationInfo *sim_info) const;

        // -----------------------------------------------------------------------------------------
        // # Generic Functions for handling synapse types
        // ---------------------------------------------

        // Tracks the number of parameters that have been read by read params -
        // kind of a hack to do error handling for read params
        int m_read_params;

        // TODO
        Connections *m_conns;

        //
        IAllNeurons *m_neurons;

        //
        IAllSynapses *m_synapses;

        // 
        Layout *m_layout;

        //
        SynapseIndexMap *m_synapseIndexMap;

    private:
        /**
         * Populate an instance of IAllNeurons with an initial state for each neuron.
         *
         * @param sim_info - parameters defining the simulation to be run with the given collection of neurons.
         */
        void createAllNeurons(SimulationInfo *sim_info);

};
