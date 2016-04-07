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
 * essential components of the spiking neunal network.
 *    -# IAllNeurons: A class to define a list of partiular type of neurons.
 *    -# IAllSynapses: A class to define a list of partiular type of synapses.
 *    -# Connections: A class to define connections of the neunal network.
 *    -# Layout: A class to define neurons' layout information in the network.
 *
 * \image html bg_data_layout.png
 *
 * The network is composed of 3 superimposed 2-d arrays: neurons, synapses, and
 * summation points.
 *
 * Synapses in the synapse map are located at the coordinates of the neuron
 * from which they receive output.  Each synapse stores a pointer into a
 * summation point. 
 * 
 * If, during an advance cycle, a neuron \f$A\f$ at coordinates \f$x,y\f$ fires, every synapse
 * which receives output is notified of the spike. Those synapses then hold
 * the spike until their delay period is completed.  At a later advance cycle, once the delay
 * period has been completed, the synapses apply their PSRs (Post-Synaptic-Response) to 
 * the summation points.  
 * Finally, on the next advance cycle, each neuron \f$B\f$ adds the value stored
 * in their corresponding summation points to their \f$V_m\f$ and resets the summation points to
 * zero.
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

class Model : public IModel
{
    public:
        Model(Connections *conns, IAllNeurons *neurons, IAllSynapses *synapses, Layout *layout);
        virtual ~Model();

        /**
         * Deserializes internal state from a prior run of the simulation.
         * This allows simulations to be continued from a particular point, to be restarted, or to be
         * started from a known state.
         *
         *  @param  input       istream to read from.
         *  @param  sim_info    used as a reference to set info for neurons and synapses.
         */
        virtual void deserialize(istream& input, const SimulationInfo *sim_info);

        /**
         * Serializes internal state for the current simulation.
         * This allows simulations to be continued from a particular point, to be restarted, or to be
         * started from a known state.
         *
         *  @param  output          The filestream to write.
         *  @param  simulation_step The step of the simulation at the current time.
         */
        virtual void serialize(ostream& output, const SimulationInfo *sim_info);

        /**
         * Writes simulation results to an output destination.
         *
         *  @param  sim_info    parameters for the simulation.
         */
        virtual void saveData(SimulationInfo *sim_info);

        /**
         * Set up model state, if anym for a specific simulation run.
         *
         * @param sim_info - parameters defining the simulation to be run with the given collection of neurons.
         * @param simRecorder    Pointer to the simulation recordig object.
         */
        virtual void setupSim(SimulationInfo *sim_info);

        /**
         * Performs any finalization tasks on network following a simulation.
         *
         * @param sim_info - parameters defining the simulation to be run with the given collection of neurons.
         */
        virtual void cleanupSim(SimulationInfo *sim_info);

        /**
         *  Get the IAllNeurons class object.
         *
         *  @return Pointer to the AllNeurons class object.
         */
        virtual IAllNeurons* getNeurons();

        /**
         *  Get the Connections class object.
         *
         *  @return Pointer to the Connections class object.
         */
        virtual Connections* getConnections();

        /**
         *  Get the Layout class object.
         *
         *  @return Pointer to the Layout class object.
         */
        virtual Layout* getLayout();

        /**
         *  Update the simulation history of every epoch.
         *
         *  @param  sim_info    SimulationInfo to refer from.
         */
        virtual void updateHistory(const SimulationInfo *sim_info);

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
