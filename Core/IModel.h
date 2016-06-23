/**
 * @brief An interface for Neural Network Models.
 *
 * @class IModel IModel.h "IModel.h"
 *
 * @author Derek L. Mclean
 */

#pragma once
#ifndef _IMODEL_H_
#define _IMODEL_H_

#include <iostream>

using namespace std;

#include "tinyxml.h"

#include "Global.h"
#include "IAllNeurons.h"
#include "IAllSynapses.h"
#include "SimulationInfo.h"
#include "IRecorder.h"
#include "Connections.h"
#include "Layout.h"

/**
 * Neural Network Model interface.
 *
 * Implementations define behavior of the network specific to the model. Specifically, a model
 * implementation handles:
 * * I/O
 * * Network creation
 * * Network simulation
 *
 * It is recommended that mutations of model state, if any, are avoided during a simulation. Some
 * models, such as those with complex connection dynamics or network history, may need to modify an
 * internal state during simulation.
 *
 * This is a pure interface and, thus, not meant to be directly instanced.
 */
class IModel {
    public:
        virtual ~IModel() { }

        /* --------------------
         * # Network IO Methods
         * --------------------
         */

        /**
         * Deserializes internal state from a prior run of the simulation.
         * This allows simulations to be continued from a particular point, to be restarted, or to be
         * started from a known state.
         *
         *  @param  input       istream to read from.
         *  @param  sim_info    used as a reference to set info for neurons and synapses.
         */
        virtual void deserialize(istream& input, const SimulationInfo *sim_info) = 0;

        /**
         * Serializes internal state for the current simulation.
         * This allows simulations to be continued from a particular point, to be restarted, or to be
         * started from a known state.
         *
         *  @param  output          The filestream to write.
         *  @param  simulation_step The step of the simulation at the current time.
         */
        virtual void serialize(ostream& output, const SimulationInfo *sim_info) = 0;

        /**
         * Writes simulation results to an output destination.
         *
         *  @param  sim_info    parameters for the simulation. 
         */
        virtual void saveData(SimulationInfo *sim_info) = 0;

        /* ----------------
         * Network Creation
         * ----------------
         */

        /* --------------------------
         * Network Simulation Methods
         * --------------------------
         */

        /**
         * Set up model state, if anym for a specific simulation run.
         *
         * @param sim_info - parameters defining the simulation to be run with the given collection of neurons.
         */
        virtual void setupSim(SimulationInfo *sim_info) = 0;

        /**
         * Advances network state one simulation step.
         *
         * @param sim_info - parameters defining the simulation to be run with the given collection of neurons.
         */
        virtual void advance(const SimulationInfo *sim_info) = 0;

        /**
         * Modifies connections between neurons based on current state of the network and behavior
         * over the past epoch. Should be called once every epoch.
         *
         * @param currentStep - The epoch step in which the connections are being updated.
         * @param sim_info - parameters defining the simulation to be run with the given collection of neurons.
         */
        virtual void updateConnections(const SimulationInfo *sim_info) = 0;

        /**
         * Performs any finalization tasks on network following a simulation.
         *
         * @param sim_info - parameters defining the simulation to be run with the given collection of neurons.
         */
        virtual void cleanupSim(SimulationInfo *sim_info) = 0;

        /**
         * Prints debug information about the current state of the network.
         *
         * @param sim_info - parameters defining the simulation to be run with the given collection of neurons.
         */
        virtual void logSimStep(const SimulationInfo *sim_info) const = 0;

        /**
         *  Get the IAllNeurons class object.
         *
         *  @return Pointer to the AllNeurons class object.
         */
        virtual IAllNeurons* getNeurons() = 0;

        /**
         *  Get the Connections class object.
         *
         *  @return Pointer to the Connections class object.
         */
        virtual Connections* getConnections() = 0;

        /**
         *  Get the Layout class object.
         *
         *  @return Pointer to the Layout class object.
         */
        virtual Layout* getLayout() = 0;

        /**
         *  Update the simulation history of every epoch.
         *
         *  @param  sim_info    SimulationInfo to refer from.
         */
        virtual void updateHistory(const SimulationInfo *sim_info) = 0;
};

#endif
