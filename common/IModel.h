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

#include "../tinyxml/tinyxml.h"

#include "Global.h"
#include "AllNeurons.h"
#include "AllSynapses.h"
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
         * Read model specific parameters from the xml parameter file and finishes setting up model
         * state.
         *
         * @param source - the xml parameter document
         *
         * @return success of read (e.g. whether all parameters were read).
         */
        virtual bool readParameters(TiXmlElement *source) =0;

        /**
         * Writes model parameters to an output file. Parameters should be written in xml format.
         *
         * @param output - file to write to.
         */
        virtual void printParameters(ostream &output) const =0;

        /**
         * TODO(derek) comment.
         */
        virtual void loadMemory(istream& input, const SimulationInfo *sim_info) =0;

        /**
         * TODO(derek) comment.
         */
        virtual void saveMemory(ostream& output, const SimulationInfo *sim_info) =0;

        /**
         * TODO(derek) comment.
         */
        virtual void saveState(IRecorder* simRecorder) =0;

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
        virtual void setupSim(SimulationInfo *sim_info, IRecorder* simRecorder) =0;

        /**
         * Advances network state one simulation step.
         *
         * @param sim_info - parameters defining the simulation to be run with the given collection of neurons.
         */
        virtual void advance(const SimulationInfo *sim_info) =0;

        /**
         * Modifies connections between neurons based on current state of the network and behavior
         * over the past epoch. Should be called once every epoch.
         *
         * @param currentStep - The epoch step in which the connections are being updated.
         * @param sim_info - parameters defining the simulation to be run with the given collection of neurons.
         */
        virtual void updateConnections(const int currentStep, const SimulationInfo *sim_info, IRecorder* simRecorder) =0;

        /**
         * Performs any finalization tasks on network following a simulation.
         * @param sim_info - parameters defining the simulation to be run with the given collection of neurons.
         */
        virtual void cleanupSim(SimulationInfo *sim_info) =0;

        /**
         * Prints debug information about the current state of the network.
         *
         * @param sim_info - parameters defining the simulation to be run with the given collection of neurons.
         */
        virtual void logSimStep(const SimulationInfo *sim_info) const =0;

        virtual AllNeurons* getNeurons() = 0;

        virtual Connections* getConnections() = 0;

        virtual Layout* getLayout() = 0;
};

#endif
