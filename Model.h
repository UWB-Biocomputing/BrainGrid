/**
 * @brief An interface for Neural Network Models.
 *
 * @class Model Model.h "Model.h"
 *
 * @author Derek L. Mclean
 */

#pragma once
#ifndef _MODEL_H_
#define _MODEL_H_

#include <iostream>

using namespace std;

#include "include/tinyxml.h"

#include "global.h"
#include "AllNeurons.h"
#include "AllSynapses.h"
#include "SimulationInfo.h"

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
class Model {
    public:
        virtual ~Model() { }

        /**
         * +--------------------+
         * | Network IO Methods |
         * +--------------------+
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
        virtual void loadMemory(istream& input, AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo &sim_info) =0;

        /**
         * TODO(derek) comment.
         */
        virtual void saveMemory(ostream& output, AllNeurons &neurons, AllSynapses &synapses, BGFLOAT simulation_step) =0;

        /**
         * TODO(derek) comment.
         */
        virtual void saveState(ostream& output, const AllNeurons &neurons, const SimulationInfo &sim_info) =0;

        /**
         * +------------------+
         * | Network Creation |
         * +------------------+
         */

        /**
         * Populate an instance of AllNeurons with an initial state for each neuron.
         *
         * @param neurons - collection of neurons to populate.
         * @param sim_info - parameters defining the simulation to be run with the given collection of neurons.
         */
        virtual void createAllNeurons(AllNeurons &neurons, const SimulationInfo &sim_info) =0;

        /**
         * +----------------------------+
         * | Network Simulation Methods |
         * +----------------------------+
         */

        /**
         * Set up model state, if anym for a specific simulation run.
         *
         * @param num_neurons - count of neurons in network
         * @param sim_info - parameters defining the simulation to be run with the given collection of neurons.
         */
        virtual void setupSim(const int num_neurons, const SimulationInfo &sim_info) =0;

        /**
         * Advances network state one simulation step.
         *
         * @param neurons - collection of neurons in network
         * @param synapses - collection of connections between neurons in network.
         * @param sim_info - parameters defining the simulation to be run with the given collection of neurons.
         */
        virtual void advance(AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo &sim_info) =0;

        /**
         * Modifies connections between neurons based on current state of the network and behavior
         * over the past epoch. Should be called once every epoch.
         *
         * @param currentStep - The epoch step in which the connections are being updated.
         * @param neurons - collection of neurons in network
         * @param synapses - collection of connections between neurons in network.
         * @param sim_info - parameters defining the simulation to be run with the given collection of neurons.
         */
        virtual void updateConnections(const int currentStep, AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo &sim_info) =0;

        /**
         * Performs any finalization tasks on network following a simulation.
         * @param neurons - collection of neurons in network
         * @param sim_info - parameters defining the simulation to be run with the given collection of neurons.
         */
        virtual void cleanupSim(AllNeurons &neurons, SimulationInfo &sim_info) =0;

        /**
         * Prints debug information about the current state of the network.
         *
         * @param neurons - collection of neurons in network
         * @param synapses - collection of connections between neurons in network.
         * @param sim_info - parameters defining the simulation to be run with the given collection of neurons.
         */
        virtual void logSimStep(const AllNeurons &neurons, const AllSynapses &synapses, const SimulationInfo &sim_info) const =0;
};

#endif
