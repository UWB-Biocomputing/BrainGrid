#pragma once

#include "IModel.h"
#include "Coordinate.h"
#include "Connections.h"
#include "Layout.h"

#include <vector>
#include <iostream>

using namespace std;

/**
 * Implementation of Model for the Leaky-Integrate-and-Fire model.
 */
class Model : public IModel, TiXmlVisitor
{
    public:
        Model(Connections *conns, AllNeurons *neurons, AllSynapses *synapses, Layout *layout);
        virtual ~Model();

        /*
         * Declarations of concrete implementations of Model interface for an Leaky-Integrate-and-Fire
         * model.
         *
         * @see Model.h
         */

        virtual bool readParameters(TiXmlElement *source);
        virtual void printParameters(ostream &output) const;
        virtual void loadMemory(istream& input, const SimulationInfo *sim_info);
        virtual void saveMemory(ostream& output, const SimulationInfo *sim_info);
        virtual void saveState(IRecorder* simRecorder);
        virtual void setupSim(SimulationInfo *sim_info, IRecorder* simRecorder);
        virtual void cleanupSim(SimulationInfo *sim_info);
        virtual AllNeurons* getNeurons();
        virtual Connections* getConnections();
        virtual Layout* getLayout();

    protected:

        /* -----------------------------------------------------------------------------------------
         * # Helper Functions
         * ------------------
         */

        // # Read Parameters
        // -----------------

        // Parse an element for parameter values.
        // Required by TiXmlVisitor, which is used by #readParameters
        bool VisitEnter(const TiXmlElement& element, const TiXmlAttribute* firstAttribute);

        // # Print Parameters
        // ------------------

        // # Save State
        // ------------
	void logSimStep(const SimulationInfo *sim_info) const;

	// TODO     
        void updateHistory(int currentStep, BGFLOAT epochDuration, const SimulationInfo *sim_info, IRecorder* simRecorder);

        // -----------------------------------------------------------------------------------------
        // # Generic Functions for handling synapse types
        // ---------------------------------------------

        // Tracks the number of parameters that have been read by read params -
        // kind of a hack to do error handling for read params
        int m_read_params;

        // TODO
        Connections *m_conns;

        //
        AllNeurons *m_neurons;

        //
        AllSynapses *m_synapses;

        // 
        Layout *m_layout;

    private:
        /**
         * Populate an instance of AllNeurons with an initial state for each neuron.
         *
         * @param sim_info - parameters defining the simulation to be run with the given collection of neurons.
         */
        void createAllNeurons(SimulationInfo *sim_info);

};
