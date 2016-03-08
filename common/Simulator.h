/**
 * @file Simulator.h
 *
 * @authors Derek McLean & Sean Blackbourn
 *
 * @brief Platform independent base class for the Brain Grid simulator.
 */

#ifndef _SIMULATOR_H_
#define _SIMULATOR_H_

#include "Global.h"
#include "SimulationInfo.h"
#include "IModel.h"
#include "ISInput.h"

#include "Timer.h"

/**
 * @class Simulator Simulator.h "Simulator.h"
 *
 *
 * This class should be extended when developing the simulator for a specific platform.
 *
 * \latexonly  \subsubsection*{Credits} \endlatexonly
 * \htmlonly   <h3>Credits</h3> \endhtmlonly
 *
 * Some models in this simulator is a rewrite of CSIM (2006) and other
 * work (Stiber and Kawasaki (2007?))
 */
class Simulator
{
    public:

       /**
        *  Constructor
        *
        *  @param  model
        *          pointer to a neural network implementation to be simulated by BrainGrid. (It would be
        *          nice if this was a parameter to #simulate). Note: this reference will not be deleted.
        *  @simRecorder        Pointer to the simulation recordig object.
        *  @sInput             Pointer to the stimulus input object.
        *  @param  sim_info    parameters for the simulation.
        */
	Simulator(IModel *model, IRecorder* simRecorder, ISInput *sInput, SimulationInfo *sim_info);

        /** Destructor */
        virtual ~Simulator();

        /**
         * Setup simulation.
         */
        void setup();

        /**
         * Cleanup after simulation.
         */
        void finish();

        /** 
         * Reset simulation objects.
         */
        void reset();

        /**
         * Performs the simulation.
         */
        void simulate();

        /**
         * Advance simulation to next growth cycle. Helper for #simulate().
         *
         * @param currentStep the current epoch in which the network is being simulated.
         */
        void advanceUntilGrowth(const int currentStep);

        /**
         * Writes simulation results to an output destination.
         */
        void saveData() const;

        /**
         * Read serialized internal state from a previous run of the simulator.
         * This allows simulations to be continued from a particular point, to be restarted, or to be
         * started from a known state.
         *
         * @param memory_in - where to read the state from.
         */
        void deserialize(istream &memory_in);

        /**
         * Serializes internal state for the current simulation.
         * This allows simulations to be continued from a particular point, to be restarted, or to be
         * started from a known state.
         *
         * @param memory_out - where to write the state to.
         * This method needs to be debugged to verify that it works.
         */
        void serialize(ostream &memory_out) const;

    private:
        /**
         * Frees dynamically allocated memory associated with the maps.
         */
        void freeResources();

        /**
         * Timer for measuring performance of an epoch.
         */
        Timer timer;
        /**
         * Timer for measuring performance of connection update.
         */
        Timer short_timer;

        /**
         * Parameters for the simulation.
         */
        SimulationInfo *m_sim_info;

        /**
         * Pointer to the Neural Network Model interface.
         */
        IModel *m_model;

        /**
         * Pointer to the simulation recordig object.
         */
        IRecorder* m_simRecorder;

        /**
         * Pointer to the stimulus input object.
         */
        ISInput* m_sInput;
};

#endif /* _SIMULATOR_H_ */
