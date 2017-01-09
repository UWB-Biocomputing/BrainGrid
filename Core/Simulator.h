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

// Home-brewed performance measurement
#ifdef PERFORMANCE_METRICS
#include "Timer.h"
#endif

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
        */
	Simulator();

        /** Destructor */
        virtual ~Simulator();

        /**
         * Setup simulation.
         *
         *  @param  sim_info    parameters for the simulation.
         */
        void setup(SimulationInfo *sim_info);

        /**
         * Cleanup after simulation.
         *
         *  @param  sim_info    parameters for the simulation.
         */
        void finish(SimulationInfo *sim_info);

        /** 
         * Reset simulation objects.
         *
         *  @param  sim_info    parameters for the simulation.
         */
        void reset(SimulationInfo *sim_info);

        /**
         * Performs the simulation.
         *
         *  @param  sim_info    parameters for the simulation.
         */
        void simulate(SimulationInfo *sim_info);

        /**
         * Advance simulation to next growth cycle. Helper for #simulate().
         *
         *  @param currentStep the current epoch in which the network is being simulated.
         *  @param  sim_info    parameters for the simulation.
         */
        void advanceUntilGrowth(const int currentStep, SimulationInfo *sim_info);

        /**
         * Writes simulation results to an output destination.
         *
         *  @param  sim_info    parameters for the simulation.
         */
        void saveData(SimulationInfo *sim_info) const;

        /**
         * Read serialized internal state from a previous run of the simulator.
         * This allows simulations to be continued from a particular point, to be restarted, or to be
         * started from a known state.
         *
         *  @param memory_in - where to read the state from.
         *  @param  sim_info    parameters for the simulation.
         */
        void deserialize(istream &memory_inn, SimulationInfo *sim_info);

        /**
         * Serializes internal state for the current simulation.
         * This allows simulations to be continued from a particular point, to be restarted, or to be
         * started from a known state.
         * This method needs to be debugged to verify that it works.
         *
         *  @param memory_out - where to write the state to.
         *  @param  sim_info    parameters for the simulation.
         */
        void serialize(ostream &memory_out, SimulationInfo *sim_info) const;

    private:
        /**
         * Frees dynamically allocated memory associated with the maps.
         */
        void freeResources();

#ifdef PERFORMANCE_METRICS
        /**
         * Timer for measuring performance of an epoch.
         */
        Timer timer;
        /**
         * Timer for measuring performance of connection update.
         */
        Timer short_timer;
#endif
};

#endif /* _SIMULATOR_H_ */
