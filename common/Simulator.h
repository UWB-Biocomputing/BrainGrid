/**
 * @file Simulator.h
 *
 * @authors Derek McLean & Sean Blackbourn
 *
 * @brief Abstract base class for BrainGrid simulator for different platforms
 */

#ifndef _SIMULATOR_H_
#define _SIMULATOR_H_

#include "Global.h"
#include "SimulationInfo.h"
#include "Network.h"

#include "Timer.h"

/**
 * @class Simulator
 *
 * @brief Platform independent base class for the Brain Grid simulator.
 *
 * This class should be extended when developing the simulator for a specific platform.
 */
class Simulator
{
    public:

	 Simulator(Network *network, SimulationInfo *sim_info);
        /** Destructor */
        virtual ~Simulator() = 0;

        /**
         * Performs the simulation.
         */
        void simulate();

        /**
         * Advance simulation to next growth cycle. Helper for #simulate().
         */
        void advanceUntilGrowth(const int currentStep);

        /**
         * Write the result of the simulation.
         */
        void saveState(ostream &state_out) const;

        /**
         * Read serialized internal state from a previous run of the simulator.
         */
        void readMemory(istream &memory_in);
        /**
         * Write current internal state of the simulator.
         */
        void saveMemory(ostream &memory_out) const;

    protected:
        /**
         * Timer for measuring performance of an epoch.
         */
        Timer timer;
        /**
         * Timer for measuring performance of connection update.
         */
        Timer short_timer;

        /**
         * The network being simulated.
         */
        Network *network;

        /**
         * Parameters for the simulation.
         */
        SimulationInfo *m_sim_info;
};

#endif /* _SIMULATOR_H_ */
