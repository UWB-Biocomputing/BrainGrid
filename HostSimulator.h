/**
 * @file HostSimulator.h
 *
 * @authors Derek McLean
 *
 * @brief Abstract base class for BrainGrid simulator for different platforms
 */

#ifndef _HOSTSIMULATOR_H_
#define _HOSTSIMULATOR_H_

#include "Simulator.h"

#include "global.h"
#include "SimulationInfo.h"
#include "Network.h"

#include "include/Timer.h"

/**
 * @class HostSimulator
 * @implements Simulator
 *
 * @brief Platform independent base class for the Brain Grid simulator.
 *
 * This class should be extended when developing the simulator for a specific platform.
 */
class HostSimulator: public Simulator
{
    public:
        /**
         * Constructor
         * Constructs a simulator with a given network and given parameters.
         */
        HostSimulator(Network *network, SimulationInfo sim_info);
        /** Destructor */
        virtual ~HostSimulator();

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

    private:
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
        SimulationInfo m_sim_info;
};

#endif /* _HOSTSIMULATOR_H_ */
