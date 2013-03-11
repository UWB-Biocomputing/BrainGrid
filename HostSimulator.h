/*
 * HostSimulator.h
 *
 *  Created on: 2013-03-06
 *      Author: derek
 */

#ifndef _HOSTSIMULATOR_H_
#define _HOSTSIMULATOR_H_

#include "Simulator.h"

#include "global.h"
#include "SimulationInfo.h"
#include "Network.h"

#include "include/Timer.h"

class HostSimulator: public Simulator
{
    public:
        HostSimulator(Network *network, SimulationInfo sim_info);
        virtual ~HostSimulator();

        //Performs the simulation.
        void simulate();

        // Advance simulation to next growth cycle
        void advanceUntilGrowth(const int currentStep);

        void saveState(ostream &state_out) const;

        void readMemory(istream &memory_in);
        void saveMemory(ostream &memory_out) const;

    private:
        Timer timer;
        Timer short_timer;

        Network *network;

        SimulationInfo m_sim_info;
};

#endif /* _HOSTSIMULATOR_H_ */
