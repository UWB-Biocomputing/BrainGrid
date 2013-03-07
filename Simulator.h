/**
 * @file Simulator.h
 *
 * @authors Derek McLean
 *
 * @brief Base class for model-independent simulators targeting different
 *          platforms.
 */

#pragma once

#ifndef _SIMULATOR_H_
#define _SIMULATOR_H_

#include "global.h"
#include "SimulationInfo.h"
#include "Network.h"

#include "include/Timer.h"

/**
 * @class Simulator
 *
 * This class provides an interface and common logic for Simulators running on
 * different platforms.
 *
 * As of the current version, this class is a staging area of extracting into
 * a common location core simulator code.
 *
 * @todo link this code back into the ISimulator hierarchy.
 *
 * @version 0.1
 */

class Simulator
{
    public:
        Simulator(Network *network, SimulationInfo sim_info);
        
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

#endif // _SIMULATOR_H_

