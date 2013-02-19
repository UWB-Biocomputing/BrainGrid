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
#include "NetworkUpdater.h"

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
        Simulator(Network *network, SimulationInfo sim_info,
            bool write_mem_image,
            ostream& memory_out
        );
        
        //Performs the simulation.
        void simulate(FLOAT growthStepDuration, FLOAT maxGrowthSteps);
        
        // Advance simulation to next growth cycle
        void advanceUntilGrowth(const int currentStep, const int maxGrowthSteps);
    
    private:
        Timer timer;
        Timer short_timer;
        
        Network *network;
        NetworkUpdater updater;
        
        SimulationInfo sim_info;
        
        //! True if dumped memory image is written after simulation. 
        bool write_mem_image;
        ostream& memory_out;
};

#endif // _SIMULATOR_H_

