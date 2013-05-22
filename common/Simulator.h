/**
 *  @file Simulator.h
 *  @authors Derek McLean
 *  @brief Interface for model-independent simulators targeting different 
 *  platforms.
 */

#pragma once

#ifndef _SIMULATOR_H_
#define _SIMULATOR_H_

#include <iostream>

using namespace std;

/**
 * @class Simulator
 *
 * This class provides an interface and common logic for Simulators running on
 * different platforms.
 * As of the current version, this class is a staging area of extracting into
 * a common location core simulator code.
 */

class Simulator
{
    public:
        virtual ~Simulator() {};
        
        /**
         *  Performs the simulation.
         */
        virtual void simulate() =0;
        
        /**
         *  Advance simulation to next growth cycle.
         *  @param  currentStep the current epoch of the simulation.
         */
        virtual void advanceUntilGrowth(const int currentStep) =0;

        /**
         *  Write the result of the simulation.
         *  @param  state_out   ostream object to write out to.
         */
        virtual void saveState(ostream &state_out) const =0;

        /**
         *  Read serialized internal state from a previous run of the 
         *  simulator.
         *  @param  memory_in   istream object to read data in from.
         */
        virtual void readMemory(istream &memory_in) =0;

        /**
         *  Write current internal state of the simulator.
         *  @param  memory_out  ostream object to save data to.
         */
        virtual void saveMemory(ostream &memory_out) const =0;
};

#endif // _SIMULATOR_H_

