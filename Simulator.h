/**
 * @file Simulator.h
 *
 * @authors Derek McLean
 *
 * @brief Base class for model-independent simulators targeting different platforms.
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
 *
 * As of the current version, this class is a staging area of extracting into
 * a common location core simulator code.
 */

class Simulator
{
    public:
        virtual ~Simulator() {};
        
        /**
         * Performs the simulation.
         */
        virtual void simulate() =0;
        
        /**
         * Advance simulation to next growth cycle.
         *
         * @param currentStep - the current epoch of the simulation.
         */
        virtual void advanceUntilGrowth(const int currentStep) =0;

        /**
         * TODO(derek) comment.
         */
        virtual void saveState(ostream &state_out) const =0;

        /**
         * TODO(derek) comment.
         */
        virtual void readMemory(istream &memory_in) =0;

        /**
         * TODO(derek) comment.
         */
        virtual void saveMemory(ostream &memory_out) const =0;
};

#endif // _SIMULATOR_H_

