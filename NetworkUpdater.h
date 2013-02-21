/**
 * @file NetworkUpdater.h
 *
 * @authors Derek McLean
 *
 * @brief Base class for model-independent simulators targeting different
 *          platforms.
 */

#pragma once

#ifndef _NETWORKUPDATER_H_
#define _NETWORKUPDATER_H_

#include "global.h"
#include "SimulationInfo.h"
#include "Network.h"
#include "include/Timer.h"

/**
 * @class NetworkUpdater
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

class NetworkUpdater
{
    public:
        NetworkUpdater(const int neuron_count, const FLOAT start_radius);
        
        void update(int currentStep, Network *network, SimulationInfo *sim_info);
        
        int *spikeCounts;
        
        //! synapse weight
        CompleteMatrix W;

        //! neuron radii
        VectorMatrix radii;

        //! spiking rate
        VectorMatrix rates;

        //! Inter-neuron distance squared
        CompleteMatrix dist2;

        //! distance between connection frontiers
        CompleteMatrix delta;

        //! the true inter-neuron distance
        CompleteMatrix dist;

        //! areas of overlap
        CompleteMatrix area;

        //! neuron's outgrowth
        VectorMatrix outgrowth;

        //! displacement of neuron radii
        VectorMatrix deltaR;
        
        // Helper Functions
        
        void updateHistory(int currentStep, Network *network, SimulationInfo *sim_info);
        void updateRadii(Network *network, SimulationInfo *sim_info);
        void updateFrontiers(SimulationInfo *sim_info);
        void updateOverlap(SimulationInfo *sim_info);
        void updateWeights(SimulationInfo* sim_info);
};

#endif
