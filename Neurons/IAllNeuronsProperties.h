/**
 *      @file IAllNeuronsProperties.h
 *
 *      @brief An interface for neuron properties class.
 */

#pragma once

#include "SimulationInfo.h"
#include "ClusterInfo.h"

class IAllNeuronsProperties
{
    public:
        virtual ~IAllNeuronsProperties() {};

        /**
         *  Setup the internal structure of the class.
         *  Allocate memories to store all neurons' state.
         *
         *  @param  sim_info  SimulationInfo class to read information from.
         *  @param  clr_info  ClusterInfo class to read information from.
         */
        virtual void setupNeuronsProperties(SimulationInfo *sim_info, ClusterInfo *clr_info) = 0;
};
