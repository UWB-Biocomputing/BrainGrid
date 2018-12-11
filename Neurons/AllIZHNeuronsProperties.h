/**
 *      @file AllIZHNeuronsProperties.h
 *
 *      @brief A container of the base class of all neuron data
 */

#pragma once 

#include "AllIFNeuronsProperties.h"

class AllIZHNeuronsProperties : public AllIFNeuronsProperties
{
    public:
        AllIZHNeuronsProperties();
        virtual ~AllIZHNeuronsProperties();

        /**
         *  Setup the internal structure of the class.
         *  Allocate memories to store all neurons' state.
         *
         *  @param  sim_info  SimulationInfo class to read information from.
         *  @param  clr_info  ClusterInfo class to read information from.
         */
        virtual void setupNeuronsProperties(SimulationInfo *sim_info, ClusterInfo *clr_info);

    protected:
        /**
         *  Cleanup the class.
         *  Deallocate memories.
         */
        void cleanupNeuronsProperties();

    public:
        /**
         *  A constant (0.02, 01) describing the coupling of variable u to Vm.
         */
        BGFLOAT *Aconst;

        /**
         *  A constant controlling sensitivity of u.
         */
        BGFLOAT *Bconst;

        /**
         *  A constant controlling reset of Vm.
         */
        BGFLOAT *Cconst;

        /**
         *  A constant controlling reset of u.
         */
        BGFLOAT *Dconst;

        /**
         *  internal variable.
         */
        BGFLOAT *u;

        /**
         *  Internal constant for the exponential Euler integration.
         */
        BGFLOAT *C3;
};
