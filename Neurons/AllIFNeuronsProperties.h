/**
 *      @file AllIFNeuronsProperties.h
 *
 *      @brief A container of the base class of all neuron data
 */

#pragma once 

#include "AllSpikingNeuronsProperties.h"

class AllIFNeuronsProperties : public AllSpikingNeuronsProperties
{
    public:
        AllIFNeuronsProperties();
        virtual ~AllIFNeuronsProperties();

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
         *  The length of the absolute refractory period. [units=sec; range=(0,1);]
         */
        BGFLOAT *Trefract;

        /**
         *  If \f$V_m\f$ exceeds \f$V_{thresh}\f$ a spike is emmited. [units=V; range=(-10,100);]
         */
        BGFLOAT *Vthresh;

        /**
         *  The resting membrane voltage. [units=V; range=(-1,1);]
         */
        BGFLOAT *Vrest;

        /**
         *  The voltage to reset \f$V_m\f$ to after a spike. [units=V; range=(-1,1);]
         */
        BGFLOAT *Vreset;

        /**
         *  The initial condition for \f$V_m\f$ at time \f$t=0\f$. [units=V; range=(-1,1);]
         */
        BGFLOAT *Vinit;

        /**
         *  The membrane capacitance \f$C_m\f$ [range=(0,1); units=F;]
         *  Used to initialize Tau (no use after that)
         */
        BGFLOAT *Cm;

        /**
         *  The membrane resistance \f$R_m\f$ [units=Ohm; range=(0,1e30)]
         */
        BGFLOAT *Rm;

        /**
         * The standard deviation of the noise to be added each integration time constant. [range=(0,1); units=A;]
         */
        BGFLOAT *Inoise;

        /**
         *  A constant current to be injected into the LIF neuron. [units=A; range=(-1,1);]
         */
        BGFLOAT *Iinject;

        /**
         * What the hell is this used for???
         *  It does not seem to be used; seems to be a candidate for deletion.
         *  Possibly from the old code before using a separate summation point
         *  The synaptic input current.
         */
        BGFLOAT *Isyn;

        /**
         * The remaining number of time steps for the absolute refractory period.
         */
        int *nStepsInRefr;

        /**
         * Internal constant for the exponential Euler integration of f$V_m\f$.
         */
        BGFLOAT *C1;

        /**
         * Internal constant for the exponential Euler integration of \f$V_m\f$.
         */
        BGFLOAT *C2;

        /**
         * Internal constant for the exponential Euler integration of \f$V_m\f$.
         */
        BGFLOAT *I0;

        /**
         * The membrane voltage \f$V_m\f$ [readonly; units=V;]
         */
        BGFLOAT *Vm;

        /**
         * The membrane time constant \f$(R_m \cdot C_m)\f$
         */
        BGFLOAT *Tau;
};
