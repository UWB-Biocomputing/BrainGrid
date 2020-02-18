/**
 * @file Simulator.h
 *
 * @authors Derek McLean & Sean Blackbourn
 *
 * @brief Platform independent base class for the Brain Grid simulator.
 */

#ifndef _SIMULATOR_H_
#define _SIMULATOR_H_

#include "Global.h"
#include "SimulationInfo.h"
#include "IModel.h"
#include "ISInput.h"

/**
 * @class Simulator Simulator.h "Simulator.h"
 *
 *
 * This class should be extended when developing the simulator for a specific platform.
 *
 * \latexonly  \subsubsection*{Credits} \endlatexonly
 * \htmlonly   <h3>Credits</h3> \endhtmlonly
 *
 * Some models in this simulator is a rewrite of CSIM (2006) and other
 * work (Stiber and Kawasaki (2007?))
 */
class Simulator
{
    public:

       /**
        *  Constructor
        */
	Simulator();

        /** Destructor */
        virtual ~Simulator();

        /**
         * Setup simulation.
         *
         *  @param  sim_info    parameters for the simulation.
         */
        void setup(SimulationInfo *sim_info);

        /**
         * Cleanup after simulation.
         *
         *  @param  sim_info    parameters for the simulation.
         */
        void finish(SimulationInfo *sim_info);
      
        /**
         * Copy GPU Synapse data to CPU.
         *
         *  @param  sim_info    parameters for the simulation.
         */
        void copyGPUSynapseToCPU(SimulationInfo *sim_info);

        /**
         * Copy CPU Synapse data to GPU.
         *
         *  @param  sim_info    parameters for the simulation.
         */
        void copyCPUSynapseToGPU(SimulationInfo *sim_info);

        /** 
         * Reset simulation objects.
         *
         *  @param  sim_info    parameters for the simulation.
         */
        void reset(SimulationInfo *sim_info);

        /**
         * Performs the simulation.
         *
         *  @param  sim_info    parameters for the simulation.
         */
        void simulate(SimulationInfo *sim_info);

        /**
         * Advance simulation to next growth cycle. Helper for #simulate().
         *
         *  @param currentStep the current epoch in which the network is being simulated.
         *  @param  sim_info    parameters for the simulation.
         */
        void advanceUntilGrowth(const int currentStep, SimulationInfo *sim_info);

        /**
         * Writes simulation results to an output destination.
         *
         *  @param  sim_info    parameters for the simulation.
         */
        void saveData(SimulationInfo *sim_info) const;

    private:
        /**
         * Frees dynamically allocated memory associated with the maps.
         */
        void freeResources();
};

#endif /* _SIMULATOR_H_ */
