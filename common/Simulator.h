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
#include "Network.h"
#include "ISInput.h"

#include "Timer.h"

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

	 Simulator(Network *network, SimulationInfo *sim_info);
        /** Destructor */
        virtual ~Simulator() = 0;

        /**
         * Performs the simulation.
         */
        void simulate(ISInput* pInput);

        /**
         * Advance simulation to next growth cycle. Helper for #simulate().
         */
        void advanceUntilGrowth(const int currentStep, ISInput* pInput);

        /**
         * Write the result of the simulation.
         */
        void saveState() const;

        /**
         * Read serialized internal state from a previous run of the simulator.
         */
        void readMemory(istream &memory_in);
        /**
         * Write current internal state of the simulator.
         */
        void saveMemory(ostream &memory_out) const;

    protected:
        /**
         * Timer for measuring performance of an epoch.
         */
        Timer timer;
        /**
         * Timer for measuring performance of connection update.
         */
        Timer short_timer;

        /**
         * The network being simulated.
         */
        Network *network;

        /**
         * Parameters for the simulation.
         */
        SimulationInfo *m_sim_info;
};

#endif /* _SIMULATOR_H_ */
