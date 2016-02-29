/**
 *	@file Network.h
 *
 *
 *	@brief A collection of Neurons and their connecting synapses.
 */

/**
 ** \class Network Network.h "Network.h"
 **
 **\latexonly  \subsubsection*{Implementation} \endlatexonly
 ** \htmlonly  <h3>Implementation</h3> \endhtmlonly
 **
 ** \image html bg_data_layout.png
 **
 ** The network is composed of 3 superimposed 2-d arrays: neurons, synapses, and
 ** summation points.
 **
 ** Synapses in the synapse map are located at the coordinates of the neuron
 ** from which they receive output.  Each synapse stores a pointer into a
 ** summation point. 
 ** 
 ** If, during an advance cycle, a neuron \f$A\f$ at coordinates \f$x,y\f$ fires, every synapse
 ** which receives output is notified of the spike. Those synapses then hold
 ** the spike until their delay period is completed.  At a later advance cycle, once the delay
 ** period has been completed, the synapses apply their PSRs (Post-Synaptic-Response) to 
 ** the summation points.  Finally, on the next advance cycle, each neuron \f$B\f$ adds the value stored
 ** in their corresponding summation points to their \f$V_m\f$ and resets the summation points to
 ** zero.
 **
 ** \latexonly \subsubsection*{Credits} \endlatexonly
 ** \htmlonly <h3>Credits</h3> \endhtmlonly
 **
 ** Some models in this simulator is a rewrite of CSIM (2006) and other 
 ** work (Stiber and Kawasaki (2007?))
 **
 **
 **	@authors Allan Ortiz and Cory Mayberry
 **/

#ifndef _NETWORK_H_
#define _NETWORK_H_

#include "Global.h"
#include "Timer.h"

#include "IModel.h"
#include "ISInput.h"

class Network
{
    public:
        //! The constructor for Network.
        Network(IModel *model, SimulationInfo *sim_info, IRecorder* simRecorder);
        ~Network();

        //! Frees dynamically allocated memory associated with the maps.
        void freeResources();

        //! Reset simulation objects
        void reset();

        //! Writes simulation results to an output destination.
        void saveData();

        /**
         * Serializes internal state for the current simulation.
         * This allows simulations to be continued from a particular point, to be restarted, or to be
         * started from a known state.
         *
         * @param  os  The filestream to write.
         * This method needs to be debugged to verify that it works.
         */
        void serialize(ostream& os);

        /**
         * Deserializes internal state from a prior run of the simulation.
         * This allows simulations to be continued from a particular point, to be restarted, or to be
         * started from a known state.
         *
         * @param  is  the filestream to read.
         */
        void deserialize(istream& is);

        // Setup simulation
        void setup(ISInput* pInput);

        // Cleanup after simulation
        void finish();

        /**
         * Advance the network one step in an epoch.
         */
        void advance(ISInput* pInput);

        /**
         * Performs growth in the network: updating connections between neurons for the current epoch.
         */
        void updateConnections();

        void updateHistory();

        //! Print network radii to console.
        void logSimStep() const;

        // TODO comment
        IModel *m_model;
        // TODO comment
        IRecorder* m_simRecorder;

    private:
        // Struct that holds information about a simulation
        SimulationInfo *m_sim_info;
    
        Network(); // DOES THIS NEED TO BE DEFINED?
};
#endif // _NETWORK_H_
