/**
 *	@file Network.h
 *
 *
 *	@brief Header file for Network.
 */
//! A collection of Neurons and their connecting synapses.

/**
 ** \class Network Network.h "Network.h"
 **
 **\latexonly  \subsubsection*{Implementation} \endlatexonly
 ** \htmlonly  <h3>Implementation</h3> \endhtmlonly
 **
 ** \image html bg_data_layout.png
 **
 ** The network is composed of 3 superimposed 2-d arrays: neurons, synapses, and
 ** summation points (m_neuronList, m_rgSynapseMap, and m_summationMap).
 **
 ** Synapses in the synapse map are located at the coordinates of the neuron
 ** from which they receive output.  Each synapse stores a pointer into a
 ** m_summationMap bin.  Bins in the m_summationMap map directly to their output neurons.
 ** 
 ** If, during an advance cycle, a neuron \f$A\f$ at coordinates \f$x,y\f$ fires, every synapse
 ** \f$B\f$ at \f$x,y\f$ in the m_rgSynapseMap is notified of the spike. Those synapses then hold
 ** the spike until their delay period is completed.  At a later advance cycle, once the delay
 ** period has been completed, the synapses apply their PSRs (Post-Synaptic-Response) to their output bins \f$C\f$ in
 ** the m_summationMap.  Finally, on the next advance cycle, each neuron \f$D\f$ adds the value stored
 ** in their corresponding m_summationMap bin to their \f$V_m\f$ and resets the m_summationMap bin to
 ** zero.
 **
 ** \latexonly \subsubsection*{Credits} \endlatexonly
 ** \htmlonly <h3>Credits</h3> \endhtmlonly
 **
 ** This simulator is a rewrite of CSIM (2006) and other work (Stiber and Kawasaki (2007?))
 **
 **
 **	@authors Allan Ortiz and Cory Mayberry
 **/

#ifndef _NETWORK_H_
#define _NETWORK_H_

#include "Global.h"
#include "Timer.h"

#include "Model.h"

class Network
{
    public:
        //! The constructor for Network.
        Network(Model *model, SimulationInfo *sim_info);
        ~Network();

        //! Frees dynamically allocated memory associated with the maps.
        void freeResources();

        //! Reset simulation objects
        void reset();

        //! Write the network state to an ostream.
        void saveState(ostream& os);

        //! Write the simulation memory image to an ostream
        void writeSimMemory(BGFLOAT simulation_step, ostream& os);

        //! Read the simulation memory image from an istream
        void readSimMemory(istream& is);

        // Setup simulation
        void setup();

        // Cleanup after simulation
        void finish();

        /**
         * Advance the network one step in an epoch.
         */
        void advance();

        /**
         * Performs growth in the network: updating connections between neurons for the current epoch.
         */
        void updateConnections(const int currentStep);

        //! Print network radii to console.
        void logSimStep() const;

        // TODO comment
        Model *m_model;
        // TODO comment
        AllNeurons neurons;
        // TODO comment
        AllSynapses synapses;

        //! The map of summation points.
        BGFLOAT* m_summationMap;

    private:
        // Struct that holds information about a simulation
        SimulationInfo * m_pSim_info;
    
        Network(); // DOES THIS NEED TO BE DEFINED?
};
#endif // _NETWORK_H_
