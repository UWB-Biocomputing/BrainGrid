/**
 *      @file SingleThreadedSpikingModel.h
 *
 *      @brief Implementation of Model for the spiking neunal networks.
 */

/**
 * @class  SingleThreadedSpikingModel SingleThreadedSpikingModel.h "SingleThreadedSpikingModel.h"
 *
 * \latexonly  \subsubsection*{Implementation} \endlatexonly
 * \htmlonly   <h3>Implementation</h3> \endhtmlonly
 *
 * The Model class maintains and manages classes of objects that make up
 * essential components of the spiking neunal network.
 *    -# IAllNeurons: A class to define a list of partiular type of neurons.
 *    -# IAllSynapses: A class to define a list of partiular type of synapses.
 *    -# Connections: A class to define connections of the neunal network.
 *    -# Layout: A class to define neurons' layout information in the network.
 *
 * \image html bg_data_layout.png
 *
 * The network is composed of 3 superimposed 2-d arrays: neurons, synapses, and
 * summation points.
 *
 * Synapses in the synapse map are located at the coordinates of the neuron
 * from which they receive output.  Each synapse stores a pointer into a
 * summation point. 
 * 
 * If, during an advance cycle, a neuron \f$A\f$ at coordinates \f$x,y\f$ fires, every synapse
 * which receives output is notified of the spike. Those synapses then hold
 * the spike until their delay period is completed.  At a later advance cycle, once the delay
 * period has been completed, the synapses apply their PSRs (Post-Synaptic-Response) to 
 * the summation points.  
 * Finally, on the next advance cycle, each neuron \f$B\f$ adds the value stored
 * in their corresponding summation points to their \f$V_m\f$ and resets the summation points to
 * zero.
 *
 * The model runs on a single thread.
 *
 * \latexonly  \subsubsection*{Credits} \endlatexonly
 * \htmlonly   <h3>Credits</h3> \endhtmlonly
 *
 * Some models in this simulator is a rewrite of CSIM (2006) and other
 * work (Stiber and Kawasaki (2007?))
 *
 * @authors Derek McLean
 */

#pragma once

#include "Model.h"

class SingleThreadedSpikingModel : public Model {

public:
	//Constructor & Destructor
	SingleThreadedSpikingModel(Connections *conns, IAllNeurons *neurons, IAllSynapses *synapses, Layout *layout);
	virtual ~SingleThreadedSpikingModel();

        /**
         * Set up model state, if anym for a specific simulation run.
         *
         * @param sim_info - parameters defining the simulation to be run with the given collection of neurons.
         * @param simRecorder    Pointer to the simulation recordig object.
         */
        virtual void setupSim(SimulationInfo *sim_info);

        /**
         * Advances network state one simulation step.
         *
         * @param sim_info - parameters defining the simulation to be run with the given collection of neurons.
         */
	virtual void advance(const SimulationInfo *sim_info);

        /**
         * Modifies connections between neurons based on current state of the network and behavior
         * over the past epoch. Should be called once every epoch.
         *
         * @param currentStep - The epoch step in which the connections are being updated.
         * @param sim_info - parameters defining the simulation to be run with the given collection of neurons.
         */
	virtual void updateConnections(const SimulationInfo *sim_info);

	/* -----------------
	* # Helper Functions
	* ------------------
	*/

protected:

private:

};




