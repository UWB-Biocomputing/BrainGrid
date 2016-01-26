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
 * essential components of the spiking neunal networks.
 *    -# IAllNeurons: A class to define a list of partiular type of neurons.
 *    -# IAllSynapses: A class to define a list of partiular type of synapses.
 *    -# Connections: A class to define connections of the neunal network.
 *    -# Layout: A class to define neurons' layout information in the network.
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

	virtual void advance(const SimulationInfo *sim_info);
	virtual void updateConnections(const SimulationInfo *sim_info);

	/* -----------------
	* # Helper Functions
	* ------------------
	*/

protected:

private:

};




