/**
 * @class  SingleThreadedSpikingModel SingleThreadedSpikingModel.h "SingleThreadedSpikingModel.h"
 *
 * Implements both neuron and synapse behaviour.
 *
 * @authors Derek McLean
 */

#pragma once

#include "Model.h"

class SingleThreadedSpikingModel : public Model {

public:
	//Constructor & Destructor
	SingleThreadedSpikingModel(Connections *conns, AllNeurons *neurons, AllSynapses *synapses, Layout *layout);
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




