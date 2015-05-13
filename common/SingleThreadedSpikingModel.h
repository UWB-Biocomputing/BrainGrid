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
	virtual void updateConnections(const int currentStep, const SimulationInfo *sim_info, IRecorder* simRecorder);

	/* -----------------
	* # Helper Functions
	* ------------------
	*/

protected:

private:
	// # Advance Network/Model
	// -----------------------

	// # Update Connections
	// --------------------

	// TODO
	void updateWeights(const int num_neurons, AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo *sim_info);

	// -----------------------------------------------------------------------------------------
	// # Generic Functions for handling synapse types
	// ---------------------------------------------

	// Determines the direction of the weight for a given synapse type.
	int synSign(const synapseType t);

};




