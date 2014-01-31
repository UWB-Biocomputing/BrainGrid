#include "LIFGPUModel.h"

/*
*  Constructor
*/
LIFGPUModel::LIFGPUModel() : LIFModel()
{

}

/*
* Destructor
*/
LIFGPUModel::~LIFGPUModel() 
{
	//Let LIFModel base class handle de-allocation
}



/**
 *  Advance everything in the model one time step. In this case, that
 *  means calling all of the kernels that do the "micro step" updating
 *  (i.e., NOT the stuff associated with growth).
 *  @param  neurons the Neuron list to search from.
 *  @param  synapses    the Synapse list to search from.
 *  @param  sim_info    SimulationInfo class to read information from.
 */
void LIFGPUModel::advance(AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo &sim_info)
{
#ifdef PERFORMANCE_METRICS
  cudaEventRecord( start, 0 );
#endif // PERFORMANCE_METRICS
  normalMTGPU(randNoise_d);
#ifdef PERFORMANCE_METRICS
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );
  cudaEventElapsedTime( &time, start, stop );
  t_gpu_rndGeneration += time;
#endif // PERFORMANCE_METRICS

  // display running info to console
  // Advance neurons ------------->
  blocksPerGrid = ( neuron_count + threadsPerBlock - 1 ) / threadsPerBlock;
#ifdef PERFORMANCE_METRICS
  cudaEventRecord( start, 0 );
#endif // PERFORMANCE_METRICS
#ifdef STORE_SPIKEHISTORY
  advanceNeuronsDevice <<< blocksPerGrid, threadsPerBlock >>> ( neuron_count, spikeHistory_d, g_simulationStep, maxSpikes, delayIdx.getIndex(), maxSynapses );
#else
  advanceNeuronsDevice <<< blocksPerGrid, threadsPerBlock >>> ( neuron_count, g_simulationStep, delayIdx.getIndex(), maxSynapses );
#endif // STORE_SPIKEHISTORY
#ifdef PERFORMANCE_METRICS
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );
  cudaEventElapsedTime( &time, start, stop );
  t_gpu_advanceNeurons += time;
#endif // PERFORMANCE_METRICS

  // Advance synapses ------------->
  blocksPerGrid = ( synapse_count + threadsPerBlock - 1 ) / threadsPerBlock;
#ifdef PERFORMANCE_METRICS
  cudaEventRecord( start, 0 );
#endif // PERFORMANCE_METRICS
  uint32_t bmask = delayIdx.getBitmask(  );
  advanceSynapsesDevice <<< blocksPerGrid, threadsPerBlock >>> ( synapse_count, width, g_simulationStep, bmask );
#ifdef PERFORMANCE_METRICS
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );
  cudaEventElapsedTime( &time, start, stop );
  t_gpu_advanceSynapses += time;
#endif // PERFORMANCE_METRICS

  // calculate summation point
  blocksPerGrid = ( neuron_count + threadsPerBlock - 1 ) / threadsPerBlock;
#ifdef PERFORMANCE_METRICS
  cudaEventRecord( start, 0 );
#endif // PERFORMANCE_METRICS
  calcSummationMap <<< blocksPerGrid, threadsPerBlock >>> ( neuron_count, inverseMap_d );
#ifdef PERFORMANCE_METRICS
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );
  cudaEventElapsedTime( &time, start, stop );
  t_gpu_calcSummation += time;
#endif // PERFORMANCE_METRICS

  // Advance the delayed queue index
  delayIdx.inc();
}

/**
 *  Update the connection of all the Neurons and Synapses of the simulation.
 *  @param  currentStep the current step of the simulation.
 *  @param  neurons the Neuron list to search from.
 *  @param  synapses    the Synapse list to search from.
 *  @param  sim_info    SimulationInfo class to read information from.
 */
void LIFGPUModel::updateConnections(const int currentStep, AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo &sim_info)
{

}

/**
 *  Outputs the spikes of the simulation.
 *  Note: only done if STORE_SPIKEHISTORY is true.
 *  @param  neurons list of all Neurons.
 *  @param  sim_info    SimulationInfo to refer.
 */
void LIFGPUModel::cleanupSim(AllNeurons &neurons, SimulationInfo &sim_info)
{

}

/* -----------------------------------------------------------------------------------------
* # Helper Functions
* ------------------
*/



/**
*  Updates the decay if the synapse selected.
*  @param  synapses    synapse list to find the indexed synapse from.
*  @param  neuron_index    index of the neuron that the synapse belongs to.
*  @param  synapse_index   index of the synapse to set.
*/
bool LIFGPUModel::updateDecay(AllSynapses &synapses, const int neuron_index, const int synapse_index)
{

}

/**
 *  Updates the Neuron at the indexed location.
 *  @param  synapses    synapse list to find the indexed synapse from.
 *  @param  neuron_index    index of the Neuron that the synapse belongs to.
 */
void LIFGPUModel::updateNeuron(AllNeurons &neurons, int neuron_index)
{

}

/**
 *  Notify outgoing synapses if neuron has fired.
 *  @param  neurons the Neuron list to search from
 *  @param  synapses    the Synapse list to search from.
 *  @param  sim_info    SimulationInfo class to read information from.
 */
void LIFGPUModel::advanceNeurons(AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo &sim_info)
{

}

/**
 *  Update the indexed Neuron.
 *  @param  neurons the Neuron list to search from.
 *  @param  index   index of the Neuron to update.
 */
void LIFGPUModel::advanceNeuron(AllNeurons &neurons, const int index)
{

}

/**
 *  Prepares Synapse for a spike hit.
 *  @param  synapses    the Synapse list to search from.
 *  @param  neuron_index   index of the Neuron that the Synapse connects to.
 *  @param  synapse_index   index of the Synapse to update.
 */
void LIFGPUModel::preSpikeHit(AllSynapses &synapses, const int neuron_index, const int synapse_index)
{

}

/**
 *  Fire the selected Neuron and calculate the result.
 *  @param  neurons the Neuron list to search from.
 *  @param  index   index of the Neuron to update.
 */
void LIFGPUModel::fire(AllNeurons &neurons, const int index) const
{

}

/**
 *  Advance all the Synapses in the simulation.
 *  @param  num_neurons number of neurons in the simulation to run.
 *  @param  synapses    list of Synapses to update.
 */
void LIFGPUModel::advanceSynapses(const int num_neurons, AllSynapses &synapses)
{

}

/**
 *  Advance one specific Synapse.
 *  @param  synapses    list of the Synapses to advance.
 *  @param  neuron_index    index of the Neuron that the Synapse connects to.
 *  @param  synapse_index   index of the Synapse to connect to.
 */
void LIFGPUModel::advanceSynapse(AllSynapses &synapses, const int neuron_index, const int synapse_index)
{
  
}

/**
 *  Checks if there is an input spike in the queue.
 *  @param  synapses    list of the Synapses to advance.
 *  @param  neuron_index    index of the Neuron that the Synapse connects to.
 *  @param  synapse_index   index of the Synapse to connect to.
 *  @return true if there is an input spike event.
 */
bool LIFGPUModel::isSpikeQueue(AllSynapses &synapses, const int neuron_index, const int synapse_index)
{

}

/**
 *  Get the spike counts from all Neurons in the simulation into the given pointer.
 *  @param  neurons the Neuron list to search from.
 *  @param  spikeCounts integer array to fill with the spike counts.
 */
void LIFGPUModel::getSpikeCounts(const AllNeurons &neurons, int *spikeCounts)
{

}

/**
 *  Clear the spike counts out of all Neurons.
 *  @param  neurons the Neuron list to search from.
 */
//! Clear spike count of each neuron.
void LIFGPUModel::clearSpikeCounts(AllNeurons &neurons)
{

}

/**
 *  Update the distance between frontiers of Neurons.
 *  @param  num_neurons in the simulation to update.
 */
void LIFGPUModel::updateFrontiers(const int num_neurons)
{

}

/**
 *  Update the areas of overlap in between Neurons.
 *  @param  num_neurons number of Neurons to update.
 */
void LIFGPUModel::updateOverlap(BGFLOAT num_neurons)
{

}

/**
 *  Update the weight of the Synapses in the simulation.
 *  Note: Platform Dependent.
 *  @param  num_neurons number of neurons to update.
 *  @param  neurons the Neuron list to search from.
 *  @param  synapses    the Synapse list to search from.
 *  @param  sim_info    SimulationInfo to refer from.
 */
void LIFGPUModel::updateWeights(const int num_neurons, AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo &sim_info)
{

}

/**
* Remove a synapse from the network.
* @param neuron_i	Index of a neuron.
* @param syn_i		Index of a synapse.
*/
__device__ void removeSynapse( int neuron_i, int syn_i )
{
	neuron_st_d[0].synapseCount[neuron_i]--;
	synapse_st_d[0].inUse[syn_i] = false;
}

/**
* Adds a synapse to the network.  Requires the locations of the source and
* destination neurons.
* @param W_new			The weight (scaling factor, strength, maximal amplitude) of the synapse.
* @param summationPoint_d	The summagtion map.
* @param rgNeuronTypeMap_d	The neuron type map (INH, EXC).
* @param neuron_i		Index of the source neuron.
* @param source_x		X location of source.
* @param source_y		Y location of source.
* @param dest_x			X location of destination.
* @param dest_y			Y location of destination.
* @param width			Width of neuron map (assumes square).
* @param deltaT			The time step size.
* @param maxSynapses		Maximum number of synapses per neuron.
*/
__device__ void addSynapse( FLOAT W_new, FLOAT* summationPoint_d, neuronType* rgNeuronTypeMap_d, int neuron_i, int source_x, int source_y, int dest_x, int dest_y, int width, FLOAT deltaT, int maxSynapses )
{
	if ( neuron_st_d[0].synapseCount[neuron_i] >= maxSynapses )
		return;			// TODO: ERROR!

	// locate summation point
	FLOAT* sp = &( summationPoint_d[dest_x + dest_y * width] );

	// determine the synapse type
	synapseType type = synType( rgNeuronTypeMap_d, source_x, source_y, dest_x, dest_y, width );

	// add it to the list
	int syn_i = neuron_st_d[0].outgoingSynapse_begin[neuron_i];
	for ( int i = 0; i < maxSynapses; i++, syn_i++ )
		if ( synapse_st_d[0].inUse[syn_i] != true )
			break;

	neuron_st_d[0].synapseCount[neuron_i]++;

	// create a synapse
	createSynapse( syn_i, source_x, source_y, dest_x, dest_y, sp, deltaT, type );	
	synapse_st_d[0].W[syn_i] = W_new;
}

/**
* Create a synapse.
* @param syn_i		Index of the synapse.
* @param source_x	X location of source.
* @param source_y	Y location of source.
* @param dest_x		X location of destination.
* @param dest_y		Y location of destination.
* @param sp		Summation point.
* @param deltaT		The time step size.
* @param type		Type of the synapse.
*/
__device__ void createSynapse( int syn_i, int source_x, int source_y, int dest_x, int dest_y, FLOAT* sp, FLOAT deltaT, synapseType type )
{
	FLOAT delay;

	synapse_st_d[0].inUse[syn_i] = true;
	synapse_st_d[0].summationPoint[syn_i] = sp;
	synapse_st_d[0].summationCoord[syn_i].x = dest_x;
	synapse_st_d[0].summationCoord[syn_i].y = dest_y;
	synapse_st_d[0].synapseCoord[syn_i].x = source_x;	
	synapse_st_d[0].synapseCoord[syn_i].y = source_y;	
	synapse_st_d[0].deltaT[syn_i] = deltaT;
	synapse_st_d[0].W[syn_i] = 10.0e-9;
	synapse_st_d[0].psr[syn_i] = 0.0;
	synapse_st_d[0].delayQueue[syn_i] = 0;
	synapse_st_d[0].ldelayQueue[syn_i] = LENGTH_OF_DELAYQUEUE;
	synapse_st_d[0].r[syn_i] = 1.0;
	synapse_st_d[0].u[syn_i] = 0.4;		// DEFAULT_U
	synapse_st_d[0].lastSpike[syn_i] = ULONG_MAX;
	synapse_st_d[0].type[syn_i] = type;

	FLOAT tau;
	switch ( type ) {
	case II:
		tau = 6e-3;
		delay = 0.8e-3;
		break;
	case IE:
		tau = 6e-3;
		delay = 0.8e-3;
		break;
	case EI:
		tau = 3e-3;
		delay = 0.8e-3;
		break;
	case EE:
		tau = 3e-3;
		delay = 1.5e-3;
		break;
	}

	synapse_st_d[0].tau[syn_i] = tau;
	synapse_st_d[0].total_delay[syn_i] = static_cast<int>( delay / deltaT ) + 1;
	synapse_st_d[0].decay[syn_i] = exp( -deltaT / tau );
}

/**
* Returns the type of synapse at the given coordinates
* @param rgNeuronTypeMap_d	The neuron type map (INH, EXC).
* @param ax	Source coordinate(x).
* @param ay	Source coordinate(y).
* @param bx	Destination coordinate(x).
* @param by	Destination coordinate(y).
* @param width	Width of neuron map (assumes square).
* @return type of synapse at the given coordinate or -1 on error
*/
__device__ synapseType synType( neuronType* rgNeuronTypeMap_d, int ax, int ay, int bx, int by, int width )
{
	if ( rgNeuronTypeMap_d[ax + ay * width] == INH && rgNeuronTypeMap_d[bx + by * width] == INH )
		return II;
	else if ( rgNeuronTypeMap_d[ax + ay * width] == INH && rgNeuronTypeMap_d[bx + by * width] == EXC )
		return IE;
	else if ( rgNeuronTypeMap_d[ax + ay * width] == EXC && rgNeuronTypeMap_d[bx + by * width] == INH )
		return EI;
	else if ( rgNeuronTypeMap_d[ax + ay * width] == EXC && rgNeuronTypeMap_d[bx + by * width] == EXC )
		return EE;

	return STYPE_UNDEF;
}

/**
* Return 1 if originating neuron is excitatory, -1 otherwise.
* @param[in] t	synapseType I to I, I to E, E to I, or E to E
* @return 1 or -1
*/
__device__ int synSign( synapseType t )
{
	switch ( t )
	{
	case II:
	case IE:
		return -1;
	case EI:
	case EE:
		return 1;
	}

	return 0;
}






