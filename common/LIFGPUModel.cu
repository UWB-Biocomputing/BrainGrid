/** - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - **\ 
 * @authors Aaron Oziel, Sean Blackbourn 
 *
 * Aaron Wrote (2/3/14):
 * All comments are now tracking progress in conversion from old GpuSim_struct.cu
 * file to the new one here. This is a quick key to keep track of their meanings. 
 *
 *	TODO = 	Needs work and/or is blank. Used to indicate possibly problematic 
 *				functions. 
 *	DONE = 	Likely complete functions. Will still need to be checked for
 *				variable continuity and proper arguments. 
 *   REMOVED =	Deleted, likely due to it becoming unnecessary or not necessary 
 *				for GPU implementation. These functions will likely have to be 
 *				removed from the LIFModel super class.
 *    COPIED = 	These functions were in the original GpuSim_struct.cu file 
 *				and were directly copy-pasted across to this file. 
 *
\** - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - **/

#include "LIFGPUModel.h"

LIFGPUModel::LIFGPUModel() : LIFModel()
{

}

LIFGPUModel::~LIFGPUModel() 
{
	//Let LIFModel base class handle de-allocation
}

/** DONE
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

/** TODO
*  Update the connection of all the Neurons and Synapses of the simulation.
*  @param  currentStep the current step of the simulation.
*  @param  neurons the Neuron list to search from.
*  @param  synapses    the Synapse list to search from.
*  @param  sim_info    SimulationInfo class to read information from.
*/
void LIFGPUModel::updateConnections(const int currentStep, AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo &sim_info)
{

}

/** TODO
*  Outputs the spikes of the simulation.
*  Note: only done if STORE_SPIKEHISTORY is true.
*  @param  neurons list of all Neurons.
*  @param  sim_info    SimulationInfo to refer.
*/
void LIFGPUModel::cleanupSim(AllNeurons &neurons, SimulationInfo &sim_info)
{

}

/* ------------------*\
|* # Helper Functions
\* ------------------*/

/** TODO
*  Updates the decay if the synapse selected.
*  @param  synapses    synapse list to find the indexed synapse from.
*  @param  neuron_index    index of the neuron that the synapse belongs to.
*  @param  synapse_index   index of the synapse to set.
*/
bool LIFGPUModel::updateDecay(AllSynapses &synapses, const int neuron_index, const int synapse_index)
{

}

/** REMOVED
*  Updates the Neuron at the indexed location.
*  @param  synapses    synapse list to find the indexed synapse from.
*  @param  neuron_index    index of the Neuron that the synapse belongs to.

void LIFGPUModel::updateNeuron(AllNeurons &neurons, int neuron_index)
{
	// GPU Does not advance single elements at a time.
}
*/

/** TODO: Determine if this can be replaced by __global__ advanceNeuronDevice
*  Notify outgoing synapses if neuron has fired.
*  @param  neurons the Neuron list to search from
*  @param  synapses    the Synapse list to search from.
*  @param  sim_info    SimulationInfo class to read information from.
*/
void LIFGPUModel::advanceNeurons(AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo &sim_info)
{

}

/** REMOVED
*  Update the indexed Neuron.
*  @param  neurons the Neuron list to search from.
*  @param  index   index of the Neuron to update.

void LIFGPUModel::advanceNeuron(AllNeurons &neurons, const int index)
{
	// GPU Does not advance single elements at a time.
}
*/

/** TODO
*  Prepares Synapse for a spike hit.
*  @param  synapses    the Synapse list to search from.
*  @param  neuron_index   index of the Neuron that the Synapse connects to.
*  @param  synapse_index   index of the Synapse to update.
*/
void LIFGPUModel::preSpikeHit(AllSynapses &synapses, const int neuron_index, const int synapse_index)
{

}

/** TODO
*  Fire the selected Neuron and calculate the result.
*  @param  neurons the Neuron list to search from.
*  @param  index   index of the Neuron to update.
*/
void LIFGPUModel::fire(AllNeurons &neurons, const int index) const
{

}

/** TODO: Determine if this can be replaced by __global__ advanceSynapsesDevice
*  Advance all the Synapses in the simulation.
*  @param  num_neurons number of neurons in the simulation to run.
*  @param  synapses    list of Synapses to update.
*/
void LIFGPUModel::advanceSynapses(const int num_neurons, AllSynapses &synapses)
{

}

/** REMOVED
*  Advance one specific Synapse.
*  @param  synapses    list of the Synapses to advance.
*  @param  neuron_index    index of the Neuron that the Synapse connects to.
*  @param  synapse_index   index of the Synapse to connect to.

void LIFGPUModel::advanceSynapse(AllSynapses &synapses, const int neuron_index, const int synapse_index)
{
	// GPU Does not advance single elements at a time.
}
*/

/** TODO
*  Checks if there is an input spike in the queue.
*  @param  synapses    list of the Synapses to advance.
*  @param  neuron_index    index of the Neuron that the Synapse connects to.
*  @param  synapse_index   index of the Synapse to connect to.
*  @return true if there is an input spike event.
*/
bool LIFGPUModel::isSpikeQueue(AllSynapses &synapses, const int neuron_index, const int synapse_index)
{

}

/** DONE
*  Get the spike counts from all Neurons in the simulation into the given pointer.
*  @param  neurons the Neuron list to search from.
*  @param  spikeCounts integer array to fill with the spike counts.
*/
void LIFGPUModel::getSpikeCounts(const AllNeurons &neurons, int *spikeCounts)
{
	LifNeuron_struct neuron;
	HANDLE_ERROR( cudaMemcpyFromSymbol ( &neuron, neuron_st_d, sizeof( LifNeuron_struct ) ) );
	HANDLE_ERROR( cudaMemcpy ( spikeCounts, neuron.spikeCount, neuron_count * sizeof( int ), cudaMemcpyDeviceToHost ) );
}

/** DONE
*  Clear the spike counts out of all Neurons.
*  @param  neurons the Neuron list to search from.
*/
//! Clear spike count of each neuron.
void LIFGPUModel::clearSpikeCounts(AllNeurons &neurons)
{
	LifNeuron_struct neuron;
	HANDLE_ERROR( cudaMemcpyFromSymbol ( &neuron, neuron_st_d, sizeof( LifNeuron_struct ) ) );
	HANDLE_ERROR( cudaMemset( neuron.spikeCount, 0, neuron_count * sizeof( int ) ) );
}

/** TODO
*  Update the distance between frontiers of Neurons.
*  @param  num_neurons in the simulation to update.
*/
void LIFGPUModel::updateFrontiers(const int num_neurons)
{

}

/** TODO
*  Update the areas of overlap in between Neurons.
*  @param  num_neurons number of Neurons to update.
*/
void LIFGPUModel::updateOverlap(BGBGBGFLOAT num_neurons)
{

}

/** TODO (Parameters are wrong)
*  Update the weight of the Synapses in the simulation.
*  Note: Platform Dependent.
*  @param  num_neurons number of neurons to update.
*  @param  neurons the Neuron list to search from.
*  @param  synapses    the Synapse list to search from.
*  @param  sim_info    SimulationInfo to refer from.
*/
void LIFGPUModel::updateWeights(const int num_neurons, AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo &sim_info)
{
	int neuron_count = psi->cNeurons;
	int width = psi->width;
	BGFLOAT deltaT = psi->deltaT;

	// CUDA parameters
	const int threadsPerBlock = 256;
	int blocksPerGrid;

	// allocate memories
	size_t W_d_size = neuron_count * neuron_count * sizeof (BGFLOAT);
	BGFLOAT* W_h = new BGFLOAT[W_d_size];
	BGFLOAT* W_d;
	HANDLE_ERROR( cudaMalloc ( ( void ** ) &W_d, W_d_size ) );

	// copy weight data to the device memory
	for ( int i = 0 ; i < neuron_count; i++ )
	for ( int j = 0; j < neuron_count; j++ )
	W_h[i * neuron_count + j] = W(i, j);

	HANDLE_ERROR( cudaMemcpy ( W_d, W_h, W_d_size, cudaMemcpyHostToDevice ) );

	blocksPerGrid = ( neuron_count + threadsPerBlock - 1 ) / threadsPerBlock;
	updateNetworkDevice <<< blocksPerGrid, threadsPerBlock >>> ( summationPoint_d, rgNeuronTypeMap_d, neuron_count, width, deltaT, W_d, maxSynapses );

	// free memories
	HANDLE_ERROR( cudaFree( W_d ) );
	delete[] W_h;

	// create synapse inverse map
	//! createSynapseImap( psi, maxSynapses );
}

/* ------------------*\
|* # Global Functions
\* ------------------*/

/** COPIED
* @param[in] n			Number of synapses.
* @param[in] simulationStep	The current simulation step.
* @param[in] delayIdx		Index of the delayed list (spike queue).
* @param[in] maxSynapses	Maximum number of synapses per neuron.
*/
__global__ void advanceNeuronsDevice( int n, uint64_t simulationStep, int delayIdx, int maxSynapses ) {
	// determine which neuron this thread is processing
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx >= n )
	return;

	neuron_st_d[0].hasFired[idx] = false;
	BGFLOAT& sp = *neuron_st_d[0].summationPoint[idx];
	BGFLOAT& vm = neuron_st_d[0].Vm[idx];
	BGFLOAT r_sp = sp;
	BGFLOAT r_vm = vm;

	if ( neuron_st_d[0].nStepsInRefr[idx] > 0 ) { // is neuron refractory?
		--neuron_st_d[0].nStepsInRefr[idx];
	} else if ( r_vm >= neuron_st_d[0].Vthresh[idx] ) { // should it fire?
		// Note that the neuron has fired!
		neuron_st_d[0].hasFired[idx] = true;

#ifdef STORE_SPIKEHISTORY
		// record spike time
		spikeHistory_d[(idx * maxSpikes) + neuron_st_d[0].spikeCount[idx]] = simulationStep;
#endif // STORE_SPIKEHISTORY
		neuron_st_d[0].spikeCount[idx]++;

		// calculate the number of steps in the absolute refractory period
		neuron_st_d[0].nStepsInRefr[idx] = static_cast<int> ( neuron_st_d[0].Trefract[idx] / neuron_st_d[0].deltaT[idx] + 0.5 );

		// reset to 'Vreset'
		vm = neuron_st_d[0].Vreset[idx];

		// notify synapses of spike
		int syn_i = neuron_st_d[0].outgoingSynapse_begin[idx];
		for ( int i = 0; i < maxSynapses; i++ ) {
			if ( synapse_st_d[0].inUse[syn_i + i] == true )
			{
				// notify synapses of spike...
				int idx0 = delayIdx + synapse_st_d[0].total_delay[syn_i + i];
				if ( idx0 >= LENGTH_OF_DELAYQUEUE )
				idx0 -= LENGTH_OF_DELAYQUEUE;

				// set a spike
				synapse_st_d[0].delayQueue[syn_i + i] |= (0x1 << idx0);
			}
		}
	} else {

		r_sp += neuron_st_d[0].I0[idx]; // add IO
		
		// Random number alg. goes here    
		r_sp += (*neuron_st_d[0].randNoise[idx] * neuron_st_d[0].Inoise[idx]); // add cheap noise
		vm = neuron_st_d[0].C1[idx] * r_vm + neuron_st_d[0].C2[idx] * ( r_sp ); // decay Vm and add inputs
	}

	// clear synaptic input for next time step
	sp = 0;
}

/** COPIED
* @param[in] n			Number of synapses.
* @param[in] width		Width of neuron map (assumes square).
* @param[in] simulationStep	The current simulation step.
* @param[in] bmask		Bit mask for the delayed list (spike queue).
*/
__global__ void advanceSynapsesDevice ( int n, int width, uint64_t simulationStep, uint32_t bmask ) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx >= n )
	return;

	if ( synapse_st_d[0].inUse[idx] != true )
	return;

	int itype = synapse_st_d[0].type[idx];

	// is there a spike in the queue?
	uint32_t s_delayQueue = synapse_st_d[0].delayQueue[idx];
	bool isFired = s_delayQueue & bmask;
	synapse_st_d[0].delayQueue[idx] = s_delayQueue & (~bmask);
	BGFLOAT s_decay = synapse_st_d[0].decay[idx];
	if ( isFired ) {
		// adjust synapse paramaters
		if ( synapse_st_d[0].lastSpike[idx] != ULONG_MAX ) {
			BGFLOAT isi = (simulationStep - synapse_st_d[0].lastSpike[idx]) * synapse_st_d[0].deltaT[idx];
			synapse_st_d[0].r[idx] = 1 + ( synapse_st_d[0].r[idx] * ( 1 - synapse_st_d[0].u[idx] ) - 1 ) * exp ( -isi / synapse_D_d[itype] );
			synapse_st_d[0].u[idx] = synapse_U_d[itype] + synapse_st_d[0].u[idx] * ( 1 - synapse_U_d[itype] ) * exp ( -isi / synapse_F_d[itype] );
		}

		synapse_st_d[0].psr[idx] += ( ( synapse_st_d[0].W[idx] / s_decay ) * synapse_st_d[0].u[idx] * synapse_st_d[0].r[idx] );// calculate psr
		synapse_st_d[0].lastSpike[idx] = simulationStep; // record the time of the spike
	}

	// decay the post spike response
	synapse_st_d[0].psr[idx] *= s_decay;
}

/** COPIED
* @param[in] n			Number of neurons.
* @param[in] inverseMap	Inverse map, which is a table indexed by an input neuron and maps to the synapses that provide input to that neuron..
*/
__global__ void calcSummationMap( int n, uint32_t* inverseMap ) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx >= n )
	return;
	
	uint32_t* inverseMap_begin = &inverseMap[neuron_st_d[0].incomingSynapse_begin[idx]];
	BGFLOAT sum = 0.0;
	uint32_t iCount = neuron_st_d[0].inverseCount[idx];
	for ( uint32_t i = 0; i < iCount; i++ ) {
		uint32_t syn_i = inverseMap_begin[i];
		sum += synapse_st_d[0].psr[syn_i];
	}
	*neuron_st_d[0].summationPoint[idx] = sum;
} 

/** COPIED
* @param[in] n			Number of neurons.
* @param[in] summationPoint_d	The summation map.
* @param[in] width		Width of neuron map (assumes square).
* @param[in] randNoise_d	Array of randum numbers. 
*/
__global__ void calcOffsets( int n, BGFLOAT* summationPoint_d, int width, BGFLOAT* randNoise_d ) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx >= n )
	return;

	// set summation pointer
	neuron_st_d[0].summationPoint[idx] = &summationPoint_d[idx];
	*neuron_st_d[0].summationPoint[idx] = 0;

	neuron_st_d[0].randNoise[idx] = &randNoise_d[idx];
}

/** COPIED
* Adjust the strength of the synapse or remove it from the synapse map if it has gone below 
* zero.
* @param[in] summationPoint_d	The summation map.
* @param[in] rgNeuronTypeMap_d	The neuron type map (INH, EXC).
* @param[in] n			Number of neurons.
* @param[in] width		Width of neuron map (assumes square).
* @param[in] deltaT		The time step size.
* @param[in] W_d		Array of synapse weight.
* @param[in] maxSynapses	Maximum number of synapses per neuron.
*/
__global__ void updateNetworkDevice( BGFLOAT* summationPoint_d, neuronType* rgNeuronTypeMap_d, int n, int width, BGFLOAT deltaT, BGFLOAT* W_d, int maxSynapses )
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx >= n )
	return;

	int adjusted = 0;
	int removed = 0;
	int added = 0;

	// source neuron coordinate
	int a = idx;
	int xa = a % width;
	int ya = a / width;

	// and each destination neuron 'b'
	for ( int b = 0; b < n; b++ )
	{
		int xb = b % width;
		int yb = b / width;

		// visit each synapse at (xa, ya)
		bool connected = false;

		// for each existing synapse
		int syn_i = neuron_st_d[0].outgoingSynapse_begin[a];
		for ( int i = 0; i < maxSynapses; i++ )
		{
			if ( synapse_st_d[0].inUse[syn_i + i] != true)
			continue;
			// if there is a synapse between a and b
			if ( synapse_st_d[0].summationCoord[syn_i + i].x == xb &&
					synapse_st_d[0].summationCoord[syn_i + i].y == yb )
			{
				connected = true;
				adjusted++;

				// adjust the strength of the synapse or remove 
				// it from the synapse map if it has gone below
				// zero.
				if ( W_d[a * n + b] < 0 )
				{
					removed++;
					removeSynapse( a, syn_i + i );
				}
				else
				{
					// adjust
					// g_synapseStrengthAdjustmentConstant is 1.0e-8;
					synapse_st_d[0].W[syn_i + i] = W_d[a * n + b] 
					* synSign( synType( rgNeuronTypeMap_d, xa, ya, xb, yb, width ) ) 
					* g_synapseStrengthAdjustmentConstant_d;
				}
			}
		}

		// if not connected and weight(a,b) > 0, add a new synapse from a to b
		if ( !connected && ( W_d[a * n + b] > 0 ) )
		{
			added++;
			BGFLOAT W_new = W_d[a * n + b] 
			* synSign( synType( rgNeuronTypeMap_d, xa, ya, xb, yb, width ) ) 
			* g_synapseStrengthAdjustmentConstant_d;	
			addSynapse( W_new, summationPoint_d, rgNeuronTypeMap_d, a, xa, ya, xb, yb, width, deltaT, maxSynapses );
		}
	}
}

/* ------------------*\
|* # Device Functions
\* ------------------*/

/** DONE
* Remove a synapse from the network.
* @param neuron_i	Index of a neuron.
* @param syn_i		Index of a synapse.
*/
__device__ void removeSynapse( int neuron_i, int syn_i )
{
	neuron_st_d[0].synapseCount[neuron_i]--;
	synapse_st_d[0].inUse[syn_i] = false;
}

/** DONE
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
__device__ void addSynapse( BGBGFLOAT W_new, BGBGFLOAT* summationPoint_d, neuronType* rgNeuronTypeMap_d, int neuron_i, int source_x, int source_y, int dest_x, int dest_y, int width, BGBGFLOAT deltaT, int maxSynapses )
{
	if ( neuron_st_d[0].synapseCount[neuron_i] >= maxSynapses )
	return;			// TODO: ERROR!

	// locate summation point
	BGBGFLOAT* sp = &( summationPoint_d[dest_x + dest_y * width] );

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

/** DONE
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
__device__ void createSynapse( int syn_i, int source_x, int source_y, int dest_x, int dest_y, BGBGFLOAT* sp, BGBGFLOAT deltaT, synapseType type )
{
	BGBGFLOAT delay;

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

	BGBGFLOAT tau;
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

/** DONE
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

/** DONE
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


