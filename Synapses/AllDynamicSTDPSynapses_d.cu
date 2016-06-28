/*
 * AllDynamicSTDPSynapses_d.cu
 *
 */

#include "AllDynamicSTDPSynapses.h"
#include "Book.h"

/*
 *  Allocate GPU memories to store all synapses' states,
 *  and copy them from host to GPU memory.
 *
 *  @param  allSynapsesDevice  Reference to the allSynapses struct on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllDynamicSTDPSynapses::allocSynapseDeviceStruct( void** allSynapsesDevice, const SimulationInfo *sim_info ) {
	allocSynapseDeviceStruct( allSynapsesDevice, sim_info->totalNeurons, sim_info->maxSynapsesPerNeuron );
}

/*
 *  Allocate GPU memories to store all synapses' states,
 *  and copy them from host to GPU memory.
 *
 *  @param  allSynapsesDevice     Reference to the allSynapses struct on device memory.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllDynamicSTDPSynapses::allocSynapseDeviceStruct( void** allSynapsesDevice, int num_neurons, int maxSynapsesPerNeuron ) {
	AllDynamicSTDPSynapses allSynapses;

	allocDeviceStruct( allSynapses, num_neurons, maxSynapsesPerNeuron );

	HANDLE_ERROR( cudaMalloc( allSynapsesDevice, sizeof( AllDynamicSTDPSynapses ) ) );
	HANDLE_ERROR( cudaMemcpy ( *allSynapsesDevice, &allSynapses, sizeof( AllDynamicSTDPSynapses ), cudaMemcpyHostToDevice ) );
}

/*
 *  Allocate GPU memories to store all synapses' states,
 *  and copy them from host to GPU memory.
 *  (Helper function of allocSynapseDeviceStruct)
 *
 *  @param  allSynapsesDevice  Reference to the allSynapses struct on device memory.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllDynamicSTDPSynapses::allocDeviceStruct( AllDynamicSTDPSynapses &allSynapses, int num_neurons, int maxSynapsesPerNeuron ) {
        AllSTDPSynapses::allocDeviceStruct( allSynapses, num_neurons, maxSynapsesPerNeuron );

        BGSIZE max_total_synapses = maxSynapsesPerNeuron * num_neurons;

        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.lastSpike, max_total_synapses * sizeof( uint64_t ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.r, max_total_synapses * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.u, max_total_synapses * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.D, max_total_synapses * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.U, max_total_synapses * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.F, max_total_synapses * sizeof( BGFLOAT ) ) );
}

/*
 *  Delete GPU memories.
 *
 *  @param  allSynapsesDevice  Reference to the allSynapses struct on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllDynamicSTDPSynapses::deleteSynapseDeviceStruct( void* allSynapsesDevice ) {
	AllDynamicSTDPSynapses allSynapses;

	HANDLE_ERROR( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllDynamicSTDPSynapses ), cudaMemcpyDeviceToHost ) );

	deleteDeviceStruct( allSynapses );

	HANDLE_ERROR( cudaFree( allSynapsesDevice ) );
}

/*
 *  Delete GPU memories.
 *  (Helper function of deleteSynapseDeviceStruct)
 *
 *  @param  allSynapsesDevice  Reference to the allSynapses struct on device memory.
 */
void AllDynamicSTDPSynapses::deleteDeviceStruct( AllDynamicSTDPSynapses& allSynapses ) {
        HANDLE_ERROR( cudaFree( allSynapses.lastSpike ) );
	HANDLE_ERROR( cudaFree( allSynapses.r ) );
	HANDLE_ERROR( cudaFree( allSynapses.u ) );
	HANDLE_ERROR( cudaFree( allSynapses.D ) );
	HANDLE_ERROR( cudaFree( allSynapses.U ) );
	HANDLE_ERROR( cudaFree( allSynapses.F ) );

        AllSTDPSynapses::deleteDeviceStruct( allSynapses );
}

/*
 *  Copy all synapses' data from host to device.
 *
 *  @param  allSynapsesDevice  Reference to the allSynapses struct on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllDynamicSTDPSynapses::copySynapseHostToDevice( void* allSynapsesDevice, const SimulationInfo *sim_info ) { // copy everything necessary
	copySynapseHostToDevice( allSynapsesDevice, sim_info->totalNeurons, sim_info->maxSynapsesPerNeuron );	
}

/*
 *  Copy all synapses' data from host to device.
 *
 *  @param  allSynapsesDevice  Reference to the allSynapses struct on device memory.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllDynamicSTDPSynapses::copySynapseHostToDevice( void* allSynapsesDevice, int num_neurons, int maxSynapsesPerNeuron ) { // copy everything necessary
	AllDynamicSTDPSynapses allSynapses;

        HANDLE_ERROR( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllDynamicSTDPSynapses ), cudaMemcpyDeviceToHost ) );

	copyHostToDevice( allSynapsesDevice, allSynapses, num_neurons, maxSynapsesPerNeuron );	
}

/*
 *  Copy all synapses' data from host to device.
 *  (Helper function of copySynapseHostToDevice)
 *
 *  @param  allSynapsesDevice  Reference to the allSynapses struct on device memory.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllDynamicSTDPSynapses::copyHostToDevice( void* allSynapsesDevice, AllDynamicSTDPSynapses& allSynapses, int num_neurons, int maxSynapsesPerNeuron ) { // copy everything necessary 
        AllSTDPSynapses::copyHostToDevice( allSynapsesDevice, allSynapses, num_neurons, maxSynapsesPerNeuron );

        BGSIZE max_total_synapses = maxSynapsesPerNeuron * num_neurons;
        
        HANDLE_ERROR( cudaMemcpy ( allSynapses.lastSpike, lastSpike,
                max_total_synapses * sizeof( uint64_t ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses.r, r,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses.u, u,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses.D, D,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses.U, U,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses.F, F,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
}

/*
 *  Copy all synapses' data from device to host.
 *
 *  @param  allSynapsesDevice  Reference to the allSynapses struct on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllDynamicSTDPSynapses::copySynapseDeviceToHost( void* allSynapsesDevice, const SimulationInfo *sim_info ) {
	// copy everything necessary
	AllDynamicSTDPSynapses allSynapses;

        HANDLE_ERROR( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllDynamicSTDPSynapses ), cudaMemcpyDeviceToHost ) );

	copyDeviceToHost( allSynapses, sim_info );
}

/*
 *  Copy all synapses' data from device to host.
 *  (Helper function of copySynapseDeviceToHost)
 *
 *  @param  allSynapsesDevice  Reference to the allSynapses struct on device memory.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllDynamicSTDPSynapses::copyDeviceToHost( AllDynamicSTDPSynapses& allSynapses, const SimulationInfo *sim_info ) {
        AllSTDPSynapses::copyDeviceToHost( allSynapses, sim_info ) ;

	int num_neurons = sim_info->totalNeurons;
	BGSIZE max_total_synapses = sim_info->maxSynapsesPerNeuron * num_neurons;

        HANDLE_ERROR( cudaMemcpy ( lastSpike, allSynapses.lastSpike,
                max_total_synapses * sizeof( uint64_t ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( r, allSynapses.r,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( u, allSynapses.u,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( D, allSynapses.D,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( U, allSynapses.U,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( F, allSynapses.F,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
}

__device__ fpCreateSynapse_t fpCreateDynamicSTDPSynapse_d = (fpCreateSynapse_t)createDynamicSTDPSSynapse;

/*
 *  Get a pointer to the device function createDynamicSTDPSSynapse.
 *  The function will be called from updateSynapsesWeightsDevice device function.
 *  Because we cannot use virtual function (Polymorphism) in device functions,
 *  we use this scheme.
 *
 *  @param  fpCreateSynapse_h     Reference to the memory location 
 *                                where the function pointer will be set.
 */
void AllDynamicSTDPSynapses::getFpCreateSynapse(fpCreateSynapse_t& fpCreateSynapse_h)
{
    HANDLE_ERROR( cudaMemcpyFromSymbol(&fpCreateSynapse_h, fpCreateDynamicSTDPSynapse_d, sizeof(fpCreateSynapse_t)) );
}

__device__ fpChangeSynapsesPSR_t fpChangeDynamicSTDPSynapsesPSR_d = (fpChangeSynapsesPSR_t)changeDynamicSTDPSynapsePSR;

/*
 *  Get a pointer to the device function changeDynamicSTDPSynapsePSR.
 *  The function will be called from advanceSynapsesDevice device function.
 *  Because we cannot use virtual function (Polymorphism) in device functions,
 *  we use this scheme.
 *
 *  @param  fpChangePSR_h         Reference to the memory location
 *                                where the function pointer will be set.
 */
void AllDynamicSTDPSynapses::getFpChangePSR(fpChangeSynapsesPSR_t& fpChangePSR_h)
{
    HANDLE_ERROR( cudaMemcpyFromSymbol(&fpChangePSR_h, fpChangeDynamicSTDPSynapsesPSR_d, sizeof(fpChangeSynapsesPSR_t)) );
}

/* ------------------*\
|* # Global Functions
\* ------------------*/

/* ------------------*\
|* # Device Functions
\* ------------------*/

/*
 *  Create a Synapse and connect it to the model.
 *
 *  @param allSynapsesDevice    Pointer to the Synapse structures in device memory.
 *  @param neuron_index         Index of the source neuron.
 *  @param synapse_index        Index of the Synapse to create.
 *  @param source_x             X location of source.
 *  @param source_y             Y location of source.
 *  @param dest_x               X location of destination.
 *  @param dest_y               Y location of destination.
 *  @param sum_point            Pointer to the summation point.
 *  @param deltaT               The time step size.
 *  @param type                 Type of the Synapse to create.
 */
__device__ void createDynamicSTDPSSynapse(AllDynamicSTDPSynapses* allSynapsesDevice, const int neuron_index, const int synapse_index, int source_index, int dest_index, BGFLOAT *sum_point, const BGFLOAT deltaT, synapseType type)
{
    BGFLOAT delay;
    BGSIZE max_synapses = allSynapsesDevice->maxSynapsesPerNeuron;
    BGSIZE iSyn = max_synapses * neuron_index + synapse_index;

    allSynapsesDevice->in_use[iSyn] = true;
    allSynapsesDevice->summationPoint[iSyn] = sum_point;
    allSynapsesDevice->destNeuronIndex[iSyn] = dest_index;
    allSynapsesDevice->sourceNeuronIndex[iSyn] = source_index;
    allSynapsesDevice->W[iSyn] = synSign(type) * 10.0e-9;

    allSynapsesDevice->delayQueue[iSyn] = 0;
    allSynapsesDevice->delayIdx[iSyn] = 0;
    allSynapsesDevice->ldelayQueue[iSyn] = LENGTH_OF_DELAYQUEUE;

    allSynapsesDevice->psr[iSyn] = 0.0;
    allSynapsesDevice->r[iSyn] = 1.0;
    allSynapsesDevice->u[iSyn] = 0.4;     // DEFAULT_U
    allSynapsesDevice->lastSpike[iSyn] = ULONG_MAX;
    allSynapsesDevice->type[iSyn] = type;

    allSynapsesDevice->U[iSyn] = DEFAULT_U;
    allSynapsesDevice->tau[iSyn] = DEFAULT_tau;

    BGFLOAT U;
    BGFLOAT D;
    BGFLOAT F;
    BGFLOAT tau;
    switch (type) {
        case II:
            U = 0.32;
            D = 0.144;
            F = 0.06;
            tau = 6e-3;
            delay = 0.8e-3;
            break;
        case IE:
            U = 0.25;
            D = 0.7;
            F = 0.02;
            tau = 6e-3;
            delay = 0.8e-3;
            break;
        case EI:
            U = 0.05;
            D = 0.125;
            F = 1.2;
            tau = 3e-3;
            delay = 0.8e-3;
            break;
        case EE:
            U = 0.5;
            D = 1.1;
            F = 0.05;
            tau = 3e-3;
            delay = 1.5e-3;
            break;
        default:
            break;
    }

    allSynapsesDevice->U[iSyn] = U;
    allSynapsesDevice->D[iSyn] = D;
    allSynapsesDevice->F[iSyn] = F;

    allSynapsesDevice->tau[iSyn] = tau;
    allSynapsesDevice->decay[iSyn] = exp( -deltaT / tau );
    allSynapsesDevice->total_delay[iSyn] = static_cast<int>( delay / deltaT ) + 1;

    uint32_t size = allSynapsesDevice->total_delay[iSyn] / ( sizeof(uint8_t) * 8 ) + 1;
    assert( size <= BYTES_OF_DELAYQUEUE );

    allSynapsesDevice->Apos[iSyn] = 0.5;
    allSynapsesDevice->Aneg[iSyn] = -0.5;
    allSynapsesDevice->STDPgap[iSyn] = 2e-3;

    allSynapsesDevice->total_delayPost[iSyn] = 0;

    allSynapsesDevice->tauspost[iSyn] = 0;
    allSynapsesDevice->tauspre[iSyn] = 0;

    allSynapsesDevice->taupos[iSyn] = 15e-3;
    allSynapsesDevice->tauneg[iSyn] = 35e-3;
    allSynapsesDevice->Wex[iSyn] = 1.0;

    allSynapsesDevice->mupos[iSyn] = 0;
    allSynapsesDevice->muneg[iSyn] = 0;

    allSynapsesDevice->useFroemkeDanSTDP[iSyn] = false;
}

/*
 *  Update PSR (post synapse response)
 *
 *  @param  allSynapsesDevice  Reference to the allSynapses struct on device memory.
 *  @param  iSyn               Index of the synapse to set.
 *  @param  simulationStep     The current simulation step.
 *  @param  deltaT             Inner simulation step duration.
 */
__device__ void changeDynamicSTDPSynapsePSR(AllDynamicSTDPSynapses* allSynapsesDevice, const BGSIZE iSyn, const uint64_t simulationStep, const BGFLOAT deltaT)
{
    uint64_t &lastSpike = allSynapsesDevice->lastSpike[iSyn];
    BGFLOAT &r = allSynapsesDevice->r[iSyn];
    BGFLOAT &u = allSynapsesDevice->u[iSyn];
    BGFLOAT D = allSynapsesDevice->D[iSyn];
    BGFLOAT F = allSynapsesDevice->F[iSyn];
    BGFLOAT U = allSynapsesDevice->U[iSyn];
    BGFLOAT W = allSynapsesDevice->W[iSyn];
    BGFLOAT &psr = allSynapsesDevice->psr[iSyn];
    BGFLOAT decay = allSynapsesDevice->decay[iSyn];

    // adjust synapse parameters
    if (lastSpike != ULONG_MAX) {
            BGFLOAT isi = (simulationStep - lastSpike) * deltaT ;
            r = 1 + ( r * ( 1 - u ) - 1 ) * exp( -isi / D );
            u = U + u * ( 1 - U ) * exp( -isi / F );
    }
    psr += ( ( W / decay ) * u * r );// calculate psr
    lastSpike = simulationStep; // record the time of the spike
}
