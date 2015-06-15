/*
 * AllDynamicSTDPSynapses_d.cu
 *
 */

#include "AllDynamicSTDPSynapses.h"
//#include "GPUSpikingModel.h"
#include "Book.h"

__global__ void getFpCreateSynapseDevice(void (**fpCreateSynapse_d)(AllDynamicSTDPSynapses*, const int, const int, int, int, int, int, BGFLOAT*, const BGFLOAT, synapseType));

__global__ void getFpChangePSRDevice(void (**fpChangePSR_d)(AllDynamicSTDPSynapses*, const uint32_t, const uint64_t, const BGFLOAT));

__device__ void createSynapse(AllDynamicSTDPSynapses* allSynapsesDevice, const int neuron_index, const int synapse_index, int source_x, int source_y, int dest_x, int dest_y, BGFLOAT *sum_point, const BGFLOAT deltaT, synapseType type);

__device__ void changePSR(AllDynamicSTDPSynapses* allSynapsesDevice, const uint32_t, const uint64_t, const BGFLOAT deltaT);

void AllDynamicSTDPSynapses::allocSynapseDeviceStruct( void** allSynapsesDevice, const SimulationInfo *sim_info ) {
	allocSynapseDeviceStruct( allSynapsesDevice, sim_info->totalNeurons, sim_info->maxSynapsesPerNeuron );
}

void AllDynamicSTDPSynapses::allocSynapseDeviceStruct( void** allSynapsesDevice, int num_neurons, int maxSynapsesPerNeuron ) {
	AllDynamicSTDPSynapses allSynapses;

	allocDeviceStruct( allSynapses, num_neurons, maxSynapsesPerNeuron );

	HANDLE_ERROR( cudaMalloc( allSynapsesDevice, sizeof( AllDynamicSTDPSynapses ) ) );
	HANDLE_ERROR( cudaMemcpy ( *allSynapsesDevice, &allSynapses, sizeof( AllDynamicSTDPSynapses ), cudaMemcpyHostToDevice ) );
}

void AllDynamicSTDPSynapses::allocDeviceStruct( AllDynamicSTDPSynapses &allSynapses, int num_neurons, int maxSynapsesPerNeuron ) {
        AllSTDPSynapses::allocDeviceStruct( allSynapses, num_neurons, maxSynapsesPerNeuron );

        uint32_t max_total_synapses = maxSynapsesPerNeuron * num_neurons;

        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.lastSpike, max_total_synapses * sizeof( uint64_t ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.r, max_total_synapses * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.u, max_total_synapses * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.D, max_total_synapses * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.U, max_total_synapses * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.F, max_total_synapses * sizeof( BGFLOAT ) ) );
}

void AllDynamicSTDPSynapses::deleteSynapseDeviceStruct( void* allSynapsesDevice ) {
	AllDynamicSTDPSynapses allSynapses;

	HANDLE_ERROR( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllDynamicSTDPSynapses ), cudaMemcpyDeviceToHost ) );

	deleteDeviceStruct( allSynapses );

	HANDLE_ERROR( cudaFree( allSynapsesDevice ) );
}

void AllDynamicSTDPSynapses::deleteDeviceStruct( AllDynamicSTDPSynapses& allSynapses ) {
        HANDLE_ERROR( cudaFree( allSynapses.lastSpike ) );
	HANDLE_ERROR( cudaFree( allSynapses.r ) );
	HANDLE_ERROR( cudaFree( allSynapses.u ) );
	HANDLE_ERROR( cudaFree( allSynapses.D ) );
	HANDLE_ERROR( cudaFree( allSynapses.U ) );
	HANDLE_ERROR( cudaFree( allSynapses.F ) );

        AllSTDPSynapses::deleteDeviceStruct( allSynapses );
}

void AllDynamicSTDPSynapses::copySynapseHostToDevice( void* allSynapsesDevice, const SimulationInfo *sim_info ) { // copy everything necessary
	copySynapseHostToDevice( allSynapsesDevice, sim_info->totalNeurons, sim_info->maxSynapsesPerNeuron );	
}

void AllDynamicSTDPSynapses::copySynapseHostToDevice( void* allSynapsesDevice, int num_neurons, int maxSynapsesPerNeuron ) { // copy everything necessary
	AllDynamicSTDPSynapses allSynapses;

        HANDLE_ERROR( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllDynamicSTDPSynapses ), cudaMemcpyDeviceToHost ) );

	copyHostToDevice( allSynapsesDevice, allSynapses, num_neurons, maxSynapsesPerNeuron );	
}

void AllDynamicSTDPSynapses::copyHostToDevice( void* allSynapsesDevice, AllDynamicSTDPSynapses& allSynapses, int num_neurons, int maxSynapsesPerNeuron ) { // copy everything necessary 
        AllSTDPSynapses::copyHostToDevice( allSynapsesDevice, allSynapses, num_neurons, maxSynapsesPerNeuron );

        uint32_t max_total_synapses = maxSynapsesPerNeuron * num_neurons;
        
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

void AllDynamicSTDPSynapses::copySynapseDeviceToHost( void* allSynapsesDevice, const SimulationInfo *sim_info ) {
	// copy everything necessary
	AllDynamicSTDPSynapses allSynapses;

        HANDLE_ERROR( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllDynamicSTDPSynapses ), cudaMemcpyDeviceToHost ) );

	copyDeviceToHost( allSynapses, sim_info );
}

void AllDynamicSTDPSynapses::copyDeviceToHost( AllDynamicSTDPSynapses& allSynapses, const SimulationInfo *sim_info ) {
        AllSTDPSynapses::copyDeviceToHost( allSynapses, sim_info ) ;

	int num_neurons = sim_info->totalNeurons;
	uint32_t max_total_synapses = sim_info->maxSynapsesPerNeuron * num_neurons;

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

/**
 *  Get synapse_counts in AllSynapses struct on device memory.
 *  @param  sim_info    SimulationInfo to refer from.
 */
void AllDynamicSTDPSynapses::copyDeviceSynapseCountsToHost(void* allSynapsesDevice, const SimulationInfo *sim_info)
{
        AllDynamicSTDPSynapses allSynapses;
        int neuron_count = sim_info->totalNeurons;

        HANDLE_ERROR( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllDynamicSTDPSynapses ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( synapse_counts, allSynapses.synapse_counts, neuron_count * sizeof( size_t ), cudaMemcpyDeviceToHost ) );
}

/** 
 *  Get summationCoord and in_use in AllSynapses struct on device memory.
 *  @param  sim_info    SimulationInfo to refer from.
 */
void AllDynamicSTDPSynapses::copyDeviceSynapseSumCoordToHost(void* allSynapsesDevice, const SimulationInfo *sim_info)
{
        AllDynamicSTDPSynapses allSynapses;
	uint32_t max_total_synapses = sim_info->maxSynapsesPerNeuron * sim_info->totalNeurons;

        HANDLE_ERROR( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllDynamicSTDPSynapses ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( summationCoord, allSynapses.summationCoord,
                max_total_synapses * sizeof( Coordinate ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( in_use, allSynapses.in_use,
                max_total_synapses * sizeof( bool ), cudaMemcpyDeviceToHost ) );
}

void AllDynamicSTDPSynapses::getFpCreateSynapse(unsigned long long& fpCreateSynapse_h)
{
    unsigned long long *fpCreateSynapse_d;

    HANDLE_ERROR( cudaMalloc(&fpCreateSynapse_d, sizeof(unsigned long long)) );

    getFpCreateSynapseDevice<<<1,1>>>((void (**)(AllDynamicSTDPSynapses*, const int, const int, int, int, int, int, BGFLOAT*, const BGFLOAT, synapseType))fpCreateSynapse_d);

    HANDLE_ERROR( cudaMemcpy(&fpCreateSynapse_h, fpCreateSynapse_d, sizeof(unsigned long long), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaFree( fpCreateSynapse_d ) );
}

void AllDynamicSTDPSynapses::getFpChangePSR(unsigned long long& fpChangePSR_h)
{
    unsigned long long *fpChangePSR_d;

    HANDLE_ERROR( cudaMalloc(&fpChangePSR_d, sizeof(unsigned long long)) );

    getFpChangePSRDevice<<<1,1>>>((void (**)(AllDynamicSTDPSynapses*, const uint32_t, const uint64_t, const BGFLOAT))fpChangePSR_d);

    HANDLE_ERROR( cudaMemcpy(&fpChangePSR_h, fpChangePSR_d, sizeof(unsigned long long), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaFree( fpChangePSR_d ) );
}

/* ------------------*\
|* # Global Functions
\* ------------------*/

__global__ void getFpCreateSynapseDevice(void (**fpCreateSynapse_d)(AllDynamicSTDPSynapses*, const int, const int, int, int, int, int, BGFLOAT*, const BGFLOAT, synapseType))
{
    *fpCreateSynapse_d = createSynapse;
}

__global__ void getFpChangePSRDevice(void (**fpChangePSR_d)(AllDynamicSTDPSynapses*, const uint32_t, const uint64_t, const BGFLOAT))
{
    *fpChangePSR_d = changePSR;
}

/**
 *  Create a Synapse and connect it to the model.
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
__device__ void createSynapse(AllDynamicSTDPSynapses* allSynapsesDevice, const int neuron_index, const int synapse_index, int source_x, int source_y, int dest_x, int dest_y, BGFLOAT *sum_point, const BGFLOAT deltaT, synapseType type)
{
    BGFLOAT delay;
    size_t max_synapses = allSynapsesDevice->maxSynapsesPerNeuron;
    uint32_t iSyn = max_synapses * neuron_index + synapse_index;

    allSynapsesDevice->in_use[iSyn] = true;
    allSynapsesDevice->summationPoint[iSyn] = sum_point;
    allSynapsesDevice->summationCoord[iSyn].x = dest_x;
    allSynapsesDevice->summationCoord[iSyn].y = dest_y;
    allSynapsesDevice->synapseCoord[iSyn].x = source_x;
    allSynapsesDevice->synapseCoord[iSyn].y = source_y;
    allSynapsesDevice->W[iSyn] = 10.0e-9;

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

    size_t size = allSynapsesDevice->total_delay[iSyn] / ( sizeof(uint8_t) * 8 ) + 1;
    assert( size <= BYTES_OF_DELAYQUEUE );

    allSynapsesDevice->Apos[iSyn] = 0.5;
    allSynapsesDevice->Aneg[iSyn] = -0.5;
    allSynapsesDevice->STDPgap[iSyn] = 2e-3;

/* TODO: these values need to be initialized
    total_delayPost = 0;
    delayQueuePost = 0;
    delayIdxPost = 0;
    ldelayQueuePost = 0;
    tauspost = 0;
    tauspre = 0;
    taupos = 0;
    tauneg = 0;
    STDPgap = 0;
    Wex = 0;
    Aneg = 0;
    Apos = 0;
    mupos = 0;
    muneg = 0;
*/
}

__device__ void changePSR(AllDynamicSTDPSynapses* allSynapsesDevice, const uint32_t iSyn, const uint64_t simulationStep, const BGFLOAT deltaT)
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
