/*
 * AllDSSynapses_d.cu
 *
 */

#include "AllDSSynapses.h"
#include "GPUSpikingModel.h"
#include "AllSynapsesDeviceFuncs.h"
#include "Book.h"

/*
 *  Allocate GPU memories to store all synapses' states,
 *  and copy them from host to GPU memory.
 *
 *  @param  allSynapsesDevice  Reference to the AllDSSynapsesDeviceProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllDSSynapses::allocSynapseDeviceStruct( void** allSynapsesDevice, const SimulationInfo *sim_info ) {
	allocSynapseDeviceStruct( allSynapsesDevice, sim_info->totalNeurons, sim_info->maxSynapsesPerNeuron );
}

/*
 *  Allocate GPU memories to store all synapses' states,
 *  and copy them from host to GPU memory.
 *
 *  @param  allSynapsesDevice     Reference to the AllDSSynapsesDeviceProperties struct 
 *                                on device memory.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllDSSynapses::allocSynapseDeviceStruct( void** allSynapsesDevice, int num_neurons, int maxSynapsesPerNeuron ) {
	AllDSSynapsesDeviceProperties allSynapses;

	allocDeviceStruct( allSynapses, num_neurons, maxSynapsesPerNeuron );

	HANDLE_ERROR( cudaMalloc( allSynapsesDevice, sizeof( AllDSSynapsesDeviceProperties ) ) );
	HANDLE_ERROR( cudaMemcpy ( *allSynapsesDevice, &allSynapses, sizeof( AllDSSynapsesDeviceProperties ), cudaMemcpyHostToDevice ) );
}

/*
 *  Allocate GPU memories to store all synapses' states,
 *  and copy them from host to GPU memory.
 *  (Helper function of allocSynapseDeviceStruct)
 *
 *  @param  allSynapsesDevice     Reference to the AllDSSynapsesDeviceProperties struct 
 *                                on device memory.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllDSSynapses::allocDeviceStruct( AllDSSynapsesDeviceProperties &allSynapses, int num_neurons, int maxSynapsesPerNeuron ) {
        AllSpikingSynapses::allocDeviceStruct( allSynapses, num_neurons, maxSynapsesPerNeuron );

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
 *  @param  allSynapsesDevice  Reference to the AllDSSynapsesDeviceProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllDSSynapses::deleteSynapseDeviceStruct( void* allSynapsesDevice ) {
	AllDSSynapsesDeviceProperties allSynapses;

	HANDLE_ERROR( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllDSSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );

	deleteDeviceStruct( allSynapses );

	HANDLE_ERROR( cudaFree( allSynapsesDevice ) );
}

/*
 *  Delete GPU memories.
 *  (Helper function of deleteSynapseDeviceStruct)
 *
 *  @param  allSynapsesDevice  Reference to the AllDSSynapsesDeviceProperties struct 
 *                             on device memory.
 */
void AllDSSynapses::deleteDeviceStruct( AllDSSynapsesDeviceProperties& allSynapses ) {
        HANDLE_ERROR( cudaFree( allSynapses.lastSpike ) );
	HANDLE_ERROR( cudaFree( allSynapses.r ) );
	HANDLE_ERROR( cudaFree( allSynapses.u ) );
	HANDLE_ERROR( cudaFree( allSynapses.D ) );
	HANDLE_ERROR( cudaFree( allSynapses.U ) );
	HANDLE_ERROR( cudaFree( allSynapses.F ) );

        AllSpikingSynapses::deleteDeviceStruct( allSynapses );
}

/*
 *  Copy all synapses' data from host to device.
 *
 *  @param  allSynapsesDevice  Reference to the AllDSSynapsesDeviceProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllDSSynapses::copySynapseHostToDevice( void* allSynapsesDevice, const SimulationInfo *sim_info ) { // copy everything necessary
	copySynapseHostToDevice( allSynapsesDevice, sim_info->totalNeurons, sim_info->maxSynapsesPerNeuron );	
}

/*
 *  Copy all synapses' data from host to device.
 *
 *  @param  allSynapsesDevice     Reference to the AllDSSynapsesDeviceProperties struct 
 *                                on device memory.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllDSSynapses::copySynapseHostToDevice( void* allSynapsesDevice, int num_neurons, int maxSynapsesPerNeuron ) { // copy everything necessary
	AllDSSynapsesDeviceProperties allSynapses;

        HANDLE_ERROR( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllDSSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );
	copyHostToDevice( allSynapsesDevice, allSynapses, num_neurons, maxSynapsesPerNeuron );	
}

/*
 *  Copy all synapses' data from host to device.
 *  (Helper function of copySynapseHostToDevice)
 *
 *  @param  allSynapsesDevice     Reference to the AllDSSynapsesDeviceProperties struct 
 *                                on device memory.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllDSSynapses::copyHostToDevice( void* allSynapsesDevice, AllDSSynapsesDeviceProperties& allSynapses, int num_neurons, int maxSynapsesPerNeuron ) { // copy everything necessary 
        AllSpikingSynapses::copyHostToDevice( allSynapsesDevice, allSynapses, num_neurons, maxSynapsesPerNeuron );

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
 *  @param  allSynapsesDevice  Reference to the AllDSSynapsesDeviceProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllDSSynapses::copySynapseDeviceToHost( void* allSynapsesDevice, const SimulationInfo *sim_info ) {
	// copy everything necessary
	AllDSSynapsesDeviceProperties allSynapses;

        HANDLE_ERROR( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllDSSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );

	copyDeviceToHost( allSynapses, sim_info );
}

/*
 *  Copy all synapses' data from device to host.
 *  (Helper function of copySynapseDeviceToHost)
 *
 *  @param  allSynapsesDevice     Reference to the AllDSSynapsesDeviceProperties struct 
 *                                on device memory.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllDSSynapses::copyDeviceToHost( AllDSSynapsesDeviceProperties& allSynapses, const SimulationInfo *sim_info ) {
        AllSpikingSynapses::copyDeviceToHost( allSynapses, sim_info ) ;

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

/**     
 *  Set synapse class ID defined by enumClassSynapses for the caller's Synapse class.
 *  The class ID will be set to classSynapses_d in device memory,
 *  and the classSynapses_d will be referred to call a device function for the
 *  particular synapse class.
 *  Because we cannot use virtual function (Polymorphism) in device functions,
 *  we use this scheme.
 *  Note: we used to use a function pointer; however, it caused the growth_cuda crash
 *  (see issue#137).
 */
void AllDSSynapses::setSynapseClassID()
{
    enumClassSynapses classSynapses_h = classAllDSSynapses;

    HANDLE_ERROR( cudaMemcpyToSymbol(classSynapses_d, &classSynapses_h, sizeof(enumClassSynapses)) );
}

/*
 *  Prints GPU SynapsesProps data.
 * 
 *  @param  allSynapsesDeviceProps   Reference to the AllDSSynapsesProps class on device memory.
 */
void AllDSSynapses::printGPUSynapsesProps( void* allSynapsesDeviceProps ) const
{
    AllDSSynapsesDeviceProperties allSynapsesProps;

    //allocate print out data members
    BGSIZE size = maxSynapsesPerNeuron * count_neurons;
    if (size != 0) {
        BGSIZE *synapse_countsPrint = new BGSIZE[count_neurons];
        BGSIZE maxSynapsesPerNeuronPrint;
        BGSIZE total_synapse_countsPrint;
        int count_neuronsPrint;
        int *sourceNeuronIndexPrint = new int[size];
        int *destNeuronIndexPrint = new int[size];
        BGFLOAT *WPrint = new BGFLOAT[size];

        synapseType *typePrint = new synapseType[size];
        BGFLOAT *psrPrint = new BGFLOAT[size];
        bool *in_usePrint = new bool[size];

        for (BGSIZE i = 0; i < size; i++) {
            in_usePrint[i] = false;
        }

        for (int i = 0; i < count_neurons; i++) {
            synapse_countsPrint[i] = 0;
        }

        BGFLOAT *decayPrint = new BGFLOAT[size];
        int *total_delayPrint = new int[size];
        BGFLOAT *tauPrint = new BGFLOAT[size];

        uint64_t *lastSpikePrint = new uint64_t[size];
        BGFLOAT *rPrint = new BGFLOAT[size];
        BGFLOAT *uPrint = new BGFLOAT[size];
        BGFLOAT *DPrint = new BGFLOAT[size];
        BGFLOAT *UPrint = new BGFLOAT[size];
        BGFLOAT *FPrint = new BGFLOAT[size];


        // copy everything
        checkCudaErrors( cudaMemcpy ( &allSynapsesProps, allSynapsesDeviceProps, sizeof( AllDSSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( synapse_countsPrint, allSynapsesProps.synapse_counts, count_neurons * sizeof( BGSIZE ), cudaMemcpyDeviceToHost ) );
        maxSynapsesPerNeuronPrint = allSynapsesProps.maxSynapsesPerNeuron;
        total_synapse_countsPrint = allSynapsesProps.total_synapse_counts;
        count_neuronsPrint = allSynapsesProps.count_neurons;

        // Set count_neurons to 0 to avoid illegal memory deallocation
        // at AllSynapsesProps deconstructor.
        allSynapsesProps.count_neurons = 0;

        checkCudaErrors( cudaMemcpy ( sourceNeuronIndexPrint, allSynapsesProps.sourceNeuronIndex, size * sizeof( int ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( destNeuronIndexPrint, allSynapsesProps.destNeuronIndex, size * sizeof( int ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( WPrint, allSynapsesProps.W, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( typePrint, allSynapsesProps.type, size * sizeof( synapseType ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( psrPrint, allSynapsesProps.psr, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( in_usePrint, allSynapsesProps.in_use, size * sizeof( bool ), cudaMemcpyDeviceToHost ) );


        checkCudaErrors( cudaMemcpy ( decayPrint, allSynapsesProps.decay, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( tauPrint, allSynapsesProps.tau, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( total_delayPrint, allSynapsesProps.total_delay,size * sizeof( int ), cudaMemcpyDeviceToHost ) );


        checkCudaErrors( cudaMemcpy ( lastSpikePrint, allSynapsesProps.lastSpike, size * sizeof( uint64_t ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( rPrint, allSynapsesProps.r, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( uPrint, allSynapsesProps.u, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( DPrint, allSynapsesProps.D, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( UPrint, allSynapsesProps.U, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( FPrint, allSynapsesProps.F, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );


        for(int i = 0; i < maxSynapsesPerNeuron * count_neurons; i++) {
            if (WPrint[i] != 0.0) {
                cout << "GPU W[" << i << "] = " << WPrint[i];
                cout << " GPU sourNeuron: " << sourceNeuronIndexPrint[i];
                cout << " GPU desNeuron: " << destNeuronIndexPrint[i];
                cout << " GPU type: " << typePrint[i];
                cout << " GPU psr: " << psrPrint[i];
                cout << " GPU in_use:" << in_usePrint[i];

                cout << " GPU decay: " << decayPrint[i];
                cout << " GPU tau: " << tauPrint[i];
                cout << " GPU total_delay: " << total_delayPrint[i];

                cout << " GPU lastSpike: " << lastSpikePrint[i];
                cout << " GPU r: " << rPrint[i];
                cout << " GPU u: " << uPrint[i];
                cout << " GPU D: " << DPrint[i];
                cout << " GPU U: " << UPrint[i];
                cout << " GPU F: " << FPrint[i] << endl;
            }
        }

        for (int i = 0; i < count_neurons; i++) {
            cout << "GPU synapse_counts:" << "neuron[" << i  << "]" << synapse_countsPrint[i] << endl;
        }

        cout << "GPU total_synapse_counts:" << total_synapse_countsPrint << endl;
        cout << "GPU maxSynapsesPerNeuron:" << maxSynapsesPerNeuronPrint << endl;
        cout << "GPU count_neurons:" << count_neuronsPrint << endl;


        // Set count_neurons to 0 to avoid illegal memory deallocation
        // at AllDSSynapsesProps deconstructor.
        allSynapsesProps.count_neurons = 0;


        delete[] destNeuronIndexPrint;
        delete[] WPrint;
        delete[] sourceNeuronIndexPrint;
        delete[] psrPrint;
        delete[] typePrint;
        delete[] in_usePrint;
        delete[] synapse_countsPrint;
        destNeuronIndexPrint = NULL;
        WPrint = NULL;
        sourceNeuronIndexPrint = NULL;
        psrPrint = NULL;
        typePrint = NULL;
        in_usePrint = NULL;
        synapse_countsPrint = NULL;

        delete[] decayPrint;
        delete[] total_delayPrint;
        delete[] tauPrint;
        decayPrint = NULL;
        total_delayPrint = NULL;
        tauPrint = NULL;

        delete[] lastSpikePrint;
        delete[] rPrint;
        delete[] uPrint;
        delete[] DPrint;
        delete[] UPrint;
        delete[] FPrint;
        lastSpikePrint = NULL;
        rPrint = NULL;
        uPrint = NULL;
        DPrint = NULL;
        UPrint = NULL;
        FPrint = NULL;
    }
}

