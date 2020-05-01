/*
 * AllDynamicSTDPSynapses_d.cu
 *
 */

#include "AllDynamicSTDPSynapses.h"
#include "AllSynapsesDeviceFuncs.h"
#include "Book.h"

/*
 *  Allocate GPU memories to store all synapses' states,
 *  and copy them from host to GPU memory.
 *
 *  @param  allSynapsesDevice  Reference to the AllDynamicSTDPSynapsesDeviceProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllDynamicSTDPSynapses::allocSynapseDeviceStruct( void** allSynapsesDevice, const SimulationInfo *sim_info ) {
	allocSynapseDeviceStruct( allSynapsesDevice, sim_info->totalNeurons, sim_info->maxSynapsesPerNeuron );
}

/*
 *  Allocate GPU memories to store all synapses' states,
 *  and copy them from host to GPU memory.
 *
 *  @param  allSynapsesDevice     Reference to the AllDynamicSTDPSynapsesDeviceProperties struct 
 *                                on device memory.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllDynamicSTDPSynapses::allocSynapseDeviceStruct( void** allSynapsesDevice, int num_neurons, int maxSynapsesPerNeuron ) {
	AllDynamicSTDPSynapsesDeviceProperties allSynapses;

	allocDeviceStruct( allSynapses, num_neurons, maxSynapsesPerNeuron );

	HANDLE_ERROR( cudaMalloc( allSynapsesDevice, sizeof( AllDynamicSTDPSynapsesDeviceProperties ) ) );
	HANDLE_ERROR( cudaMemcpy ( *allSynapsesDevice, &allSynapses, sizeof( AllDynamicSTDPSynapsesDeviceProperties ), cudaMemcpyHostToDevice ) );
}

/*
 *  Allocate GPU memories to store all synapses' states,
 *  and copy them from host to GPU memory.
 *  (Helper function of allocSynapseDeviceStruct)
 *
 *  @param  allSynapsesDevice     Reference to the AllDynamicSTDPSynapsesDeviceProperties struct 
 *                                on device memory.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllDynamicSTDPSynapses::allocDeviceStruct( AllDynamicSTDPSynapsesDeviceProperties &allSynapses, int num_neurons, int maxSynapsesPerNeuron ) {
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
 *  @param  allSynapsesDevice  Reference to the AllDynamicSTDPSynapsesDeviceProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllDynamicSTDPSynapses::deleteSynapseDeviceStruct( void* allSynapsesDevice ) {
	AllDynamicSTDPSynapsesDeviceProperties allSynapses;

	HANDLE_ERROR( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllDynamicSTDPSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );

	deleteDeviceStruct( allSynapses );

	HANDLE_ERROR( cudaFree( allSynapsesDevice ) );
}

/*
 *  Delete GPU memories.
 *  (Helper function of deleteSynapseDeviceStruct)
 *
 *  @param  allSynapsesDevice  Reference to the AllDynamicSTDPSynapsesDeviceProperties struct 
 *                             on device memory.
 */
void AllDynamicSTDPSynapses::deleteDeviceStruct( AllDynamicSTDPSynapsesDeviceProperties& allSynapses ) {
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
 *  @param  allSynapsesDevice  Reference to the AllDynamicSTDPSynapsesDeviceProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllDynamicSTDPSynapses::copySynapseHostToDevice( void* allSynapsesDevice, const SimulationInfo *sim_info ) { // copy everything necessary
	copySynapseHostToDevice( allSynapsesDevice, sim_info->totalNeurons, sim_info->maxSynapsesPerNeuron );	
}

/*
 *  Copy all synapses' data from host to device.
 *
 *  @param  allSynapsesDevice     Reference to the AllDynamicSTDPSynapsesDeviceProperties struct 
 *                                on device memory.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllDynamicSTDPSynapses::copySynapseHostToDevice( void* allSynapsesDevice, int num_neurons, int maxSynapsesPerNeuron ) { // copy everything necessary
	AllDynamicSTDPSynapsesDeviceProperties allSynapses;

        HANDLE_ERROR( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllDynamicSTDPSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );

	copyHostToDevice( allSynapsesDevice, allSynapses, num_neurons, maxSynapsesPerNeuron );	
}

/*
 *  Copy all synapses' data from host to device.
 *  (Helper function of copySynapseHostToDevice)
 *
 *  @param  allSynapsesDevice     Reference to the AllDynamicSTDPSynapsesDeviceProperties struct 
 *                                on device memory.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllDynamicSTDPSynapses::copyHostToDevice( void* allSynapsesDevice, AllDynamicSTDPSynapsesDeviceProperties& allSynapses, int num_neurons, int maxSynapsesPerNeuron ) { // copy everything necessary 
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
 *  @param  allSynapsesDevice  Reference to the AllDynamicSTDPSynapsesDeviceProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllDynamicSTDPSynapses::copySynapseDeviceToHost( void* allSynapsesDevice, const SimulationInfo *sim_info ) {
	// copy everything necessary
	AllDynamicSTDPSynapsesDeviceProperties allSynapses;

        HANDLE_ERROR( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllDynamicSTDPSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );

	copyDeviceToHost( allSynapses, sim_info );
}

/*
 *  Copy all synapses' data from device to host.
 *  (Helper function of copySynapseDeviceToHost)
 *
 *  @param  allSynapsesDevice     Reference to the AllDynamicSTDPSynapsesDeviceProperties struct 
 *                                on device memory.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllDynamicSTDPSynapses::copyDeviceToHost( AllDynamicSTDPSynapsesDeviceProperties& allSynapses, const SimulationInfo *sim_info ) {
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
void AllDynamicSTDPSynapses::setSynapseClassID()
{
    enumClassSynapses classSynapses_h = classAllDynamicSTDPSynapses;

    HANDLE_ERROR( cudaMemcpyToSymbol(classSynapses_d, &classSynapses_h, sizeof(enumClassSynapses)) );
}

/*
 *  Prints GPU SynapsesProps data.
 *   
 *  @param  allSynapsesDeviceProps   Reference to the corresponding SynapsesDeviceProperties struct on device memory.
 */
void AllDynamicSTDPSynapses::printGPUSynapsesProps( void* allSynapsesDeviceProps ) const
{
    AllDynamicSTDPSynapsesDeviceProperties allSynapsesProps;

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

        int *total_delayPostPrint = new int[size];
        BGFLOAT *tauspostPrint = new BGFLOAT[size];
        BGFLOAT *tausprePrint = new BGFLOAT[size];
        BGFLOAT *tauposPrint = new BGFLOAT[size];
        BGFLOAT *taunegPrint = new BGFLOAT[size];
        BGFLOAT *STDPgapPrint = new BGFLOAT[size];
        BGFLOAT *WexPrint = new BGFLOAT[size];
        BGFLOAT *AnegPrint = new BGFLOAT[size];
        BGFLOAT *AposPrint = new BGFLOAT[size];
        BGFLOAT *muposPrint = new BGFLOAT[size];
        BGFLOAT *munegPrint = new BGFLOAT[size];
        bool *useFroemkeDanSTDPPrint = new bool[size];

        uint64_t *lastSpikePrint = new uint64_t[size];
        BGFLOAT *rPrint = new BGFLOAT[size];
        BGFLOAT *uPrint = new BGFLOAT[size];
        BGFLOAT *DPrint = new BGFLOAT[size];
        BGFLOAT *UPrint = new BGFLOAT[size];
        BGFLOAT *FPrint = new BGFLOAT[size];

        // copy everything
        HANDLE_ERROR( cudaMemcpy ( &allSynapsesProps, allSynapsesDeviceProps, sizeof( AllDynamicSTDPSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( synapse_countsPrint, allSynapsesProps.synapse_counts, count_neurons * sizeof( BGSIZE ), cudaMemcpyDeviceToHost ) );
        maxSynapsesPerNeuronPrint = allSynapsesProps.maxSynapsesPerNeuron;
        total_synapse_countsPrint = allSynapsesProps.total_synapse_counts;
        count_neuronsPrint = allSynapsesProps.count_neurons;

        // Set count_neurons to 0 to avoid illegal memory deallocation
        // at AllSynapsesProps deconstructor.
        allSynapsesProps.count_neurons = 0;

        HANDLE_ERROR( cudaMemcpy ( sourceNeuronIndexPrint, allSynapsesProps.sourceNeuronIndex, size * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( destNeuronIndexPrint, allSynapsesProps.destNeuronIndex, size * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( WPrint, allSynapsesProps.W, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( typePrint, allSynapsesProps.type, size * sizeof( synapseType ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( psrPrint, allSynapsesProps.psr, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( in_usePrint, allSynapsesProps.in_use, size * sizeof( bool ), cudaMemcpyDeviceToHost ) );

        HANDLE_ERROR( cudaMemcpy ( decayPrint, allSynapsesProps.decay, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( tauPrint, allSynapsesProps.tau, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( total_delayPrint, allSynapsesProps.total_delay,size * sizeof( int ), cudaMemcpyDeviceToHost ) );

        HANDLE_ERROR( cudaMemcpy ( total_delayPostPrint, allSynapsesProps.total_delayPost, size * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( tauspostPrint, allSynapsesProps.tauspost, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( tausprePrint, allSynapsesProps.tauspre, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( tauposPrint, allSynapsesProps.taupos, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( taunegPrint, allSynapsesProps.tauneg, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( STDPgapPrint, allSynapsesProps.STDPgap, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( WexPrint, allSynapsesProps.Wex, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( AnegPrint, allSynapsesProps.Aneg, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( AposPrint, allSynapsesProps.Apos, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( muposPrint, allSynapsesProps.mupos, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( munegPrint, allSynapsesProps.muneg, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( useFroemkeDanSTDPPrint, allSynapsesProps.useFroemkeDanSTDP, size * sizeof( bool ), cudaMemcpyDeviceToHost ) );

        HANDLE_ERROR( cudaMemcpy ( lastSpikePrint, allSynapsesProps.lastSpike, size * sizeof( uint64_t ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( rPrint, allSynapsesProps.r, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( uPrint, allSynapsesProps.u, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( DPrint, allSynapsesProps.D, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( UPrint, allSynapsesProps.U, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( FPrint, allSynapsesProps.F, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );

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

                cout << " GPU total_delayPost: " << total_delayPostPrint[i];
                cout << " GPU tauspost: " << tauspostPrint[i];
                cout << " GPU tauspre: " << tausprePrint[i];
                cout << " GPU taupos: " << tauposPrint[i];
                cout << " GPU tauneg: " << taunegPrint[i];
                cout << " GPU STDPgap: " << STDPgapPrint[i];
                cout << " GPU Wex: " << WexPrint[i];
                cout << " GPU Aneg: " << AnegPrint[i];
                cout << " GPU Apos: " << AposPrint[i];
                cout << " GPU mupos: " << muposPrint[i];
                cout << " GPU muneg: " << munegPrint[i];
                cout << " GPU useFroemkeDanSTDP: " << useFroemkeDanSTDPPrint[i];

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

        delete[] total_delayPostPrint;
        delete[] tauspostPrint;
        delete[] tausprePrint;
        delete[] tauposPrint;
        delete[] taunegPrint;
        delete[] STDPgapPrint;
        delete[] WexPrint;
        delete[] AnegPrint;
        delete[] AposPrint;
        delete[] muposPrint;
        delete[] munegPrint;
        delete[] useFroemkeDanSTDPPrint;
        total_delayPostPrint = NULL;
        tauspostPrint = NULL;
        tausprePrint = NULL;
        tauposPrint = NULL;
        taunegPrint = NULL;
        STDPgapPrint = NULL;
        WexPrint = NULL;
        AnegPrint = NULL;
        AposPrint = NULL;
        muposPrint = NULL;
        munegPrint = NULL;
        useFroemkeDanSTDPPrint = NULL;

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

