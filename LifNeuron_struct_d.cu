/*
 * LifNeuron_struct_d.cu
 * CUDA side struct of LifNeuron
 */

/**
 * Allocate data members in the LifNeuron_struct_d.
 * @param count
 */
void allocNeuronStruct_d( int count ) {
	LifNeuron_struct neuron;

	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.deltaT, count * sizeof( double ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.summationPoint, count * sizeof( PFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.Cm, count * sizeof( FLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.Rm, count * sizeof( FLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.Vthresh, count * sizeof( FLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.Vrest, count * sizeof( FLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.Vreset, count * sizeof( FLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.Vinit, count * sizeof( FLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.Trefract, count * sizeof( FLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.Inoise, count * sizeof( FLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.randNoise, count * sizeof( PFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.Iinject, count * sizeof( FLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.Isyn, count * sizeof( FLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.nStepsInRefr, count * sizeof( int ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.C1, count * sizeof( FLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.C2, count * sizeof( FLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.I0, count * sizeof( FLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.Vm, count * sizeof( FLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.hasFired, count * sizeof( bool ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.Tau, count * sizeof( FLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.spikeCount, count * sizeof( int ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.outgoingSynapse_begin, count * sizeof( int ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.synapseCount, count * sizeof( int ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.incomingSynapse_begin, count * sizeof( int ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.inverseCount, count * sizeof( int ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.numNeurons, count * sizeof( int ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.stepDuration, count * sizeof( int ) ) );
	
	HANDLE_ERROR( cudaMemcpyToSymbol ( neuron_st_d, &neuron, sizeof( LifNeuron_struct ) ) );
}

/**
 * Deallocate data members in the LifNeuron_struct_d.
 */
void deleteNeuronStruct_d(  ) {
	LifNeuron_struct neuron;
	HANDLE_ERROR( cudaMemcpyFromSymbol ( &neuron, neuron_st_d, sizeof( LifNeuron_struct ) ) );

	HANDLE_ERROR( cudaFree( neuron.deltaT ) );
	HANDLE_ERROR( cudaFree( neuron.summationPoint ) );
	HANDLE_ERROR( cudaFree( neuron.Cm ) );
	HANDLE_ERROR( cudaFree( neuron.Rm ) );
	HANDLE_ERROR( cudaFree( neuron.Vthresh ) );
	HANDLE_ERROR( cudaFree( neuron.Vrest ) );
	HANDLE_ERROR( cudaFree( neuron.Vreset ) );
	HANDLE_ERROR( cudaFree( neuron.Vinit ) );
	HANDLE_ERROR( cudaFree( neuron.Trefract ) );
	HANDLE_ERROR( cudaFree( neuron.Inoise ) );
	HANDLE_ERROR( cudaFree( neuron.randNoise ) );
	HANDLE_ERROR( cudaFree( neuron.Iinject ) );
	HANDLE_ERROR( cudaFree( neuron.Isyn ) );
	HANDLE_ERROR( cudaFree( neuron.nStepsInRefr ) );
	HANDLE_ERROR( cudaFree( neuron.C1 ) );
	HANDLE_ERROR( cudaFree( neuron.C2 ) );
	HANDLE_ERROR( cudaFree( neuron.I0 ) );
	HANDLE_ERROR( cudaFree( neuron.Vm ) );
	HANDLE_ERROR( cudaFree( neuron.hasFired ) );
	HANDLE_ERROR( cudaFree( neuron.Tau ) );
	HANDLE_ERROR( cudaFree( neuron.spikeCount ) );
	HANDLE_ERROR( cudaFree( neuron.outgoingSynapse_begin ) );
	HANDLE_ERROR( cudaFree( neuron.synapseCount ) );
	HANDLE_ERROR( cudaFree( neuron.incomingSynapse_begin ) );
	HANDLE_ERROR( cudaFree( neuron.inverseCount ) );
	HANDLE_ERROR( cudaFree( neuron.numNeurons ) );
	HANDLE_ERROR( cudaFree( neuron.stepDuration ) );
}

/**
 * Copy LifNeuron_struct data for GPU processing.
 * @param neuron_h
 * @param count
 */
void copyNeuronHostToDevice( LifNeuron_struct& neuron_h, int count ) {
	LifNeuron_struct neuron;
	HANDLE_ERROR( cudaMemcpyFromSymbol ( &neuron, neuron_st_d, sizeof( LifNeuron_struct ) ) );

	HANDLE_ERROR( cudaMemcpy ( neuron.deltaT, neuron_h.deltaT, count * sizeof( FLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.summationPoint, neuron_h.summationPoint, count * sizeof( PFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.Cm, neuron_h.Cm, count * sizeof( FLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.Rm, neuron_h.Rm, count * sizeof( FLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.Vthresh, neuron_h.Vthresh, count * sizeof( FLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.Vrest, neuron_h.Vrest, count * sizeof( FLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.Vreset, neuron_h.Vreset, count * sizeof( FLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.Vinit, neuron_h.Vinit, count * sizeof( FLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.Trefract, neuron_h.Trefract, count * sizeof( FLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.Inoise, neuron_h.Inoise, count * sizeof( FLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.Iinject, neuron_h.Iinject, count * sizeof( FLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.Isyn, neuron_h.Isyn, count * sizeof( FLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.nStepsInRefr, neuron_h.nStepsInRefr, count * sizeof( int ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.C1, neuron_h.C1, count * sizeof( FLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.C2, neuron_h.C2, count * sizeof( FLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.I0, neuron_h.I0, count * sizeof( FLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.Vm, neuron_h.Vm, count * sizeof( FLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.hasFired, neuron_h.hasFired, count * sizeof( bool ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.Tau, neuron_h.Tau, count * sizeof( FLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.spikeCount, neuron_h.spikeCount, count * sizeof( int ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.outgoingSynapse_begin, neuron_h.outgoingSynapse_begin, count * sizeof( int ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.synapseCount, neuron_h.synapseCount, count * sizeof( int ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.incomingSynapse_begin, neuron_h.incomingSynapse_begin, count * sizeof( int ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.inverseCount, neuron_h.inverseCount, count * sizeof( int ), cudaMemcpyHostToDevice ) );
}

/**
 * Copy data from GPU into LifNeuron_struct.
 * @param neuron_h
 * @param count
 */
void copyNeuronDeviceToHost( LifNeuron_struct& neuron_h, int count ) {
	LifNeuron_struct neuron;
	HANDLE_ERROR( cudaMemcpyFromSymbol ( &neuron, neuron_st_d, sizeof( LifNeuron_struct ) ) );

	HANDLE_ERROR( cudaMemcpy ( neuron_h.C1, neuron.C1, count * sizeof( FLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( neuron_h.C2, neuron.C2, count * sizeof( FLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( neuron_h.Cm, neuron.C1, count * sizeof( FLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( neuron_h.I0, neuron.I0, count * sizeof( FLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( neuron_h.Iinject, neuron.Iinject, count * sizeof( FLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( neuron_h.Inoise, neuron.Inoise, count * sizeof( FLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( neuron_h.Isyn, neuron.Isyn, count * sizeof( FLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( neuron_h.Rm, neuron.Rm, count * sizeof( FLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( neuron_h.Tau, neuron.Tau, count * sizeof( FLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( neuron_h.Trefract, neuron.Trefract, count * sizeof( FLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( neuron_h.Vinit, neuron.Vinit, count * sizeof( FLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( neuron_h.Vm, neuron.Vm, count * sizeof( FLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( neuron_h.Vrest, neuron.Vrest, count * sizeof( FLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( neuron_h.Vreset, neuron.Vreset, count * sizeof( FLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( neuron_h.Vthresh, neuron.Vthresh, count * sizeof( FLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( neuron_h.nStepsInRefr, neuron.nStepsInRefr, count * sizeof( int ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( neuron_h.spikeCount, neuron.spikeCount, count * sizeof( int ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( neuron_h.synapseCount, neuron.synapseCount, count * sizeof( int ), cudaMemcpyDeviceToHost ) );
}
