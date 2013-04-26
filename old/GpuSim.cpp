/**
 *      \file GpuSim.cpp
 *      
 *      \author Fumitaka Kawasaki
 *
 *      \brief A class that performs the simulation on GPU.
 */
#include "GpuSim.h"

extern "C" {
void advanceGPU( 
#ifdef STORE_SPIKEHISTORY
		SimulationInfo* psi,
		int maxSynapses,
		uint64_t* spikeArray,
		int maxSpikes
#else
		SimulationInfo* psi, 
		int maxSynapses
#endif // STORE_SPIKEHISTORY
		);

void allocDeviceStruct(SimulationInfo* psi,
		LifNeuron_struct* neuron_st, 
		DynamicSpikingSynapse_struct* synapse_st,
#ifdef STORE_SPIKEHISTORY
		int maxSynapses,
		int maxSpikes
#else
		int maxSynapses
#endif // STORE_SPIKEHISTORY
		);

void copySynapseDeviceToHost( DynamicSpikingSynapse_struct* synapse_h, int count );

void copyNeuronDeviceToHost( LifNeuron_struct* neuron_h, int count );

//void initMTGPU(int seed, int mt_rng_count);
void initMTGPU(unsigned int seed, unsigned int blocks, unsigned int threads, unsigned int nPerRng, unsigned int mt_rng_count);

void deleteDeviceStruct( );

void getSpikeCounts(int neuron_count, int* spikeCounts);

void clearSpikeCounts(int neuron_count);

void updateNetworkGPU(SimulationInfo* psi, CompleteMatrix& W, int maxSynapses);
}

/** 
 * Allocate all matixes that are used for updating the network.
 * Allocate neuron and synapse structures and copy neuron and synapse data into the structures.
 * Allocate device memories and copy neuron and synapse data into the memory.  
 * Initialize the Mersenne Twister (random number generator).
 * The variable maxFiringRate in SimulationInfo determines the size of an array to store spikes.
 * The variable maxSynapsesPerNeuron in SImulationInfo determines the size of synapse array in device memory.
 * @param[in] psi	Pointer to the simulation information.
 * @post All data structures are allocated. 
 */
GpuSim::GpuSim(SimulationInfo* psi) :
        W("complete", "const", psi->cNeurons, psi->cNeurons, 0),
        radii("complete", "const", 1, psi->cNeurons, psi->startRadius),
        rates("complete", "const", 1, psi->cNeurons, 0),
        dist2("complete", "const", psi->cNeurons, psi->cNeurons),
        delta("complete", "const", psi->cNeurons, psi->cNeurons),
        dist("complete", "const", psi->cNeurons, psi->cNeurons),
        area("complete", "const", psi->cNeurons, psi->cNeurons, 0),
        outgrowth("complete", "const", 1, psi->cNeurons),
        deltaR("complete", "const", 1, psi->cNeurons)
{
	LifNeuron_struct* neuron_st;
	DynamicSpikingSynapse_struct* synapse_st;

	// copy synapse and neuron maps into arrays
	dataToCStructs(psi, neuron_st, synapse_st);

	int neuron_count = psi->cNeurons;
	spikeCounts = new int[neuron_count];

#ifdef STORE_SPIKEHISTORY
        int maxSpikes = static_cast<int> (psi->stepDuration * psi->maxFiringRate);
        spikeArray = new uint64_t[maxSpikes * neuron_count];

	// allocate device memory
	allocDeviceStruct(psi, neuron_st, synapse_st, psi->maxSynapsesPerNeuron, maxSpikes);
#else
	// allocate device memory
	allocDeviceStruct(psi, neuron_st, synapse_st, psi->maxSynapsesPerNeuron);
#endif // STORE_SPIKEHISTORY

	//initialize Mersenne Twister
	//assuming neuron_count >= 100 and is a multiple of 100. Note rng_mt_rng_count must be <= MT_RNG_COUNT
	int rng_blocks = 25; //# of blocks the kernel will use
	int rng_nPerRng = 4; //# of iterations per thread (thread granularity, # of rands generated per thread)
	int rng_mt_rng_count = neuron_count/rng_nPerRng; //# of threads to generate for neuron_count rand #s
	int rng_threads = rng_mt_rng_count/rng_blocks; //# threads per block needed
	initMTGPU(777, rng_blocks, rng_threads, rng_nPerRng, rng_mt_rng_count);

	// make sure neuron objects and neuron arrays match
	DEBUG_MID( printComparison(neuronArray, psi->pNeuronList, psi->cNeurons) );

	// delete the arrays
	deleteNeuronStruct(neuron_st);
	deleteSynapseStruct(synapse_st);
}

GpuSim::~GpuSim()
{
#ifdef STORE_SPIKEHISTORY
        delete[] spikeArray;
#endif // STORE_SPIKEHISTORY
	delete[] spikeCounts;
	deleteDeviceStruct();
}

/**
 * Compute dist2, dist and delta.
 * @param[in] psi	Pointer to the simulation information.	
 * @param[in] xloc	X location of neurons.
 * @param[in] yloc	Y location of neurons.
 */
void GpuSim::init(SimulationInfo* psi, VectorMatrix& xloc, VectorMatrix& yloc)
{
    // calculate the distance between neurons
    for (int n = 0; n < psi->cNeurons - 1; n++)
    {
        for (int n2 = n + 1; n2 < psi->cNeurons; n2++)
        {            // distance^2 between two points in point-slope form
            dist2(n, n2) = (xloc[n] - xloc[n2]) * (xloc[n] - xloc[n2]) +
                (yloc[n] - yloc[n2]) * (yloc[n] - yloc[n2]); 

            // both points are equidistant from each other
            dist2(n2, n) = dist2(n, n2);
        }
    }
    
    // take the square root to get actual distance (Pythagoras was right!)
    // (The CompleteMatrix class makes this assignment look so easy...)
    dist = sqrt(dist2);
    
    // Init connection frontier distance change matrix with the current distances
    delta = dist;
}

/**
 * Terminate process.
 * @param[in] psi	Pointer to the simulation information.
 */
void GpuSim::term(SimulationInfo* psi)
{
    LifNeuron_struct* neuron_st;
    DynamicSpikingSynapse_struct* synapse_st;

    // Allocate memory
    int neuron_count = psi->cNeurons;
    allocNeuronStruct(neuron_st, neuron_count);
    allocSynapseStruct(synapse_st, neuron_count * psi->maxSynapsesPerNeuron);

    // copy device synapse and neuron structs to host memory
    copySynapseDeviceToHost( synapse_st, neuron_count * psi->maxSynapsesPerNeuron );
    copyNeuronDeviceToHost( neuron_st, neuron_count );

    // copy synapse and neuron arrays back into their respectrive maps
    synapseArrayToMap(synapse_st, psi->rgSynapseMap, psi->cNeurons, psi->maxSynapsesPerNeuron);
    neuronArrayToMap(neuron_st, psi->pNeuronList, psi->cNeurons);

    // delete the arrays
    deleteNeuronStruct(neuron_st);
    deleteSynapseStruct(synapse_st);
}

/** 
 * Set initial radii data
 * @param[in] newRadii  Radii data to set
 */ 
void GpuSim::initRadii(VectorMatrix& newRadii)
{
    radii = newRadii;
}

/**
 * Call GPU kernel functions to simultaneously update neurons and synapses.
 * @param[in] psi	Pointer to the simulation information. 
 */
void GpuSim::advanceUntilGrowth(SimulationInfo* psi)
{
#ifdef STORE_SPIKEHISTORY
        int maxSpikes = static_cast<int> (psi->stepDuration * psi->maxFiringRate);
#endif // STORE_SPIKEHISTORY

#ifdef STORE_SPIKEHISTORY
	advanceGPU(psi, psi->maxSynapsesPerNeuron, spikeArray, maxSpikes);
#else
	advanceGPU(psi, psi->maxSynapsesPerNeuron);
#endif // STORE_SPIKEHISTORY

#ifdef STORE_SPIKEHISTORY
	int neuron_count = psi->cNeurons;

	// record spike time
	getSpikeCounts(neuron_count, spikeCounts);
	for (int i = 0; i < neuron_count; i++) {
		INeuron* neuron = (*(psi->pNeuronList))[i];
		if (spikeCounts[i] > 0) {
			assert(spikeCounts[i] < maxSpikes);
			neuron->spikeHistory.insert(neuron->spikeHistory.end(), 
					&(spikeArray[i * maxSpikes]), 
					&(spikeArray[(i * maxSpikes) + spikeCounts[i]])); 
		}
	}
#endif // STORE_SPIKEHISTORY
}

/**
 * Copy synapse and neuron C++ objects into C structs
 * @param psi simulation info
 * @param [out] synapse_st object stores synapse data in arrays
 * @param [out] neuron_st object stores neuron data in arrays
 * @param [out] neuron_count
 * @param [out] synapse_count
 */ 
void GpuSim::dataToCStructs( SimulationInfo* psi, LifNeuron_struct* neuron_st, DynamicSpikingSynapse_struct* synapse_st ) 
{
	// count the synapses
	int synapse_count = 0;
	for (int i = 0; i < psi->cNeurons; i++)
	{
		synapse_count += psi->rgSynapseMap[i].size();
	}

	// Allocate memory
	int neuron_count = psi->cNeurons;
	allocNeuronStruct(neuron_st, neuron_count);
	allocSynapseStruct(synapse_st, neuron_count * psi->maxSynapsesPerNeuron);

	// Copy memory
	for (int i = 0; i < neuron_count; i++)
	{
		INeuron* pNeuron = (*psi->pNeuronList)[i];

		copyNeuronToStruct(pNeuron, neuron_st, i);
		neuron_st->synapseCount[i] = psi->rgSynapseMap[i].size();
		assert(neuron_st->synapseCount[i] <= psi->maxSynapsesPerNeuron);
		neuron_st->outgoingSynapse_begin[i] = i * psi->maxSynapsesPerNeuron;

		for (unsigned int j = 0; j < psi->rgSynapseMap[i].size(); j++)		
			copySynapseToStruct(psi->rgSynapseMap[i][j], synapse_st, i * psi->maxSynapsesPerNeuron + j);		
	}
}

void GpuSim::printComparison(LifNeuron_struct* neuron_st, vector<LifNeuron*>* neuronObjects, int neuronCount){
	cout << "--------------------------------------" << endl << endl;
	for (int i = 0; i < neuronCount; i++){
		LifNeuron* neuron = (*neuronObjects)[i];
		cout << "Neuron Object #" << i << endl;
		cout << "C1: " << neuron->C1 << endl;
		cout << "C2: " << neuron->C2 << endl;
		cout << "I0: " << neuron->I0 << endl;
		cout << "Inoise: " << neuron->Inoise << endl;
		cout << "Trefract: " << neuron->Trefract << endl;
		cout << "Vm: " << neuron->Vm << endl;
		cout << "Vthresh: " << neuron->Vthresh << endl;
		cout << "hasFired: " << neuron->hasFired << endl;
		cout << "nStepsInRefr: " << neuron->nStepsInRefr << endl;

		cout << "Neuron Array element #" << i << endl;
		cout << "C1: " << neuron_st->C1[i] << endl;
		cout << "C2: " << neuron_st->C2[i] << endl;
		cout << "I0: " << neuron_st->I0[i] << endl;
		cout << "Inoise: " << neuron_st->Inoise[i] << endl;
		cout << "Trefract: " << neuron_st->Trefract[i] << endl;
		cout << "Vm: " << neuron_st->Vm[i] << endl;
		cout << "Vthresh: " << neuron_st->Vthresh[i] << endl;
		cout << "nStepsInRefr: " << neuron_st->nStepsInRefr[i] << endl;
		cout << "spikeCount: " << neuron_st->spikeCount[i] << endl << endl;
		
		cout << "--------------------------------------" << endl << endl;
	}
}

/**
 * Calculate growth cycle firing rate for previous period.
 * Compute neuron radii change, assign new values, and record the radius to histroy matrix.
 * Update distance between frontiers, and compute areas of overlap. 
 * Adjust the strength of the synapse or remove it from the synapse map if it has gone below 
 * zero, that is done by GPU kernel functions simultaneously.
 * @param[in] psi	Pointer to the simulation information.
 * @param[in] radiiHistory	Matrix to save radius history.
 * @param[in] ratesHistory	Matrix to save firing rate history.
 */
void GpuSim::updateNetwork(SimulationInfo* psi, CompleteMatrix& radiiHistory, CompleteMatrix& ratesHistory)
{
    int neuron_count = psi->cNeurons;

    // Calculate growth cycle firing rate for previous period
    getSpikeCounts(neuron_count, spikeCounts);

    for (int i = 0; i < neuron_count; i++)
    {
        // Calculate firing rate
        rates[i] = spikeCounts[i] / psi->stepDuration;

        // record firing rate to history matrix
        ratesHistory(psi->currentStep, i) = rates[i];
    }

    // clear spike count
    clearSpikeCounts(neuron_count);

    // compute neuron radii change and assign new values
    outgrowth = 1.0 - 2.0 / (1.0 + exp((psi->epsilon - rates / psi->maxRate) / psi->beta));
    deltaR = psi->stepDuration * psi->rho * outgrowth;
    radii += deltaR;

    // Cap minimum radius size and record radii to history matrix
    for (int i = 0; i < radii.Size(); i++)
    {
        // TODO: find out why we cap this here.
        if (radii[i] < psi->minRadius)
            radii[i] = psi->minRadius;

        // record radius to history matrix
        radiiHistory(psi->currentStep, i) = radii[i];

        DEBUG_MID(cout << "radii[" << i << ":" << radii[i] << "]" << endl;);
    }

    DEBUG(cout << "Updating distance between frontiers..." << endl;)
    // Update distance between frontiers
    for (int unit = 0; unit < psi->cNeurons - 1; unit++)
    {
        for (int i = unit + 1; i < psi->cNeurons; i++)
        {
            delta(unit, i) = dist(unit, i) - (radii[unit] + radii[i]);
            delta(i, unit) = delta(unit, i);
        }
    }

    DEBUG(cout << "computing areas of overlap" << endl;)

    // Compute areas of overlap; this is only done for overlapping units
    for (int i = 0; i < psi->cNeurons; i++)
    {
        for (int j = 0; j < psi->cNeurons; j++)
        {
            area(i, j) = 0.0;

            if (delta(i, j) < 0)
            {
                BGFLOAT lenAB = dist(i, j);
                BGFLOAT r1 = radii[i];
                BGFLOAT r2 = radii[j];

                if (lenAB + min(r1, r2) <= max(r1, r2))
                {
                    area(i, j) = pi * min(r1, r2) * min(r1, r2); // Completely overlapping unit
#ifdef LOGFILE
                    logFile << "Completely overlapping (i, j, r1, r2, area): "
                    << i << ", " << j << ", " << r1 << ", " << r2 << ", " << *pAarea(i, j) << endl;
#endif // LOGFILE
                }
                else
                {
                    // Partially overlapping unit
                    BGFLOAT lenAB2 = dist2(i, j);
                    BGFLOAT r12 = r1 * r1;
                    BGFLOAT r22 = r2 * r2;

                    BGFLOAT cosCBA = (r22 + lenAB2 - r12) / (2.0 * r2 * lenAB);
                    BGFLOAT angCBA = acos(cosCBA);
                    BGFLOAT angCBD = 2.0 * angCBA;

                    BGFLOAT cosCAB = (r12 + lenAB2 - r22) / (2.0 * r1 * lenAB);
                    BGFLOAT angCAB = acos(cosCAB);
                    BGFLOAT angCAD = 2.0 * angCAB;

                    area(i, j) = 0.5 * (r22 * (angCBD - sin(angCBD)) + r12 * (angCAD - sin(angCAD)));
                }
            }
        }
    }

    // For now, we just set the weights to equal the areas. We will later
    // scale it and set its sign (when we index and get its sign).
    W = area;

    DEBUG(cout << "adjusting weights" << endl;)

    updateNetworkGPU( psi, W, psi->maxSynapsesPerNeuron );

}

/**
 * Returns a type of Neuron to be used in the Network
 */
INeuron* GpuSim::returnNeuron()
{
	return new LifNeuron();
}
