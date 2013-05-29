/** \file AMP_LIFModel.cpp
 ** 
 ** \derived from CUDA_LIFModel.cu
 **
 ** \authors Paul Bunn
 **
 ** \brief Functions that perform the GPU version of simulation
 ** \using Microsoft's C++ AMP
 **/

#define _AMP_LIFModel
using namespace std;

#include <vector>
#include <random>
#include <math.h>
#include <amp.h>
#include <amp_math.h>

#include "../tinyxml/tinyxml.h"

#include "ParseParamError.h"
#include "Util.h"
#include "Global.h"
#include "AMP_LIFModel.h"
#include "../cuda/MersenneTwisterGPU.h"

using namespace concurrency;

// This is Visual Studio's CPU implementation of Mersenne Twister
// It is only used by this code to seed the GPU version of MT

typedef minstd_rand CPU_MT; 
typedef std::mersenne_twister<unsigned int, 32, 624, 
    397, 31, 0x9908b0df, 11, 7, 0x9d2c5680, 
    15, 0xefc60000, 18> CPU_MT_ENG;  // same as mt19937 

AMP_LIFModel::AMP_LIFModel()
{
}

AMP_LIFModel::~AMP_LIFModel()
{
#ifdef STORE_SPIKEHISTORY
    delete[] spikeArray;
#endif // STORE_SPIKEHISTORY
	delete[] m_conns->spikeCounts;
	m_conns->spikeCounts = NULL;
}

bool AMP_LIFModel::initializeModel(SimulationInfo *sim_info, AllNeurons& neurons, AllSynapses& synapses)
{
	//initialize Mersenne Twister
	//assuming neuron_count >= 100 and is a multiple of 100. Note rng_mt_rng_count must be <= MT_RNG_COUNT
	uint32_t rng_blocks = 256; //# of blocks the kernel will use
	uint32_t rng_nPerRng = 16; //# of iterations per thread (thread granularity, # of rands generated per thread)
	uint32_t rng_mt_rng_count = 4096;//sim_info->cNeurons/rng_nPerRng; //# of threads to generate for neuron_count rand #s
	uint32_t rng_threads = rng_mt_rng_count/rng_blocks; //# threads per block needed
	initMTGPU_AMP(777, rng_blocks, rng_threads, rng_nPerRng, rng_mt_rng_count);
	randNoise_d.resize(sim_info->cNeurons);
	return true;
}


////////////////////////////////////////////////////////////////////////////////
// Write MT_RNG_COUNT vertical lanes of n_per_RNG random numbers to random_nums.
// For coalesced global writes MT_RNG_COUNT should be a multiple of hardware scehduling unit size.
// Hardware scheduling unit is called warp or wave or wavefront
// Initial states for each generator are the same, since the states are
// initialized from the global seed. In order to improve distribution properties
// on small n_per_RNG supply dedicated (local) seed to each twister.
// The local seeds, in their turn, can be extracted from global seed
// by means of any simple random number generator, like LCG.
////////////////////////////////////////////////////////////////////////////////
void rand_MT_kernel(index<1> idx,
               array<float, 2>& random_nums, 
               const unsigned int matrix_a, 
               const unsigned int mask_b, const unsigned int mask_c, 
               const unsigned int seed, const int n_per_RNG) restrict(amp)
{
    int state_1;
    int state_M;
    unsigned int mti, mti_M, x;
    unsigned int mti_1, mt[MT_NN];
	bool boxFlag = false;	//will perform boxmuller transform on true	
	float regVal1, regVal2;	//need 2 values for boxmuller

    //Bit-vector Mersenne Twister parameters are in matrix_a, mask_b, mask_c, seed
    //Initialize current state
    mt[0] = seed;
    for(int state = 1; state < MT_NN; state++)
        mt[state] = (1812433253U * (mt[state - 1] ^ (mt[state - 1] >> 30)) + state) & MT_WMASK;

    mti_1 = mt[0];
    for(int out = 0, state = 0; out < n_per_RNG; out++) 
    {
        state_1 = state + 1;
        state_M = state + MT_MM;
        if (state_1 >= MT_NN) state_1 -= MT_NN;
        if (state_M >= MT_NN) state_M -= MT_NN;
        mti  = mti_1;
        mti_1 = mt[state_1];
        mti_M = mt[state_M];

        x    = (mti & MT_UMASK) | (mti_1 & MT_LMASK);
        x    =  mti_M ^ (x >> 1) ^ ((x & 1) ? matrix_a : 0);
        mt[state] = x;
        state = state_1;

        //Tempering transformation
        x ^= (x >> MT_SHIFT0);
        x ^= (x << MT_SHIFTB) & mask_b;
        x ^= (x << MT_SHIFTC) & mask_c;
        x ^= (x >> MT_SHIFT1);

		// This is why the number of RNGs gernerated must be even:
		if(boxFlag){
			float r, phi;
			regVal2 = ((float)x + 1.0f) / 4294967296.0f;
			r = fast_math::sqrtf(-2.0f * fast_math::logf(regVal1));
			phi = 2 * 3.1415926535897f * regVal2;
			regVal1 = r * fast_math::cosf(phi);
			regVal2 = r * fast_math::sinf(phi);
			random_nums[index<2>(out, idx[0])] = regVal2;
			boxFlag = false;
		}else{
			regVal1 = ((float)x + 1.0f) / 4294967296.0f;
			random_nums[index<2>(out, idx[0])] = regVal1;
			boxFlag = true;
		}
    }
}

void advance_neurons_amp_kernel(index<1> idx,
               const array<float, 2>& random_nums) restrict(amp)
{
}

void AMP_LIFModel::advance(AllNeurons &neurons, AllSynapses &synapses, SimulationInfo *psi)
{
	CPU_MT MT_eng; 
    CPU_MT_ENG MT_ceng; 
    CPU_MT_ENG::result_type compval = MT_ceng(); 
	int n_per_RNG = mt_nPerRng;
	static bool firstEntry = true;

#ifdef STORE_SPIKEHISTORY
	uint32_t maxSpikes = static_cast<uint32_t> (psi->stepDuration * psi->maxFiringRate);
#endif // STORE_SPIKEHISTORY

	FLOAT deltaT = psi->deltaT;
	uint32_t width = psi->width;
	uint32_t neuron_count = psi->cNeurons;
	uint32_t synapse_count = neuron_count * psi->maxSynapsesPerNeuron;

    // simulate to next growth cycle
    uint64_t endStep = g_simulationStep + static_cast<uint64_t>(psi->stepDuration / deltaT);
	uint64_t count = 0;

	if(firstEntry) {
		firstEntry = false;
		MT_ceng.seed(1); // reseed base engine 
	}

	DEBUG(cout << "Beginning GPU sim cycle, simTime = " << g_simulationStep * deltaT << ", endTime = " << endStep * deltaT << endl;)

	assert((n_per_RNG & 1) == 0); // ensure it's even -- odd not allowed
	while ( g_simulationStep < endStep )
	{
		Concurrency::extent<1> e_c(v_matrix.size());
		Concurrency::extent<2> rn(mt_nPerRng, mt_rng_count);

		DEBUG( if(count % 10000 == 0) {
				cout << psi->currentStep << "/" << psi->maxSteps
						<< " simulating time: " << g_simulationStep * deltaT << endl;
				count = 0;
			}
			count++; )

		reseed_MTGPU_AMP(MT_ceng());

		array<float, 2> random_nums(rn); 
		array<float, 2> normalized_random_nums(rn);

		// Copy to GPU
		array<unsigned int, 1> matrix_a(e_c, v_matrix.begin());
		array<unsigned int, 1> seed(e_c, v_seed.begin());
		array<unsigned int, 1> mask_b(e_c, v_mask_b.begin());
		array<unsigned int, 1> mask_c(e_c, v_mask_c.begin());

		// generate random numbers
		parallel_for_each(e_c, [=, &random_nums, &matrix_a, &mask_b, &mask_c, &seed] (index<1> idx) restrict(amp)
		{
			rand_MT_kernel(idx, random_nums, matrix_a[idx], mask_b[idx], mask_c[idx], seed[idx], n_per_RNG);
		});

		Concurrency::extent<1> numN(neuron_count);
		array_view<uint32_t> v_hasFired(neuron_count, neurons.hasFired);
		array_view<uint32_t> v_nStepsInRefr(neuron_count, neurons.nStepsInRefr);
		parallel_for_each(numN, [=](index<1> idx) restrict(amp)
		{
			uint32_t temp = v_hasFired[idx];
			v_hasFired[idx] = v_nStepsInRefr[idx];
			v_nStepsInRefr[idx] = temp;
		});
		v_hasFired.synchronize();
		v_nStepsInRefr.synchronize();
	}

	return;
}

void AMP_LIFModel::updateWeights(const uint32_t neuron_count, AllNeurons &neurons, AllSynapses &synapses, SimulationInfo *sim_info)
{
}

