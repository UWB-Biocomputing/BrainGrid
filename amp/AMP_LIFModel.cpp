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

#include <vector>
#include <random>

#include "../tinyxml/tinyxml.h"

#include "ParseParamError.h"
#include "Util.h"
#include "Global.h"
#include "AMP_LIFModel.h"
#include "../cuda/MersenneTwisterCUDA.h"

struct
{
	int x,y,z,start;  
} mydatastruct;

typedef std::minstd_rand Myeng; 
typedef std::mersenne_twister<unsigned int, 32, 624, 
    397, 31, 0x9908b0df, 11, 7, 0x9d2c5680, 
    15, 0xefc60000, 18> Myceng;  // same as mt19937 

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


void AMP_LIFModel::advance(AllNeurons &neurons, AllSynapses &synapses, SimulationInfo *sim_info)
{
	Myeng eng; 
    Myceng ceng; 
    Myceng::result_type compval = ceng(); 

	ceng.seed(1); // reseed base engine 

	randNoise_d.resize(65536);

	for(int i = 0; i < 256 ; i++) {
		generate_rand_on_amp(randNoise_d);
		reseed_MTGPU_AMP(ceng());
	}

	return;
}

void AMP_LIFModel::updateWeights(const uint32_t neuron_count, AllNeurons &neurons, AllSynapses &synapses, SimulationInfo *sim_info)
{
}