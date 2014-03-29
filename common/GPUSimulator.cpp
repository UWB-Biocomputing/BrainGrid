#include "GPUSimulator.h"

//void initMTGPU(int seed, int mt_rng_count);
void initMTGPU(unsigned int seed, unsigned int blocks, unsigned int threads, unsigned int nPerRng, unsigned int mt_rng_count);

/**
 *  Constructor
 *  @param  network 
 *          pointer to a neural network implementation to be simulated by BrainGrid. (It would be
 *          nice if this was a parameter to #simulate). Note: this reference will not be deleted.
 *  @param  sim_info    parameters for the simulation.
 */
GPUSimulator::GPUSimulator(Network *network, const SimulationInfo *sim_info) : Simulator(network, sim_info)
{
    /*
	This was copied over from GPUSim.cpp's constructor. Investigation needs to be done on
	whether or not this is the proper way to intialize the RNG with the new design. Also,
	I was unable to find a definition for initMTGPU anywhere in the old branch.
    */


    //initialize Mersenne Twister
    //assuming neuron_count >= 100 and is a multiple of 100. Note rng_mt_rng_count must be <= MT_RNG_COUNT
    int rng_blocks = 25; //# of blocks the kernel will use
    int rng_nPerRng = 4; //# of iterations per thread (thread granularity, # of rands generated per thread)
    int rng_mt_rng_count = sim_info->totalNeurons/rng_nPerRng; //# of threads to generate for neuron_count rand #s
    int rng_threads = rng_mt_rng_count/rng_blocks; //# threads per block needed
    initMTGPU(777, rng_blocks, rng_threads, rng_nPerRng, rng_mt_rng_count);


}
