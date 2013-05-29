#define _MERSENNE_TWISTER_KERNEL
//
// Imported by Paul Bunn for use by BrainGrid project
// Notes: This file uses C++ AMP on VisualStudio 2012
// to implement the Mersenne Twister.
// Some code taken from MersenneTwisterCUDA.cu
//
//////////////////////////////////////////////////////////////////////////////
//// Copyright (c) Microsoft Corporation. All rights reserved
//// This software contains source code provided by NVIDIA Corporation.
//////////////////////////////////////////////////////////////////////////////

//----------------------------------------------------------------------------
// File: MersenneTwister.cpp
// 
// This sample implements Mersenne Twister random number generator 
// and Cartesian Box-Muller transformation on the GPU.
//----------------------------------------------------------------------------

#include <math.h>
#include <fstream>
#include <iostream>
#include <amp.h>
#include <amp_math.h>
#include <assert.h>
#include "../cuda/MersenneTwisterGPU.h"

// Disable "conversion from 'size_t' to 'int', possible loss of data" errors:
#pragma warning (disable : 4267)

using namespace concurrency;

// Variables used throughout this file for parameters
// initialized by initMTGPU_AMP
unsigned int mt_rng_count;
unsigned int mt_blocks;
unsigned int mt_threads;
unsigned int mt_nPerRng;
std::vector<unsigned int> v_matrix, v_mask_b, v_mask_c, v_seed;

//----------------------------------------------------------------------------
// Common host and device function 
//----------------------------------------------------------------------------
inline int div_up(int a, int b)
{
    return (a + b - 1)/b;
}

//Align a to nearest higher multiple of b
inline int align_up(int a, int b)
{
    return ((a + b - 1)/b) * b;
}

const unsigned int g_seed = 777;
static mt_struct h_MT[MT_RNG_COUNT];

//Load twister configurations
void loadMTGPU_AMP(const char *fname)
{
    FILE *fd = NULL;
    if(0 != fopen_s(&fd, fname, "rb") || !fread(h_MT, mt_rng_count*sizeof(mt_struct), 1, fd))
    {
        printf("loadMTGPU_AMP: failed to process %s\n", fname);
		// use default table already defined in MersenneTwisterGPU.h
		assert(mt_rng_count*sizeof(mt_struct) <= MersenneTwister_dat_len);
		memcpy(h_MT, MersenneTwister_dat, mt_rng_count*sizeof(mt_struct));
		return;
    }
	if(fd != NULL) {
		fclose(fd);
	}
}


#if 0
// this does a built-in box-muller transformation

void generate_rand_on_amp(std::vector<float>& v_random_nums)
{
    extent<1> e_c(v_matrix.size());
    int n_per_RNG = mt_nPerRng;
    extent<2> rn(mt_nPerRng, mt_rng_count);

    array<float, 2> random_nums(rn); 
    array<float, 2> normalized_random_nums(rn);

    // Copy to GPU
    array<unsigned int, 1> matrix_a(e_c, v_matrix.begin());
    array<unsigned int, 1> seed(e_c, v_seed.begin());
    array<unsigned int, 1> mask_b(e_c, v_mask_b.begin());
    array<unsigned int, 1> mask_c(e_c, v_mask_c.begin());

	assert((n_per_RNG & 1) == 0); // ensure it's even -- odd not allowed
    // generate random numbers
    parallel_for_each(e_c, [=, &random_nums, &matrix_a, &mask_b, &mask_c, &seed] (index<1> idx) restrict(amp)
    {
        rand_MT_kernel(idx, random_nums, matrix_a[idx], mask_b[idx], mask_c[idx], seed[idx], n_per_RNG);
    });

	// Because box-muller is not called, copy the un-normalized random nums
	copy(random_nums, v_random_nums.begin()); 
}
#endif

//Initialize/seed twister for current GPU context
void seed_MT(unsigned int seed0, std::vector<unsigned int>& matrix, 
               std::vector<unsigned int>& mask_b, std::vector<unsigned int>& mask_c, 
               std::vector<unsigned int>& seed) 
{
	for(uint32_t i = 0; i < mt_rng_count; i++) {
        matrix[i] = h_MT[i].matrix_a;
        mask_b[i] = h_MT[i].mask_b;
        mask_c[i] = h_MT[i].mask_c;
        seed[i] = seed0;
    }
}

//initialize globals and setup state
//Note: mt_rng_count must equal blocks*threads. mt_rng_count*nPerRng should equal the total number of random numbers to be generated
void initMTGPU_AMP(unsigned int seed, unsigned int blocks, unsigned int threads, unsigned int nPerRng, unsigned int mt_rng_c) {
	mt_blocks = blocks;
	mt_threads = threads;
	mt_nPerRng = nPerRng;
	mt_rng_count = mt_rng_c;

	accelerator default_device;
	std::wcout << L"AMP Using device : " << default_device.get_description() << std::endl;
	if (default_device == accelerator(accelerator::direct3d_ref))
		std::cout << "WARNING!! Running on very slow emulator! Only use this accelerator for debugging." << std::endl;

	loadMTGPU_AMP(MT_DATAFILE);
	v_matrix.resize(mt_rng_count);
	v_mask_b.resize(mt_rng_count);
	v_mask_c.resize(mt_rng_count);
	v_seed.resize(mt_rng_count);
    reseed_MTGPU_AMP(g_seed);
}

void reseed_MTGPU_AMP(uint32_t newseed) {
    seed_MT(newseed, v_matrix, v_mask_b, v_mask_c, v_seed);
}
