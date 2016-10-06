/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
/*
 * Edited by Warner Smidt Sep 4th 2011
 * ds_MT now stores the seed state after each call to the random number generator. 
 * Each consecutive call to the random number generator will not produce the same 
 * results now. 
 * Note: iState has replaced seed in mt_struct_stripped, therefore the .dat files
 * last parameter which was for the seed is now used for the iState.
 * Also added RandomNormGPU which combines RandomGPU and BoxMuller for normalized
 * random numbers without extra global memory transfers. 
 *
 * Edit Sep 14th 2011
 * MT_RNG_COUNT is the max total threads that will be used. initMTGP is now used 
 * to setup RandomNormGPU/RandomGPU to be called from normalMTGPU/uniformMTGPU. 
 * Allows the random number generation to be more dynamic without relying as much 
 * on #defines as well as being able to make the calculations for the needed data  
 * at initialization only once, and not everytime the random numbers are needed. 
 */


#include <iostream>
#include <stdio.h>

using namespace std;
#include "MersenneTwister_d.h"

__device__ static mt_struct_stripped ds_MT[MT_RNG_COUNT];
static mt_struct_stripped h_MT[MT_RNG_COUNT];
__device__ unsigned int mt[MT_RNG_COUNT*MT_NN];


//#define MT_DATAFILE "MersenneTwister/data/MersenneTwister.dat"
/*
//globals
__device__ static mt_struct_stripped * ds_MT;
static mt_struct_stripped * h_MT;
__device__ unsigned int * mt;
*/

unsigned int mt_rng_count;
unsigned int mt_blocks;
unsigned int mt_threads;
unsigned int mt_nPerRng;

//Load twister configurations
void loadMTGPU(const char *fname){
	FILE *fd = fopen(fname, "rb");
	if(!fd){
		cerr << "initMTGPU(): failed to open " <<  fname << endl << "FAILED" << endl;
		exit(0);
	}
	if( !fread(h_MT, mt_rng_count*sizeof(mt_struct_stripped), 1, fd) ){
		cerr << "initMTGPU(): failed to load " <<  fname << endl << "FAILED" << endl;
		exit(0);
	}
	fclose(fd);
}

//initialize the seed to mt[]
__global__ void seedMTGPUState(unsigned int seed){
	const int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int iState;
	mt[MT_NN*tid] = seed;
	for (iState = MT_NN*tid+1; iState < MT_NN*(1+tid); iState++)
	mt[iState] = (1812433253U * (mt[iState - 1] ^ (mt[iState - 1] >> 30)) + iState) & MT_WMASK;

}

//Initialize/seed twister for current GPU context
void seedMTGPU(unsigned int seed){
	int i;
    //Need to be thread-safe
	mt_struct_stripped *MT = (mt_struct_stripped *)malloc(mt_rng_count * sizeof(mt_struct_stripped));

	for(i = 0; i < mt_rng_count; i++){
		MT[i]      = h_MT[i];
		MT[i].iState = i*MT_NN;
    }

   //seed does need to be used to initialize mt[] elements.
	int threadsPerBlock = 256;
	//get ceil of MT_RNG_COUNT/threadsPerBlock
	int blocksPerGrid = (mt_rng_count+threadsPerBlock-1)/threadsPerBlock; 
	seedMTGPUState<<<blocksPerGrid,threadsPerBlock>>>(seed);

	if(cudaMemcpyToSymbol(ds_MT, MT, mt_rng_count*sizeof(mt_struct_stripped))!=cudaSuccess){
		cerr << "seedMTGP failed" << endl;
		exit(0);
	}

	free(MT);
}


////////////////////////////////////////////////////////////////////////////////
// Write MT_RNG_COUNT vertical lanes of nPerRng random numbers to *d_Random.
// For coalesced global writes MT_RNG_COUNT should be a multiple of warp size.
// Initial states for each generator are the same, since the states are
// initialized from the global seed. In order to improve distribution properties
// on small NPerRng supply dedicated (local) seed to each twister.
// The local seeds, in their turn, can be extracted from global seed
// by means of any simple random number generator, like LCG.
////////////////////////////////////////////////////////////////////////////////
__global__ void RandomGPU(
	float *d_Random,
	int nPerRng, int mt_rng_count)
{
	const int      tid = blockDim.x * blockIdx.x + threadIdx.x;
	int iState, iState1, iStateM, iOut;
	unsigned int mti, mti1, mtiM, x;
	unsigned int matrix_a, mask_b, mask_c; 

    //Load bit-vector Mersenne Twister parameters
	matrix_a = ds_MT[tid].matrix_a;
	mask_b = ds_MT[tid].mask_b;
	mask_c = ds_MT[tid].mask_c;

	iState = ds_MT[tid].iState;
	mti1 = mt[iState];
	for (iOut = 0; iOut < nPerRng; iOut++) {
		iState1 = iState + 1;
		iStateM = iState + MT_MM;
        if(iState1 >= MT_NN*(1+tid)) iState1 -= MT_NN;
        if(iStateM >= MT_NN*(1+tid)) iStateM -= MT_NN;
		mti  = mti1;
		mti1 = mt[iState1];
		mtiM = mt[iStateM];

		// MT recurrence
		x    = (mti & MT_UMASK) | (mti1 & MT_LMASK);
		x    =  mtiM ^ (x >> 1) ^ ((x & 1) ? matrix_a : 0);

		mt[iState] = x;
		iState = iState1;

		//Tempering transformation
		x ^= (x >> MT_SHIFT0);
		x ^= (x << MT_SHIFTB) & mask_b;
		x ^= (x << MT_SHIFTC) & mask_c;
		x ^= (x >> MT_SHIFT1);

		//Convert to (0, 1] float and write to global memory
		d_Random[tid + iOut * mt_rng_count] = ((float)x + 1.0f) / 4294967296.0f;
	}
	ds_MT[tid].iState = iState;
}

////////////////////////////////////////////////////////////////////////////////
// Transform each of MT_RNG_COUNT lanes of nPerRng uniformly distributed 
// random samples, produced by RandomGPU(), to normally distributed lanes
// using Cartesian form of Box-Muller transformation.
// nPerRng must be even.
////////////////////////////////////////////////////////////////////////////////
#define PI 3.14159265358979f
__device__ inline void BoxMuller(float& u1, float& u2){
    float   r = sqrtf(-2.0f * logf(u1));
    float phi = 2 * PI * u2;
    u1 = r * __cosf(phi);
    u2 = r * __sinf(phi);
}

__global__ void BoxMullerGPU(float *d_Random, int nPerRng, int mt_rng_count){
    const int      tid = blockDim.x * blockIdx.x + threadIdx.x;

    for (int iOut = 0; iOut < nPerRng; iOut += 2)
        BoxMuller(
                d_Random[tid + (iOut + 0) * mt_rng_count],
                d_Random[tid + (iOut + 1) * mt_rng_count]
                );
}


//skip the seperate BoxMullerGPU for increased speed (uses register memory). 
//nPerRng must be a multiple of 2
__global__ void RandomNormGPU(
	float *d_Random,
	int nPerRng, int mt_rng_count)
{
	const int      tid = blockDim.x * blockIdx.x + threadIdx.x;
	int iState, iState1, iStateM, iOut;
	unsigned int mti, mti1, mtiM, x;
	unsigned int matrix_a, mask_b, mask_c; 

	float regVal1, regVal2;	//need 2 values for boxmuller
	bool boxFlag = false;	//will perform boxmuller transform on true	

    //Load bit-vector Mersenne Twister parameters
	matrix_a = ds_MT[tid].matrix_a;
	mask_b = ds_MT[tid].mask_b;
	mask_c = ds_MT[tid].mask_c;

	iState = ds_MT[tid].iState;
	mti1 = mt[iState];
	for (iOut = 0; iOut < nPerRng; iOut++) {
		iState1 = iState + 1;
		iStateM = iState + MT_MM;
        if(iState1 >= MT_NN*(1+tid)) iState1 -= MT_NN;
        if(iStateM >= MT_NN*(1+tid)) iStateM -= MT_NN;
		mti  = mti1;
		mti1 = mt[iState1];
		mtiM = mt[iStateM];

		// MT recurrence
		x    = (mti & MT_UMASK) | (mti1 & MT_LMASK);
		x    =  mtiM ^ (x >> 1) ^ ((x & 1) ? matrix_a : 0);

		mt[iState] = x;
		iState = iState1;

		//Tempering transformation
		x ^= (x >> MT_SHIFT0);
		x ^= (x << MT_SHIFTB) & mask_b;
		x ^= (x << MT_SHIFTC) & mask_c;
		x ^= (x >> MT_SHIFT1);

		if(boxFlag){
			regVal2 = ((float)x + 1.0f) / 4294967296.0f;
			BoxMuller(regVal1,regVal2);
			d_Random[tid + (iOut-1) * mt_rng_count] = regVal1;
			d_Random[tid + iOut * mt_rng_count] = regVal2;
			boxFlag = false;
		}else{
			regVal1 = ((float)x + 1.0f) / 4294967296.0f;
			boxFlag = true;
		}
	}
	ds_MT[tid].iState = iState;
}

extern "C" void uniformMTGPU(float * d_random){
	RandomGPU<<<mt_blocks,mt_threads>>>(d_random, mt_nPerRng, mt_rng_count);
}

extern "C" void normalMTGPU(float * d_random){
	RandomNormGPU<<<mt_blocks,mt_threads>>>(d_random, mt_nPerRng, mt_rng_count);	
}

//initialize globals and setup state
//Note: mt_rng_count must equal blocks*threads. mt_rng_count*nPerRng should equal the total number of randon numbers to be generated
extern "C" void initMTGPU(unsigned int seed, unsigned int blocks, unsigned int threads, unsigned int nPerRng, unsigned int mt_rng_c){
	mt_blocks = blocks;
	mt_threads = threads;
	mt_nPerRng = nPerRng;
	mt_rng_count = mt_rng_c;

	loadMTGPU(MT_DATAFILE);
	seedMTGPU(seed);
}

