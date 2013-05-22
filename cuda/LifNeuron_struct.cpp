/**
 ** \file LifNeuron_struct.cpp
 **
 ** \authors Allan Ortiz & Cory Mayberry
 **
 ** \brief A leaky-integrate-and-fire neuron
 **/

#include "LifNeuron_struct.h"

/**
 * Allocate data members in the LifNeuron_struct.
 * @param neuron
 * @param count
 */
void allocNeuronStruct(LifNeuron_struct& neuron, uint32_t count) {
	neuron.deltaT 			= new TIMEFLOAT[count]();
	neuron.summationPoint 	= new PBGFLOAT[count]();
	neuron.Cm 				= new BGFLOAT[count]();
	neuron.Rm 				= new BGFLOAT[count]();
	neuron.Vthresh 			= new BGFLOAT[count]();
	neuron.Vrest 			= new BGFLOAT[count]();
	neuron.Vreset 			= new BGFLOAT[count]();
	neuron.Vinit 			= new BGFLOAT[count]();
	neuron.Trefract 		= new BGFLOAT[count]();
	neuron.Inoise 			= new BGFLOAT[count]();
	neuron.randNoise 		= new float*[count]();
	neuron.Iinject 			= new BGFLOAT[count]();
	neuron.Isyn 			= new BGFLOAT[count]();
	neuron.nStepsInRefr 	= new uint32_t[count]();
	neuron.C1 				= new BGFLOAT[count]();
	neuron.C2 				= new BGFLOAT[count]();
	neuron.I0 				= new BGFLOAT[count]();
	neuron.Vm 				= new BGFLOAT[count]();
	neuron.hasFired 		= new bool[count]();
	neuron.Tau 				= new BGFLOAT[count]();
	neuron.spikeCount 		= new uint32_t[count]();
	neuron.outgoingSynapse_begin = new uint32_t[count]();
	neuron.synapseCount 	= new uint32_t[count]();
	neuron.incomingSynapse_begin = new uint32_t[count]();
	neuron.inverseCount 	= new uint32_t[count]();
	neuron.neuronCount 		= count;
}

/**
 * Deallocate data members in the LifNeuron_struct.
 * @param neuron
 */
void deleteNeuronStruct(LifNeuron_struct& neuron) {
	delete[] neuron.deltaT;
	delete[] neuron.summationPoint;
	delete[] neuron.Cm;
	delete[] neuron.Rm;
	delete[] neuron.Vthresh;
	delete[] neuron.Vrest;
	delete[] neuron.Vreset;
	delete[] neuron.Vinit;
	delete[] neuron.Trefract;
	delete[] neuron.Inoise;
	delete[] neuron.randNoise;
	delete[] neuron.Iinject;
	delete[] neuron.Isyn;
	delete[] neuron.nStepsInRefr;
	delete[] neuron.C1;
	delete[] neuron.C2;
	delete[] neuron.I0;
	delete[] neuron.Vm;
	delete[] neuron.hasFired;
	delete[] neuron.Tau;
	delete[] neuron.spikeCount;
	delete[] neuron.outgoingSynapse_begin;
	delete[] neuron.synapseCount;
	delete[] neuron.incomingSynapse_begin;
	delete[] neuron.inverseCount;
	memset(&neuron, 0, sizeof(LifNeuron_struct));
}

/**
 * Copy INeuron data into a LifNeuron_struct for GPU processing.
 * @param in
 * @param out
 * @param idx
 */
void copyNeuronToStruct(AllNeurons &neurons, LifNeuron_struct& out, int idx) {
	out.deltaT[idx] 	= neurons.deltaT[idx];
	out.C1[idx] 		= neurons.C1[idx];
	out.C2[idx] 		= neurons.C2[idx];
	out.Cm[idx] 		= neurons.Cm[idx];
	out.I0[idx] 		= neurons.I0[idx];
	out.Iinject[idx] 	= neurons.Iinject[idx];
	out.Inoise[idx] 	= neurons.Inoise[idx];
	out.Isyn[idx] 		= neurons.Isyn[idx];
	out.Rm[idx] 		= neurons.Rm[idx];
	out.Tau[idx] 		= neurons.Tau[idx];
	out.Trefract[idx] 	= neurons.Trefract[idx];
	out.Vinit[idx] 	= neurons.Vinit[idx];
	out.Vm[idx] 		= neurons.Vm[idx];
	out.Vrest[idx] 	= neurons.Vrest[idx];
	out.Vreset[idx] 	= neurons.Vreset[idx];
	out.Vthresh[idx] 	= neurons.Vthresh[idx];
	out.nStepsInRefr[idx] 	= neurons.nStepsInRefr[idx];
	out.spikeCount[idx] 	= 0;
	out.summationPoint[idx] = 0;
	out.hasFired[idx] 	= false;
}

/**
 * Copy LifNeuron_struct array data into a INeuron.
 * @param in
 * @param out
 * @param idx
 */
void copyNeuronStructToNeuron(LifNeuron_struct& in, AllNeurons &neurons, int idx) {
	neurons.C1[idx] 			= in.C1[idx];
	neurons.C2[idx] 			= in.C2[idx];
	neurons.Cm[idx] 			= in.Cm[idx];
	neurons.I0[idx] 			= in.I0[idx];
	neurons.Iinject[idx] 		= in.Iinject[idx];
	neurons.Inoise[idx] 		= in.Inoise[idx];
	neurons.Isyn[idx] 			= in.Isyn[idx];
	neurons.Rm[idx] 			= in.Rm[idx];
	neurons.Tau[idx] 			= in.Tau[idx];
	neurons.Trefract[idx] 		= in.Trefract[idx];
	neurons.Vinit[idx] 			= in.Vinit[idx];
	neurons.Vm[idx] 			= in.Vm[idx];
	neurons.Vrest[idx] 			= in.Vrest[idx];
	neurons.Vreset[idx] 		= in.Vreset[idx];
	neurons.Vthresh[idx] 		= in.Vthresh[idx];
	neurons.nStepsInRefr[idx]	= in.nStepsInRefr[idx];
	neurons.spikeCount[idx] 	= in.spikeCount[idx];
}

/**
 * Copy a neuronArray into a neuronMap
 * @param neuron
 * @param pNeuronList
 * @param numNeurons
 */
void neuronArrayToMap(LifNeuron_struct& neuron_st, AllNeurons &neurons, int numNeurons)
{
	for (int i = 0; i < numNeurons; i++)	
		copyNeuronStructToNeuron(neuron_st, neurons, i);	
}
