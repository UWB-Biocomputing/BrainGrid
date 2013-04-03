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
void allocNeuronStruct(LifNeuron_struct* neuron, int count) {
	neuron->deltaT 				= new BGFLOAT[count];
	neuron->summationPoint 			= new PBGFLOAT[count];
	neuron->Cm 				= new BGFLOAT[count];
	neuron->Rm 				= new BGFLOAT[count];
	neuron->Vthresh 				= new BGFLOAT[count];
	neuron->Vrest 				= new BGFLOAT[count];
	neuron->Vreset 				= new BGFLOAT[count];
	neuron->Vinit 				= new BGFLOAT[count];
	neuron->Trefract 			= new BGFLOAT[count];
	neuron->Inoise 				= new BGFLOAT[count];
	neuron->randNoise 			= new float*[count];
	neuron->Iinject 				= new BGFLOAT[count];
	neuron->Isyn 				= new BGFLOAT[count];
	neuron->nStepsInRefr 			= new int[count];
	neuron->C1 				= new BGFLOAT[count];
	neuron->C2 				= new BGFLOAT[count];
	neuron->I0 				= new BGFLOAT[count];
	neuron->Vm 				= new BGFLOAT[count];
	neuron->hasFired 			= new bool[count];
	neuron->Tau 				= new BGFLOAT[count];
	neuron->spikeCount 			= new int[count];
	neuron->outgoingSynapse_begin 		= new int[count];
	neuron->synapseCount 			= new int[count];
	neuron->incomingSynapse_begin 		= new int[count];
	neuron->inverseCount 			= new int[count];
	neuron->numNeurons 			= new int[count];
	neuron->stepDuration 			= new int[count];   // NOTE: unused. TODO: delete??
}

/**
 * Deallocate data members in the LifNeuron_struct.
 * @param neuron
 */
void deleteNeuronStruct(LifNeuron_struct* neuron) {
	delete[] neuron->deltaT;
	delete[] neuron->summationPoint;
	delete[] neuron->Cm;
	delete[] neuron->Rm;
	delete[] neuron->Vthresh;
	delete[] neuron->Vrest;
	delete[] neuron->Vreset;
	delete[] neuron->Vinit;
	delete[] neuron->Trefract;
	delete[] neuron->Inoise;
	delete[] neuron->randNoise;
	delete[] neuron->Iinject;
	delete[] neuron->Isyn;
	delete[] neuron->nStepsInRefr;
	delete[] neuron->C1;
	delete[] neuron->C2;
	delete[] neuron->I0;
	delete[] neuron->Vm;
	delete[] neuron->hasFired;
	delete[] neuron->Tau;
	delete[] neuron->spikeCount;
	delete[] neuron->outgoingSynapse_begin;
	delete[] neuron->synapseCount;
	delete[] neuron->incomingSynapse_begin;
	delete[] neuron->inverseCount;
	delete[] neuron->numNeurons;
	delete[] neuron->stepDuration;
}

/**
 * Copy INeuron data into a LifNeuron_struct for GPU processing.
 * @param in
 * @param out
 * @param idx
 */
void copyNeuronToStruct(INeuron* in, LifNeuron_struct* out, int idx) {
	out->deltaT[idx] 	= in->deltaT;
	out->C1[idx] 		= in->C1;
	out->C2[idx] 		= in->C2;
	out->Cm[idx] 		= in->Cm;
	out->I0[idx] 		= in->I0;
	out->Iinject[idx] 	= in->Iinject;
	out->Inoise[idx] 	= in->Inoise;
	out->Isyn[idx] 		= in->Isyn;
	out->Rm[idx] 		= in->Rm;
	out->Tau[idx] 		= in->Tau;
	out->Trefract[idx] 	= in->Trefract;
	out->Vinit[idx] 		= in->Vinit;
	out->Vm[idx] 		= in->Vm;
	out->Vrest[idx] 		= in->Vrest;
	out->Vreset[idx] 	= in->Vreset;
	out->Vthresh[idx] 	= in->Vthresh;
	out->nStepsInRefr[idx] 	= in->nStepsInRefr;
	out->spikeCount[idx] 	= 0;
	out->summationPoint[idx] = 0;
	out->hasFired[idx] 	= false;
}

/**
 * Copy LifNeuron_struct array data into a INeuron.
 * @param in
 * @param out
 * @param idx
 */
void copyNeuronStructToNeuron(LifNeuron_struct* in, INeuron* out, int idx) {
	out->C1 			= in->C1[idx];
	out->C2 			= in->C2[idx];
	out->Cm 			= in->Cm[idx];
	out->I0 			= in->I0[idx];
	out->Iinject 		= in->Iinject[idx];
	out->Inoise 		= in->Inoise[idx];
	out->Isyn 		= in->Isyn[idx];
	out->Rm 			= in->Rm[idx];
	out->Tau 		= in->Tau[idx];
	out->Trefract 		= in->Trefract[idx];
	out->Vinit 		= in->Vinit[idx];
	out->Vm 			= in->Vm[idx];
	out->Vrest 		= in->Vrest[idx];
	out->Vreset 		= in->Vreset[idx];
	out->Vthresh 		= in->Vthresh[idx];
	out->nStepsInRefr 	= in->nStepsInRefr[idx];
	out->spikeCount 		= in->spikeCount[idx];
}

/**
 * Copy a neuronArray into a neuronMap
 * @param neuron
 * @param pNeuronList
 * @param numNeurons
 */
void neuronArrayToMap(LifNeuron_struct* neuron, vector<INeuron*>* pNeuronList, int numNeurons)
{
	for (int i = 0; i < numNeurons; i++)	
		copyNeuronStructToNeuron(neuron, (*pNeuronList)[i], i);	
}
