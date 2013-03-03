/**
 *	@file Network.h
 *
 *
 *	@brief Header file for Network.
 */
//! A collection of Neurons and their connecting synapses.

/**
 ** \class Network Network.h "Network.h"
 **
 **\latexonly  \subsubsection*{Implementation} \endlatexonly
 ** \htmlonly  <h3>Implementation</h3> \endhtmlonly
 **
 ** \image html bg_data_layout.png
 **
 ** The network is composed of 3 superimposed 2-d arrays: neurons, synapses, and
 ** summation points (m_neuronList, m_rgSynapseMap, and m_summationMap).
 **
 ** Synapses in the synapse map are located at the coordinates of the neuron
 ** from which they receive output.  Each synapse stores a pointer into a
 ** m_summationMap bin.  Bins in the m_summationMap map directly to their output neurons.
 ** 
 ** If, during an advance cycle, a neuron \f$A\f$ at coordinates \f$x,y\f$ fires, every synapse
 ** \f$B\f$ at \f$x,y\f$ in the m_rgSynapseMap is notified of the spike. Those synapses then hold
 ** the spike until their delay period is completed.  At a later advance cycle, once the delay
 ** period has been completed, the synapses apply their PSRs (Post-Synaptic-Response) to their output bins \f$C\f$ in
 ** the m_summationMap.  Finally, on the next advance cycle, each neuron \f$D\f$ adds the value stored
 ** in their corresponding m_summationMap bin to their \f$V_m\f$ and resets the m_summationMap bin to
 ** zero.
 **
 ** \latexonly \subsubsection*{Credits} \endlatexonly
 ** \htmlonly <h3>Credits</h3> \endhtmlonly
 **
 ** This simulator is a rewrite of CSIM (2006) and other work (Stiber and Kawasaki (2007?))
 **
 **
 **	@authors Allan Ortiz and Cory Mayberry
 **/

#ifndef _NETWORK_H_
#define _NETWORK_H_

#include "global.h"
#include "include/Timer.h"
#include "ISimulation.h"

#include "Model.h"

class Network
{
public:
	//! The constructor for Network.
	Network(Model *model,
            FLOAT startFrac,
            FLOAT new_targetRate,
            ostream& new_outstate,istream& new_meminput, bool fReadMemImage,
            SimulationInfo simInfo);
	~Network();

	//! Frees dynamically allocated memory associated with the maps.
	void freeResources();

	//! Reset simulation objects
	void reset();

	//! Randomly initialize entries in the neuron starter map.
	void initStarterMap();

    //! Write the network state to an ostream.
    void saveSimState(ostream& os, FLOAT growthStepDuration, FLOAT maxGrowthSteps);
    
    //! Write the simulation memory image to an ostream
    void writeSimMemory(FLOAT simulation_step, ostream& os);

	//! Read the simulation memory image from an istream
	void readSimMemory(istream& is, VectorMatrix& radii, VectorMatrix& rates);

// -----------------------------------------------------------------------------

    // Setup simulation
    void setup(FLOAT growthStepDuration, FLOAT num_growth_steps);
    
    // Cleanup after simulation
    void finish(FLOAT growthStepDuration, FLOAT num_growth_steps);
    
    // TODO comment
    void get_spike_history(VectorMatrix& burstinessHist, VectorMatrix& spikesHistory);
    
    // TODO comment
    void advance();
    
    // TODO comment
    void updateConnections(const int currentStep);

    //! Get spike counts in prep for growth
    void getSpikeCounts(int neuron_count, int* spikeCounts);  // PLATFORM DEPENDENT
    
    //! Clear spike count of each neuron.
    void clearSpikeCounts(int neuron_count);  // PLATFORM DEPENDENT
    
    //! Print network radii to console.
    void printRadii(SimulationInfo* psi) const;

// -----------------------------------------------------------------------------

	//! Output the m_rgNeuronTypeMap to a VectorMatrix.
	void getNeuronTypes(VectorMatrix& neuronTypes);

	//! Output the m_pfStarterMap to a VectorMatrix.
	void getStarterNeuronMatrix(VectorMatrix& starterNeurons);

// -----------------------------------------------------------------------------

	Model *m_model;
	AllNeurons neurons;
	AllSynapses synapses;

// -----------------------------------------------------------------------------

	//! The number of endogenously active neurons.
	int m_cStarterNeurons;

	//! List of lists of synapses
	vector<ISynapse*>* m_rgSynapseMap;

	//! The map of summation points.
	FLOAT* m_summationMap;

	//! The neuron map.
	vector<INeuron*> m_neuronList;

	//! The neuron type map (INH, EXC).
	neuronType* m_rgNeuronTypeMap;

	//! The starter existence map (T/F).
	bool* m_rgEndogenouslyActiveNeuronMap;

	//! growth param (spikes/second) TODO: more detail here
	FLOAT m_targetRate;

	//! A file stream for xml output.
	ostream& state_out;

	//! An input file stream for memory image 	
	istream& memory_in;

	//! True if dumped memory image is read before starting simulation. 
	bool m_fReadMemImage;

// -----------------------------------------------------------------------------

    VectorMatrix radii; // previous saved radii
    VectorMatrix rates; // previous saved rates

    // track radii
    CompleteMatrix radiiHistory; // state

    // track firing rate
    CompleteMatrix ratesHistory;
    
    // neuron locations matrices
    VectorMatrix xloc;
    VectorMatrix yloc;
    
    static const string MATRIX_TYPE;
    static const string MATRIX_INIT;

private:
	// Struct that holds information about a simulation
	SimulationInfo m_si;

	Network();
};
#endif // _NETWORK_H_
