/**
 *      @file SimulationInfo.h
 *
 *      @brief Header file for SimulationInfo.
 */
//! Simulation information.

/**
 ** \class SimulationInfo SimulationInfo.h "SimulationInfo.h"
 **
 ** \latexonly  \subsubsection*{Implementation} \endlatexonly
 ** \htmlonly   <h3>Implementation</h3> \endhtmlonly
 **
 ** The SimulationInfo contains all information necessary for the simulation.
 **
 ** \latexonly  \subsubsection*{Credits} \endlatexonly
 ** \htmlonly   <h3>Credits</h3> \endhtmlonly
 **
 ** Some models in this simulator is a rewrite of CSIM (2006) and other 
 ** work (Stiber and Kawasaki (2007?))
 **
 **
 **     @author Allan Ortiz & Cory Mayberry
 **/

#pragma once

#ifndef _SIMULATIONINFO_H_
#define _SIMULATIONINFO_H_

#include "Global.h"
#include "ParameterManager.h"

class IModel;
class IRecorder;
class ISInput;
#ifdef PERFORMANCE_METRICS
// Home-brewed performance measurement
#include "Timer.h"
#endif

//! Class design to hold all of the parameters of the simulation.
class SimulationInfo 
{
public:
        SimulationInfo() :
            width(0),
            height(0),
            totalNeurons(0),
            currentStep(0),
            maxSteps(0),
            epochDuration(0),
            maxFiringRate(0),
            maxSynapsesPerNeuron(0),
            minSynapticTransDelay(MIN_SYNAPTIC_TRANS_DELAY), 
            deltaT(DEFAULT_dt),
            maxRate(0),
            seed(0),
            numClusters(0),
            model(NULL),
            simRecorder(NULL),
            pInput(NULL)
        {
        }

        virtual ~SimulationInfo() {}

        /**
         *  Attempts to parse parameters from a XML file
         *  using provided ParameterManager functionality.
         *
         *  @param  paramMgr    The instance of ParameterManager.
         *  @return true if successful, false otherwise.
         */
        bool readParameters(ParameterManager* paramMgr);

        /**
         *  Prints out loaded parameters to ostream.
         *
         *  @param  output  ostream to send output to.
         */
        void printParameters(ostream &output) const;

    public:

	//! Width of neuron map (assumes square)
	int width;

	//! Height of neuron map
	int height;

	//! Count of neurons in the simulation
	int totalNeurons;

	//! Current simulation step
	int currentStep;

	//! Maximum number of simulation steps
	int maxSteps; // TODO: delete

	//! The length of each step in simulation time
	BGFLOAT epochDuration; // Epoch duration !!!!!!!!

	//! Maximum firing rate. **Only used by GPU simulation.**
	int maxFiringRate;

	//! Maximum number of synapses per neuron. **Only used by GPU simulation.**
	int maxSynapsesPerNeuron;

        //! The synaptic transmission delay (minimum), descretized into time steps
        int minSynapticTransDelay;

	//! Time elapsed between the beginning and end of the simulation step
	BGFLOAT deltaT; // Inner Simulation Step Duration !!!!!!!!

	//! growth variable (m_targetRate / m_epsilon) TODO: more detail here
	BGFLOAT maxRate;

	//! Seed used for the simulation random SINGLE THREADED
	long seed;

        //! Number of clusters.
        int numClusters;

        //! File name of the simulation results.
        string stateOutputFileName;

        //! File name of the parameter description file.
        string stateInputFileName;

        //! File name of the memory dump output file.
        string memOutputFileName;

        //! File name of the memory dump input file.
        string memInputFileName;

        //! File name of the stimulus input file.
        string stimulusInputFileName;

        //! Neural Network Model interface.
        IModel *model;

        //! Recorder object.
        IRecorder* simRecorder;

        //! Stimulus input object.
        ISInput* pInput;
    
#ifdef PERFORMANCE_METRICS
        /**
         * Timer for measuring performance of an epoch.
         */
        Timer timer;
        /**
         * Timer for measuring performance of connection update.
         */
        Timer short_timer;
#endif

};

#endif // _SIMULATIONINFO_H_
