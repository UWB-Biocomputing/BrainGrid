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

class IModel;
class IRecorder;
class ISInput;

//! Class design to hold all of the parameters of the simulation.
class SimulationInfo : public TiXmlVisitor
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
            deltaT(DEFAULT_dt),
            maxRate(0),
            pSummationMap(NULL),
	    seed(0),
            model(NULL),
            simRecorder(NULL),
            pInput(NULL)
        {
        }

        virtual ~SimulationInfo() {}

        /**
         *  Attempts to read parameters from a XML file.
         *
         *  @param  simDoc  the TiXmlDocument to read from.
         *  @return true if successful, false otherwise.
         */
        bool readParameters(TiXmlDocument* simDoc);

        /**
         *  Prints out loaded parameters to ostream.
         *
         *  @param  output  ostream to send output to.
         */
        void printParameters(ostream &output) const;

    protected:
        using TiXmlVisitor::VisitEnter;

        /*
         *  Handles loading of parameters using tinyxml from the parameter file.
         *
         *  @param  element TiXmlElement to examine.
         *  @param  firstAttribute  ***NOT USED***.
         *  @return true if method finishes without errors.
         */
        virtual bool VisitEnter(const TiXmlElement& element, const TiXmlAttribute* firstAttribute);

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

	//! Time elapsed between the beginning and end of the simulation step
	BGFLOAT deltaT; // Inner Simulation Step Duration !!!!!!!!

	//! The neuron type map (INH, EXC).
	neuronType* rgNeuronTypeMap;

	//! The starter existence map (T/F).
	bool* rgEndogenouslyActiveNeuronMap;

	//! growth variable (m_targetRate / m_epsilon) TODO: more detail here
	BGFLOAT maxRate;

	//! List of summation points (either host or device memory)
	BGFLOAT* pSummationMap;

	//! Seed used for the simulation random SINGLE THREADED
	long seed;

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
    
    private:
        /**
         *  Checks the number of required parameters to read.
         *
         *  @return true if all required parameters were successfully read, false otherwise.
         */
        virtual bool checkNumParameters();

        //! Number of parameters read.
        int nParams;
};

#endif // _SIMULATIONINFO_H_
