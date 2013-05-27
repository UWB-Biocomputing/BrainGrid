/**
 * @brief A leaky-integrate-and-fire (I&F) neural network model for DirectCompute
 *
 * @class DC_LIFModel DC_LIFModel.h "AMP_LIFModel.h"
 *
 * Implements both neuron and synapse behaviour.
 *
 * This model is a rewrite of work by Stiber, Kawasaki, Allan Ortiz, and Cory Mayberry
 * See LIFModel.h, LIFModel.cpp for more information
 *
 * @authors Paul Bunn - AMP_LifModel derived from LIFModel and GPUSim (Auth: Fumitaka Kawasaki,)
 */

#ifndef _AMP_LIFMODEL_H_
#define _AMP_LIFMODEL_H_

#include <d3d11.h>
#include "LIFModel.h"
#include "Coordinate.h"
//#include "LifNeuron_struct.h"
//#include "LifSynapse_struct.h"
//#include "DelayIdx.h"

#include <vector>
#include <iostream>

#define SAFE_RELEASE(p)      { if(p) { (p)->Release(); (p)=NULL; } }

using namespace std;

/**
 * Implementation of Model for the Leaky-Integrate-and-Fire model.
 */
class AMP_LIFModel  : public LIFModel
{

    public:
        AMP_LIFModel();
        virtual ~AMP_LIFModel();

        /*
         * Declarations of concrete implementations of Model interface for an Leaky-Integrate-and-Fire
         * model.
         *
         * @see Model.h
         */

		// Only deviations from LIFModel are defined

		void advance(AllNeurons &neurons, AllSynapses &synapses, SimulationInfo *sim_info);

    protected:

        /* -----------------------------------------------------------------------------------------
         * # Helper Functions
         * ------------------
         */

        // # Read Parameters
        // -----------------

		// NOTE: ALL functions of LIFModel::TiXmlVisitor must be declared to avoid method hiding
        // Parse an element for parameter values.
        // Required by TiXmlVisitor, which is used by #readParameters
		bool VisitEnter(const TiXmlElement& element, const TiXmlAttribute* firstAttribute) { return LIFModel::VisitEnter(element, firstAttribute); };
		/// Visit a document.
		bool VisitEnter( const TiXmlDocument& doc )	{ return LIFModel::VisitEnter(doc); }
		/// Visit a document.
		bool VisitExit( const TiXmlDocument& doc )	{ return LIFModel::VisitExit(doc); }
		/// Visit an element.
		bool VisitExit( const TiXmlElement& element )			{ return LIFModel::VisitExit(element); }
		/// Visit a declaration
		bool Visit( const TiXmlDeclaration& declaration )		{ return LIFModel::Visit(declaration); }
		/// Visit a text node
		bool Visit( const TiXmlText& text )						{ return LIFModel::Visit(text); }
		/// Visit a comment node
		bool Visit( const TiXmlComment& comment )				{ return LIFModel::Visit(comment); }
		/// Visit an unknown node
		bool Visit( const TiXmlUnknown& unknown )				{ return LIFModel::Visit(unknown); }

		bool initializeModel(SimulationInfo *sim_info, AllNeurons& neurons, AllSynapses& synapses);
		void updateWeights(const uint32_t num_neurons, AllNeurons &neurons, AllSynapses &synapses, SimulationInfo *sim_info);
	private:
#ifdef STORE_SPIKEHISTORY
		//! pointer to an array to keep spike history for one activity epoch
		uint64_t* spikeArray;
#endif // STORE_SPIKEHISTORY
};

#ifdef _AMP_LIFModel
//
// Following declarations are only used when compiling AMP_LIFModel.cpp
//

#endif//_AMP_LIFModel


#endif//_AMP_LIFMODEL_H_
