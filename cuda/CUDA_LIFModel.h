/**
 * @brief A leaky-integrate-and-fire (I&F) neural network model for GPU CUDA
 *
 * @class LIFModel LIFModel.h "LIFModel.h"
 *
 * Implements both neuron and synapse behaviour.
 *
 * A standard leaky-integrate-and-fire neuron model is implemented
 * where the membrane potential \f$V_m\f$ of a neuron is given by
 * \f[
 *   \tau_m \frac{d V_m}{dt} = -(V_m-V_{resting}) + R_m \cdot (I_{syn}(t)+I_{inject}+I_{noise})
 * \f]
 * where \f$\tau_m=C_m\cdot R_m\f$ is the membrane time constant,
 * \f$R_m\f$ is the membrane resistance, \f$I_{syn}(t)\f$ is the
 * current supplied by the synapses, \f$I_{inject}\f$ is a
 * non-specific background current and \f$I_{noise}\f$ is a
 * Gaussian random variable with zero mean and a given variance
 * noise.
 *
 * At time \f$t=0\f$ \f$V_m\f$ is set to \f$V_{init}\f$. If
 * \f$V_m\f$ exceeds the threshold voltage \f$V_{thresh}\f$ it is
 * reset to \f$V_{reset}\f$ and hold there for the length
 * \f$T_{refract}\f$ of the absolute refractory period.
 *
 * The exponential Euler method is used for numerical integration.
 *
 * This model is a rewrite of work by Stiber, Kawasaki, Allan Ortiz, and Cory Mayberry
 *
 * @authors Derek McLean
 *          Paul Bunn - GPULifModel derived from LIFModel and GPUSim (Auth: Fumitaka Kawasaki,)
 */
#pragma once
#ifndef _CUDA_LIFMODEL_H_
#define _CUDA_LIFMODEL_H_

#include "LIFModel.h"
#include "Coordinate.h"

#include <vector>
#include <iostream>

using namespace std;

/**
 * Implementation of Model for the Leaky-Integrate-and-Fire model.
 */
class CUDA_LIFModel  : public LIFModel
{

    public:
        CUDA_LIFModel();
        virtual ~CUDA_LIFModel();

        /*
         * Declarations of concrete implementations of Model interface for an Leaky-Integrate-and-Fire
         * model.
         *
         * @see Model.h
         */

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

};

#endif
