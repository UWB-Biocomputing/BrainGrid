/**
 **
 ** \brief A leaky-integrate-and-fire (I&F) neuron.
 **
 ** \class LifNeuron LifNeuron.h "LifNeuron.h"
 **
 ** \latexonly  \subsubsection*{Model} \endlatexonly
 ** \htmlonly  <h3>Model</h3> \endhtmlonly
 **
 ** A standard leaky-integrate-and-fire neuron model is implemented
 ** where the membrane potential \f$V_m\f$ of a neuron is given by
 ** \f[
 **   \tau_m \frac{d V_m}{dt} = -(V_m-V_{resting}) + R_m \cdot (I_{syn}(t)+I_{inject}+I_{noise})
 ** \f]
 ** where \f$\tau_m=C_m\cdot R_m\f$ is the membrane time constant,
 ** \f$R_m\f$ is the membrane resistance, \f$I_{syn}(t)\f$ is the
 ** current supplied by the synapses, \f$I_{inject}\f$ is a
 ** non-specific background current and \f$I_{noise}\f$ is a
 ** Gaussian random variable with zero mean and a given variance
 ** noise.
 **
 ** At time \f$t=0\f$ \f$V_m\f$ is set to \f$V_{init}\f$. If
 ** \f$V_m\f$ exceeds the threshold voltage \f$V_{thresh}\f$ it is
 ** reset to \f$V_{reset}\f$ and hold there for the length
 ** \f$T_{refract}\f$ of the absolute refractory period.
 **
 ** \latexonly  \subsubsection*{Implementation} \endlatexonly
 **
 ** \htmlonly  <h3>Implementation</h3> \endhtmlonly
 ** The exponential Euler method is used for numerical integration.
 **
 **	@authors Allan Ortiz & Cory Mayberry
 **/

/**
 * \file LifNeuron.h
 *
 * Header file for LifNeuron
 */

#pragma once

#ifndef _LIFNEURON_H_
#define _LIFNEURON_H_

#include "INeuron.h"

//! Implementation of INeuron: A leaky-integrate-and-fire (I&F) neuron
class LifNeuron : public INeuron {
public:

	//! Constructor.
	LifNeuron();
	virtual ~LifNeuron();

	//! Process another time step.
	virtual void advance(BGFLOAT& summationPoint);

	//! Emit a spike.
	virtual void fire();

	//! Update internal constants.
	virtual void updateInternal();
};

#endif
