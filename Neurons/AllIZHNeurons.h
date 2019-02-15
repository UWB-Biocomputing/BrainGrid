/**
 *      @file AllIZHNeurons.h
 *
 *      @brief A container of all Izhikevich neuron data
 */

/** 
 * 
 * @class AllIZHNeurons AllIZHNeurons.h "AllIZHNeurons.h"
 *
 * \latexonly  \subsubsection*{Implementation} \endlatexonly
 * \htmlonly   <h3>Implementation</h3> \endhtmlonly
 * A container of all spiking neuron data.
 * This is the base class of all spiking neuron classes.
 *
 * The class uses a data-centric structure, which utilizes a structure as the containers of
 * all neuron.
 *
 * The container holds neuron parameters of all neurons.
 * Each kind of neuron parameter is stored in a 1D array, of which length
 * is number of all neurons. Each array of a neuron parameter is pointed by a
 * corresponding member variable of the neuron parameter in the class.
 *
 * This structure was originally designed for the GPU implementation of the
 * simulator, and this refactored version of the simulator simply uses that design for
 * all other implementations as well. This is to simplify transitioning from
 * single-threaded to multi-threaded.
 *
 * The Izhikevich neuron model uses the quadratic integrate-and-fire model 
 * for ordinary differential equations of the form:
 * \f[
 *  \frac{d v}{dt} = 0.04v^2 + 5v + 140 - u + (I_{syn}(t) + I_{inject} + I_{noise})
 * \f]
 * \f[
 *  \frac{d u}{dt} = a \cdot (bv - u)
 * \f]
 * with the auxiliary after-spike resetting: if \f$v\ge30\f$ mv, then \f$v=c,u=u+d\f$.
 *
 * where \f$v\f$ and \f$u\f$ are dimensionless variable, and \f$a,b,c\f$, and \f$d\f$ are dimensioless parameters. 
 * The variable \f$v\f$ represents the membrane potential of the neuron and \f$u\f$ represents a membrane 
 * recovery variable, which accounts for the activation of \f$K^+\f$ ionic currents and 
 * inactivation of \f$Na^+\f$ ionic currents, and it provides negative feedback to \f$v\f$. 
 * \f$I_{syn}(t)\f$ is the current supplied by the synapses, \f$I_{inject}\f$ is a non-specific 
 * background current and Inoise is a Gaussian random variable with zero mean and 
 * a given variance noise (Izhikevich. 2003).
 *
 * The simple Euler method combined with the exponential Euler method is used for 
 * numerical integration. 
 * 
 * One step of the simple Euler method from \f$y(t)\f$ to \f$y(t + \Delta t)\f$ is:
 *  \f$y(t + \Delta t) = y(t) + \Delta t \cdot y(t)\f$
 *
 * The main idea behind the exponential Euler rule is 
 * that many biological processes are governed by an exponential decay function. 
 * For an equation of the form:
 * \f[
 *  \frac{d y}{dt} = A - By
 * \f]
 * its scheme is given by:
 * \f[
 *  y(t+\Delta t) = y(t) \cdot \mathrm{e}^{-B \Delta t} + \frac{A}{B} \cdot (1 - \mathrm{e}^{-B \Delta t}) 
 * \f]
 * After appropriate substituting all variables, we obtain the Euler step:
 * \f[
 *  v(t+\Delta t)=v(t)+ C3 \cdot (0.04v(t)^2+5v(t)+140-u(t))+C2 \cdot (I_{syn}(t)+I_{inject}+I_{noise}+\frac{V_{resting}}{R_{m}})
 * \f]
 * \f[
 *  u(t+ \Delta t)=u(t) + C3 \cdot a \cdot (bv(t)-u(t))
 * \f]
 * where \f$\tau_{m}=C_{m} \cdot R_{m}\f$ is the membrane time constant, \f$R_{m}\f$ is the membrane resistance,
 * \f$C2 = R_m \cdot (1 - \mathrm{e}^{-\frac{\Delta t}{\tau_m}})\f$,
 * and \f$C3 = \Delta t\f$.
 *
 * Because the time scale \f$t\f$ of the Izhikevich model is \f$ms\f$ scale, 
 *  so \f$C3\f$ is : \f$C3 = \Delta t = 1000 \cdot deltaT\f$ (\f$deltaT\f$ is the simulation time step in second) \f$ms\f$.
 *
 * \latexonly  \subsubsection*{Credits} \endlatexonly
 * \htmlonly   <h3>Credits</h3> \endhtmlonly
 *
 * This model is a rewrite of work by Stiber, Kawasaki, Allan Ortiz, and Cory Mayberry
 *
 */
#pragma once

#include "Global.h"
#include "AllIFNeurons.h"
#include "AllIZHNeuronsProps.h"

// Class to hold all data necessary for all the Neurons.
class AllIZHNeurons : public AllIFNeurons
{
    public:
        AllIZHNeurons();
        virtual ~AllIZHNeurons();

        static IAllNeurons* Create() { return new AllIZHNeurons(); }

        /**
         *  Create and setup neurons properties.
         */
        virtual void createNeuronsProps();

#if defined(USE_GPU)
    public:
        /**
         *  Update the state of all neurons for a time step
         *  Notify outgoing synapses if neuron has fired.
         *
         *  @param  synapses               Reference to the allSynapses struct on host memory.
         *  @param  allNeuronsDevice       Reference to the allNeurons struct on device memory.
         *  @param  allSynapsesDevice      Reference to the allSynapses struct on device memory.
         *  @param  sim_info               SimulationInfo to refer from.
         *  @param  randNoise              Reference to the random noise array.
         *  @param  synapseIndexMapDevice  Reference to the SynapseIndexMap on device memory.
         *  @param  clr_info               ClusterInfo class to read information from.
         *  @param  iStepOffset            Offset from the current simulation step.
         */
        virtual void advanceNeurons(IAllSynapses &synapses, void* allNeuronsDevice, void* allSynapsesDevice, const SimulationInfo *sim_info, float* randNoise, SynapseIndexMap* synapseIndexMapDevice, const ClusterInfo *clr_info, int iStepOffset);

#else // !defined(USE_GPU) 
    protected:
        /**
         *  Helper for #advanceNeuron. Updates state of a single neuron.
         *
         *  @param  index            Index of the neuron to update.
         *  @param  sim_info         SimulationInfo class to read information from.
         *  @param  clr_info         ClusterInfo class to read information from.
         *  @param  iStepOffset      Offset from the current simulation step.
         */
        virtual void advanceNeuron(const int index, const SimulationInfo *sim_info, const ClusterInfo *clr_info, int iStepOffset);

        /**
         *  Initiates a firing of a neuron to connected neurons.
         *
         *  @param  index            Index of the neuron to fire.
         *  @param  sim_info         SimulationInfo class to read information from.
         *  @param  iStepOffset      Offset from the current simulation step.
         */
        virtual void fire(const int index, const SimulationInfo *sim_info, int iStepOffset) const;
#endif  // defined(USE_GPU)
};
