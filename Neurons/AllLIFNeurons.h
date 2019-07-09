/**
 *      @file AllLIFNeurons.h
 *
 *      @brief A container of all LIF neuron data
 */

/** 
 * @authors Aaron Oziel, Sean Blackbourn
 * 
 * @class AllIFNeurons AllIFNeurons.h "AllIFNeurons.h"
 *
 * \latexonly  \subsubsection*{Implementation} \endlatexonly
 * \htmlonly   <h3>Implementation</h3> \endhtmlonly
 *
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
 * The main idea behind the exponential Euler rule is that many biological 
 *mprocesses are governed by an exponential decay function.
 * For an equation of the form:
 * \f[
 *  \frac{d y}{dt} = A - By
 * \f]
 *
 * its scheme is given by:
 * \f[
 *  y(t+\Delta t) = y(t) \cdot \mathrm{e}^{-B \Delta t} + \frac{A}{B} \cdot (1 - \mathrm{e}^{-B \Delta t})
 * \f]
 *
 * After appropriate substituting all variables:
 * \f[
 *  y(t) = V_m(t)
 * \f]
 * \f[
 *  A = \frac{1}{\tau_m} \cdot (R_m \cdot (I_{syn}(t)+I_{inject}+I_{noise}) + V_{resting})
 * \f]
 * and
 * \f[
 *  B = \frac{1}{\tau_m}
 * \f]
 * 
 * we obtain the exponential Euler step:
 * \f[
 *  V_m(t+\Delta t) = C1 \cdot V_m(t) + 
 *  C2 \cdot (I_{syn}(t)+I_{inject}+I_{noise}+\frac{V_{resting}}{R_m}) 
 * \f]
 * where \f$C1 = \mathrm{e}^{-\frac{\Delta t}{\tau_m}}\f$ and 
 * \f$C2 = R_m \cdot (1 - \mathrm{e}^{-\frac{\Delta t}{\tau_m}})\f$.
 *
 * \latexonly  \subsubsection*{Credits} \endlatexonly
 * \htmlonly   <h3>Credits</h3> \endhtmlonly
 *
 * This model is a rewrite of work by Stiber, Kawasaki, Allan Ortiz, and Cory Mayberry
 *
 * Some models in this simulator is a rewrite of CSIM (2006) and other
 * work (Stiber and Kawasaki (2007?))
 */
#pragma once

#include "Global.h"
#include "AllIFNeurons.h"
#include "AllSpikingSynapses.h"

// Class to hold all data necessary for all the Neurons.
class AllLIFNeurons : public AllIFNeurons
{
    public:

        CUDA_CALLABLE AllLIFNeurons();
        CUDA_CALLABLE virtual ~AllLIFNeurons();

        /**
         *  Creates an instance of the class.
         *  The function is called from FClassOfCategory.
         *
         *  @return Reference to the instance of the class.
         */
        static IAllNeurons* Create() { return new AllLIFNeurons(); }

#if defined(USE_GPU)
    public:
        /**
         *  Create an AllNeurons class object in device
         *
         *  @param pAllNeurons_d       Device memory address to save the pointer of created AllNeurons object.
         *  @param pAllNeuronsProps_d  Pointer to the neurons properties in device memory.
         */
        virtual void createAllNeuronsInDevice(IAllNeurons** pAllNeurons_d, IAllNeuronsProps *pAllNeuronsProps_d);

#endif // defined(USE_GPU)

#if defined(USE_GPU)
    public:
        /**
         *  Helper for #advanceNeuron. Updates state of a single neuron.
         *
         *  @param  index                 Index of the Neuron to update.
         *  @param  maxSpikes             Maximum number of spikes per neuron per epoch.
         *  @param  deltaT                Inner simulation step duration.
         *  @param  simulationStep        The current simulation step.
         *  @param  pINeuronsProps        Pointer to the neurons properties.
         *  @param  randNoise             Pointer to device random noise array.
         */
        CUDA_CALLABLE virtual void advanceNeuron(const int index, int maxSpikes, const BGFLOAT deltaT, uint64_t simulationStep, IAllNeuronsProps* pINeuronsProps, float* randNoise);

#else  // defined(USE_GPU)

    protected:
        /**
         *  Helper for #advanceNeuron. Updates state of a single neuron.
         *
         *  @param  index                 Index of the Neuron to update.
         *  @param  maxSpikes             Maximum number of spikes per neuron per epoch.
         *  @param  deltaT                Inner simulation step duration.
         *  @param  simulationStep        The current simulation step.
         *  @param  pINeuronsProps        Pointer to the neurons properties.
         *  @param  normRand              Pointer to the normalized random number generator.
         */
        virtual void advanceNeuron(const int index, int maxSpikes, const BGFLOAT deltaT, uint64_t simulationStep, IAllNeuronsProps* pINeuronsProps, Norm* normRand);

#endif // defined(USE_GPU)
 
    protected:
        /**
         *  Initiates a firing of a neuron to connected neurons.
         *
         *  @param  index                 Index of the neuron to fire.
         *  @param  maxSpikes             Maximum number of spikes per neuron per epoch.
         *  @param  deltaT                Inner simulation step duration.
         *  @param  simulationStep        The current simulation step.
         *  @param  pINeuronsProps        Pointer to the neurons properties.
         */
        CUDA_CALLABLE virtual void fire(const int index, int maxSpikes, const BGFLOAT deltaT, uint64_t simulationStep, IAllNeuronsProps* pINeuronsProps) const;
};

#if defined(USE_GPU)

/* -------------------------------------*\
|* # CUDA Global Functions
\* -------------------------------------*/

__global__ void allocAllLIFNeuronsDevice(IAllNeurons **pAllNeurons, IAllNeuronsProps *pAllNeuronsProps);

#endif // USE_GPU
