/** 
 * @authors Aaron Oziel, Sean Blackbourn
 * 
 * @class AllIZHNeurons AllIZHNeurons.h "AllIZHNeurons.h"
 * @brief A container of all LIF neuron data
 *
 *  The container holds neuron parameters of all neurons. 
 *  Each kind of neuron parameter is stored in a 1D array, of which length
 *  is number of all neurons. Each array of a neuron parameter is pointed by a 
 *  corresponding member variable of the neuron parameter in the class.
 *
 *  In this file you will find usage statistics for every variable in the BrainGrid 
 *  project as we find them. These statistics can be used to help 
 *  determine if a variable is being used, where it is being used, and how it
 *  is being used in each class::function()
 *  
 *  For Example
 *
 *  Usage:
 *  - LOCAL VARIABLE -- a variable for individual neuron
 *  - LOCAL CONSTANT --  a constant for individual neuron
 *  - GLOBAL VARIABLE -- a variable for all neurons
 *  - GLOBAL CONSTANT -- a constant for all neurons
 *
 *  Class::function(): --- Initialized, Modified OR Accessed
 *
 *  OtherClass::function(): --- Accessed   
 *
 *  Note: All GLOBAL parameters can be scalars. Also some LOCAL CONSTANT can be categorized 
 *  depending on neuron types. 
 */
#pragma once

#include "Global.h"
#include "AllIFNeurons.h"

// Class to hold all data necessary for all the Neurons.
class AllIZHNeurons : public AllIFNeurons
{
    public:

        //! A constant (0.02, 01) describing the coupling of variable u to Vm;
        BGFLOAT *Aconst;

        //! A constant controlling sensitivity of u
        BGFLOAT *Bconst;

        //! A constant controlling reset of Vm 
        BGFLOAT *Cconst;

        //! A constant controlling reset of u
        BGFLOAT *Dconst;

        //! internal variable
        BGFLOAT *u;

        //!
        BGFLOAT *C3;

        AllIZHNeurons();
        virtual ~AllIZHNeurons();

        virtual void setupNeurons(SimulationInfo *sim_info);
        virtual void cleanupNeurons();  
        virtual int numParameters();
        virtual int readParameters(const TiXmlElement& element);
        virtual void printParameters(ostream &output) const;
        virtual void createAllNeurons(SimulationInfo *sim_info);
        virtual string toString(const int i) const;
        virtual void readNeurons(istream &input, const SimulationInfo *sim_info);
        virtual void writeNeurons(ostream& output, const SimulationInfo *sim_info) const;

    protected:
        void createNeuron(SimulationInfo *sim_info, int neuron_index);
        void setNeuronDefaults(const int index);
        virtual void initNeuronConstsFromParamValues(int neuron_index, const BGFLOAT deltaT);
        void readNeuron(istream &input, const SimulationInfo *sim_info, int i);
        void writeNeuron(ostream& output, const SimulationInfo *sim_info, int i) const;

    private:
        static const BGFLOAT DEFAULT_a = 0.0035;
        static const BGFLOAT DEFAULT_b = 0.2;
        static const BGFLOAT DEFAULT_c = -50;
        static const BGFLOAT DEFAULT_d = 2;

        // TODO
        BGFLOAT m_Aconst[2];
        // TODO
        BGFLOAT m_Bconst[2];
        // TODO
        BGFLOAT m_Cconst[2];
        // TODO
        BGFLOAT m_Dconst[2];

        void freeResources();  
};
