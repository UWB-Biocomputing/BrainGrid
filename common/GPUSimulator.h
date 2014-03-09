/**
 * @file GPUSimulator.h
 *
 * @authors Sean Blackbourn
 *
 * @ GPU Simulator class that inherits from Simulator base class.
 */

#pragma once


#include "Simulator.h"

class GPUSimulator : public Simulator {

public:
	GPUSimulator(Network *network, SimulationInfo sim_info);
	//virtual ~SingleThreadedSim() ; 
};

