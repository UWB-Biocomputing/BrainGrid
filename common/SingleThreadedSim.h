/**
 * @file SingleThreadedSim.h
 *
 * @authors Sean Blackbourn
 *
 * @Single Threaded simulator class that inherits from Simulator base class.
 */

#pragma once



#include "Simulator.h"

class SingleThreadedSim : public Simulator {

public:
	SingleThreadedSim(Network *network, SimulationInfo sim_info);
	//virtual ~SingleThreadedSim() ; 
};

