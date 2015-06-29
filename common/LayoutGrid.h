#pragma once

#include "Global.h"
#include "SimulationInfo.h"
#include "Layout.h"

using namespace std;

class LayoutGrid : public Layout
{
    public:
        // TODO
        LayoutGrid();
        virtual ~LayoutGrid();

    protected:
        virtual void initNeuronsLocs(const SimulationInfo *sim_info);
};

