#ifndef _NETWORK_H_
#define _NETWORK_H_

#include "global.h"

class Model {

    public:
        Model(FLOAT Iinject[2], FLOAT Inoise[2],FLOAT Vthresh[2],
            FLOAT Vresting[2], FLOAT Vreset[2], FLOAT Vinit[2],
            FLOAT starter_Vthresh[2],FLOAT starter_Vreset[2],
            FLOAT new_targetRate);
};

#endif
