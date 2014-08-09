/**
 ** \brief A factoy class for stimulus input classes.
 **
 ** \class FSInput FSInput.h "FSInput.h"
 **
 ** \latexonly  \subsubsection*{Implementation} \endlatexonly
 ** \htmlonly   <h3>Implementation</h3> \endhtmlonly
 **
 ** The FSInput creates an instance of stimulus input class object.
 **
 ** \latexonly  \subsubsection*{Credits} \endlatexonly
 ** \htmlonly   <h3>Credits</h3> \endhtmlonly
 **
 ** This simulator is a rewrite of CSIM (2006) and other work (Stiber and Kawasaki (2007?))
 **
 **
 **     @author Fumitaka Kawasaki
 **/

/**
 ** \file FSInput.h
 **
 ** \brief Header file for FSInput.
 **/

#pragma once

#ifndef _FSINPUT_H_
#define _FSINPUT_H_

#include "ISInput.h"

class FSInput 
{
public:
    //! The constructor for FSInput.
    FSInput();
    ~FSInput();

    //! Create an instance.
    ISInput* CreateInstance(SimulationInfo* psi,string stimulusInputFileName);

protected:
};

#endif // _FSINPUT_H_
