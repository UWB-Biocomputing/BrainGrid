/*!
  @file RNG.h
  @brief Random number generator class
  @author Michael Stiber
  @date $Date: 2006/11/18 04:42:32 $
  @version $Revision: 1.1.1.1 $
*/

#ifndef _RNG_H_
#define _RNG_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>  // For random_shuffle
#include <cstdlib>    // For the C RNG
#include <cstdio>     // For /dev/random
#ifdef _WIN32
#include <time.h>
#include "bgtypes.h"	//defines BGFLOAT, needed for WIN32 compiling
#endif

/*!
  @class RNG
  The following class allows you to create RNG objects that are
  independently seeded upon construction, and thus should produce
  distinct sequences. It is a combination of wrapper and functor
  design patterns: it serves as a wrapper around the C srand() and
  rand() functions, and it has operator()() overloaded so that
  instances of this class can act syntactically like function calls
  --- they are function objects. RNGs produce uniformly distributed
  random numbers in the range [0,1].

  Because instances of this class save and restore the C RNG state
  upon each call, multiple instances produce independent pseudorandom
  number streams.
*/
class RNG {
public:

  /*!
    @brief The constructor seeds the C RNG
    @param seed the seed
  */
  RNG(unsigned long seed = 0);
  BGFLOAT inRange(BGFLOAT min, BGFLOAT max);
  /*!
    This method makes instances functors; it returns uniformly
    distributed reals in the range of 0 through 1.0.
    @return A pseudorandom number taken from a uniform distribution
  */
  virtual BGFLOAT operator() (void);

private:

  /*! Size of internal state of C RNG */
  static const long stateSize = 256;

  /*! Saved internal state of C RNG */
  char state[stateSize];
};

#endif
