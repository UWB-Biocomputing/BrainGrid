/*!
  @file norm.h
  @brief  Normally distributed random numbers
  @author Michael Stiber
  @date $Date: 2006/11/18 04:42:32 $
  @version $Revision: 1.1.1.1 $
*/

// $Log: norm.h,v $
// Revision 1.1.1.1  2006/11/18 04:42:32  fumik
// Import of KIIsimulator
//
// Revision 1.2  2005/03/08 19:54:02  stiber
// Modified comments for Doxygen.
//
// Revision 1.1  2005/02/09 18:56:28  stiber
// Initial revision
//
//

#ifndef _NORM_H_
#define _NORM_H_

#include "MersenneTwister.h"
#include <cmath>

/*!
  @class Norm
  @brief Generate normally distributed random numbers

   This class allows you to create RNG objects that are
   independently seeded upon construction and return normally
   distributed random numbers.

       This function generates normally distributed random numbers
   with mean of mu and standard deviation of sigma, using the
   polar method of Marsaglia and Bray, "A Convenient Method for
   Generating Normal Variables", _SIAM Rev._, 6: 260-264 (1964).

       The algorithm is as follows:
   -# Generate two uniformly distributed numbers, U1 and U2.
      Let Vi = 2*Ui-1 for i=1,2, and let W = V1*V1 + V2*V2.
   -# If W > 1, go back to step 1.  Otherwise, let
      Y = sqrt(-2*ln(W)/W), X1 = V1*Y, and X2 = V2*Y.  Then
      X1 and X2 are normally distributed with mean of 0 and
      variance of 1.
   -# Random numbers with mean of mu and standard deviation of sigma
      are calculated by: X_prime = mu + sigma   X.

   Note that numbers are generated in pairs.  On odd-numbered calls
   to operator(), pairs are calculated.  On even-numbered calls,
   the second value is returned.

   Modified from norm.c, from xneuron3
*/
class Norm : public MTRand {
public:
    inline virtual ~Norm() {}

  /*!
    The constructor allows specification of the mean,
    variance (default zero and one, respectively), and initial seed
    for the random number generator. Once created, a Norm object
    cannot have its mean or variance changed.
    @param m mean
    @param s variance
    @param seed seed for random number generator
  */
  Norm(BGFLOAT m = 0.0, BGFLOAT s = 1.0, uint32_t seed = 0)
    : MTRand(seed), odd(true), mu(m), sigma(s) {}

  /*!
    This method makes instances functors; it returns normally
    distributed random numbers. Just a cute way of doing things.
    @return pseudorandom number drawn from a normal distribution.
  */
  virtual BGFLOAT operator() (void);
private:
  // Additional state information

  /*! Which of the pair of pseudorandom numbers was last
    returned. Says whether we should calculate this time  */
  bool odd;

  /*! The second of the pair of pseudorandom numbers generated (last call) */
  BGFLOAT X2;

  /*! Distribution mean */
  BGFLOAT mu;

  /*! Distribution variance */
  BGFLOAT sigma;
};

#endif
