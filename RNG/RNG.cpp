/*!
 @file RNG.cpp
 @brief  Random Number Generator Class
 @author Michael Stiber
 @date $Date: 2006/11/22 07:07:35 $
 @version $Revision: 1.2 $
 */

/************************************************************
 RNG.cpp -- Random Number Generator Class

 The following class allows you to create RNG objects that are
 independently seeded upon construction, and thus should produce
 distinct sequences. It is a combination of wrapper and functor
 design patterns: it serves as a wrapper around the C srand() and
 rand() functions, and it has operator() overloaded so that
 instances of this class can act syntactically like function calls
 --- they are function objects. RNGs produce uniformly distributed
 random numbers in the range [0,1].

 ************************************************************/

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>  // For random_shuffle
#include <cstdlib>    // For the C RNG
#include <cstdio>     // For /dev/random
#ifdef _WIN32
#include <Windows.h>
#include <time.h>
#include "bgtypes.h"	//defines FLOAT, needed for WIN32 compiling
#endif

#include "RNG.h"

#include "SourceVersions.h"

static VersionInfo version("$Id: RNG.cpp,v 1.2 2006/11/22 07:07:35 fumik Exp $");

using namespace std;

// We want to seed the STL random_shuffle algorithm. Unfortunately,
// the only way to do this is to pass in a random number generator
// object, as the STL currently doesn't provide a way to seed the
// default RNG.


// The constructor seeds the C RNG. Note that this definition assumes
// that the OS provides a /dev/random, kernel-level RNG, which many
// Unix-like OSes do (including Linux and Mac OS X). The kernel-level
// RNG gathers environmental noise from from hardware and uses it to
// create a stream of random bytes which can be read from
// /dev/random. If your OS doesn't provide /dev/random, you will need
// to modify this to use another approach --- such as the current time
// (including seconds). In that case, MAKE SURE YOU USE CONDITIONAL
// COMPILATION DIRECTIVES SO THAT THE CODE BELOW IS STILL COMPILED ON
// OTHER OSES.
RNG::RNG(unsigned long seed) {
#ifdef _WIN32
	if (seed == 0)
	seed = (unsigned long) time(NULL);

	srand(seed);

#else
	// Seed the RNG
	if (seed == 0) { // We need to get a seed from /dev/random
		FILE *dev_random;
		if ((dev_random = fopen("/dev/random", "r")) == NULL) {
			cerr << "randpick: couldn't open /dev/random for reading" << endl;
			exit(1);
		}
		fread(&seed, sizeof(seed), 1, dev_random);
		fclose(dev_random);
	}

	// Seed the RNG and save the state array
	initstate(seed, state, stateSize);
#endif

}

FLOAT RNG::operator()(void) {
#ifdef _WIN32
	return FLOAT(rand()) / RAND_MAX;
#else
	//setstate(state);
	return FLOAT(random()) / RAND_MAX;
#endif
}

FLOAT RNG::inRange(FLOAT min, FLOAT max) {
	return min + (max - min) * this->operator ()();
}
