/**
 * @file Timer.h
 *
 * @authors Munehiro Fukuda
 *
 * @date March 5, 2004
 *
 * @brief Timer class, from CSS432
 */

#ifndef _TIMER_H_
#define _TIMER_H_

#include <iostream>

using namespace std;

extern "C"
{
#ifdef _WIN32	//windows portability 
#include <windows.h>	//includes timeval struct
//gettimeofday sets the first parameter struct timeval with the
//seconds and microseconds that have elapsed since the epoc time (2nd
//struct timeval param is not used in implementation). gettimeofday is
//part of sys/time.h which is unavailable to WIN32. A solution to this
//is to use SYSTEMTIME to receive a high resolution time.
static int gettimeofday(struct timeval *tval, struct timeval *alwaysNULL){
	if(tval != NULL){
		SYSTEMTIME systemtime;
		FILETIME filetime;
		ULARGE_INTEGER ulargeint;
		_int64 time;

		GetSystemTime(&systemtime);
		SystemTimeToFileTime(&systemtime, &filetime);

		ulargeint.LowPart = filetime.dwLowDateTime;
		ulargeint.HighPart = filetime.dwHighDateTime;

		time = ulargeint.QuadPart;	//returns as 100-nanosecond intervals 
		time -= (_int64)116444736000000000;	//subtract epoc time: 00:00:00 on January 1, 1970
		time /=10;	//dividing by 10 to get microseconds, 1000 nanoseconds = 1 microsecond
		tval->tv_sec = (long)(time/1000000);	//time interval in seconds 
		tval->tv_usec = (long)(time%1000000);	//remaining time interval in microseconds
		return 0;	//if successful, returns 0
	}
	return -1;	//return -1 on error
};
#else	//not WIN32
#include <sys/time.h>
#endif
}

/**
 * @class Timer
 *
 *
 * Class that encapsulates timing capability, for performance
 * measurement. Includes lap timing capability.
 *
 * \latexonly  \subsubsection*{Credits} \endlatexonly
 * \htmlonly   <h3>Credits</h3> \endhtmlonly
 *
 * Code originally developed by Prof. Munehiro Fukuda for CSS 432 at
 * the University of Washington Bothell.
 */
class Timer {
 public:
  Timer( );                  // Constructor
  void start( );             // Memorize the current time in startTime
  long lap( );               // endTime - startTime (in microseconds)
  long lap(long oldTv_sec, long oldTv_usec); // endTime - oldTime (in microseconds)
  long getSec( );            // get startTime.tv_sec
  long getUsec( );           // get startTime.tv_usec
 private:
  struct timeval startTime;  // Memorize the time to have started an evaluation
  struct timeval endTime;    // Memorize the time to have stopped an evaluation
};

#endif
