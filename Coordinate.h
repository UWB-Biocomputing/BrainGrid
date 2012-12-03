/**
 ** \class Coordinate Coordinate.h "Coordinate.h"
 **
 ** \brief A container for 2-dimensional coordinates.
 **
 ** \authors Allan Ortiz & Cory Mayberry
 **
 **/

/**
 ** \file Coordinate.h
 **
 ** \brief A container for 2-dimensional coordinates.
 **/

#ifndef _COORDINATE_H_
#define _COORDINATE_H_

//! Utility structure that allows coordinates of points in a 2d array to be 
//	easily tracked with some utility
struct Coordinate {

	//! The constructor for Coordinate.
	Coordinate(int x = 0, int y = 0) :
		x(x), y(y) {
	}

	//! The copy constructor for Coordinate.
	Coordinate(const Coordinate &cp) :
		x(cp.x), y(cp.y) {
	}

	//! The overloaded == operator.
	bool operator==(const Coordinate& other) {
		return other.x == x && other.y == y;
	}

	//! The location in the x dimension.
	int x;
	//! The location in the y dimension.
	int y;
};
#endif
