/*!
 @file Matrix.h
 @brief Abstract base class for Matrices
 @author Michael Stiber
 @date $Date: 2006/11/18 04:42:31 $
 @version $Revision: 1.1.1.1 $
*/

// Written December 2004 by Michael Stiber

// $Log: Matrix.h,v $
// Revision 1.1.1.1  2006/11/18 04:42:31  fumik
// Import of KIIsimulator
//
// Revision 1.3  2005/03/08 19:52:47  stiber
// Modified comments for Doxygen.
//
// Revision 1.2  2005/02/09 18:52:22  stiber
// "Competely debugged".
//
// Revision 1.1  2004/12/06 20:03:05  stiber
// Initial revision
//

#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <string>

#include "KIIexceptions.h"

// The tinyXML library, for deserialization via a MatrixFactory
#include "../include/tinyxml.h"

using namespace std;

/*!
 @class Matrix
 @brief Abstract base class for Matrices

 It is the intent of this class to provide an efficient superclass
 for self-allocating and de-allocating 1D and 2D Vectors and
 Matrices. Towards that end, subclasses are expected to implement
 member functions in ways that may require the classes to be friends
 of each other (to directly access internal data
 representation). This base class defines the common interface;
 clients should be able to perform the full range of math operations
 using only Matrix objects.
*/
class Matrix
{
public:

  /*!
    Purely here to make the destructor virtual; no need
    for base class destructor to do anything.
  */
  virtual ~Matrix() { }

  /*!
    @brief Generate text representation of the Matrix to a stream
    @param os Output stream.
  */
  virtual void Print(ostream& os) const = 0;

protected:

  /*!
    Initialize attributes at construction time. This is protected to
    prevent construction of Matrix objects themselves. Would be nice
    if C++ just allowed one to declare a class abstract. Really
    obsoleted since the Print() method is pure virtual now.
  */
  Matrix(string t = "", string i = "", int r = 0, int c = 0, FLOAT m = 0.0);

  /*!
    @brief Convenience mutator
    @param t Matrix type
    @param i Matrix initialization
    @param r rows in Matrix
    @param c columns in Matrix
    @param m multiplier used for initialization
    @param d indicates one or two dimensional
  */
  void SetAttributes(string t, string i, int r, int c, FLOAT m, int d);

  /*! @name Attributes from XML files */
  //@{

  /*! "complete", "diag", or "sparse" */
  string type;

  /*! "const", "random", or "implementation" */
  string init;

  /*! Number of rows in Matrix (>0) */
  int rows;

  /*! Number of columns in Matrix (>0) */
  int columns;

  /*! Constant used for initialization  */
  FLOAT multiplier;

  //@}

  // Other common attributes
  /*! One or two dimensional */
  int dimensions;
};

/*!
  Stream output operator for the Matrix class
  hierarchy. Subclasses must implement the Print() method
  to take advantage of this.
  @param os the output stream
  @param obj the Matrix object to send to the output stream
 */
ostream& operator<<(ostream& os, const Matrix& obj);


#endif
