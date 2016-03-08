/**
  @file CompleteMatrix.h
  @brief An efficient implementation of a dynamically-allocated 2D array.
  @author Michael Stiber
  @date January 2016
  @version 2
*/

// Written December 2004 by Michael Stiber

// $Log: CompleteMatrix.h,v $
// Revision 1.2  2006/11/22 07:07:34  fumik
// DCT growth model first check in
//
// Revision 1.1.1.1  2006/11/18 04:42:31  fumik
// Import of KIIsimulator
//
// Revision 1.3  2005/03/08 19:50:55  stiber
// Modified comments for Doxygen.
//
// Revision 1.2  2005/02/09 18:49:09  stiber
// "Completely debugged".
//
// Revision 1.1  2004/12/06 20:03:05  stiber
// Initial revision
//


#pragma once

#include <string>

#include "Matrix.h"
#include "VectorMatrix.h"

using namespace std;

// Forward declarations
class CompleteMatrix;

const CompleteMatrix sqrt(const CompleteMatrix& m);

/**
  @class CompleteMatrix
  @brief An efficient implementation of a dynamically-allocated 2D array

  This is a self-allocating and de-allocating 2D array
  that is optimized for numerical computation. A bit of trial and
  error went into this. Originally, the idea was to manipulate
  CompleteMatrices using superclass pointers, which would allow generic
  computation on mixtures of subclass objects. However, that doesn't
  work too well with numeric computation, because of the need to have
  "anonymous" intermediate results. So, instead this is implemented as
  most classes would be, with the hope that compilers will optimize
  out unnecessary copying of objects that are intermediate results in
  numerical computations.
*/
class CompleteMatrix : public Matrix
{
   friend class VectorMatrix;

public:

  /**
    Allocate storage and initialize attributes. If "v" (values) is
    not empty, it will be used as a source of data for initializing
    the matrix (and must be a list of whitespace separated textual
    numeric data with rows * columns elements, if "t" (type) is
    "complete", or number of elements in diagonal, if "t" is "diag").
    
    If "i" (initialization) is "const", then "m" will be used to initialize either all
    elements (for a "complete" matrix) or diagonal elements (for "diag").
   
    "random" initialization is not yet implemented.
   
    @throws Matrix_bad_alloc
    @throws Matrix_invalid_argument
    @param t Matrix type (defaults to "complete")
    @param i Matrix initialization (defaults to "const")
    @param r rows in Matrix (defaults to 2)
    @param c columns in Matrix (defaults to 2)
    @param m multiplier used for initialization (defaults to zero)
    @param v values for initializing CompleteMatrix (this string is parsed as a list of floating point numbers)
  */
  CompleteMatrix(string t = "complete", string i = "const", int r = 2,
		 int c = 2, BGFLOAT m = 0.0, string v = "");

  /**
    @brief Copy constructor. Performs a deep copy.
    @param oldM The source CompleteMatrix
  */
  CompleteMatrix(const CompleteMatrix& oldM);

  /**
    @brief De-allocate storage
  */
  virtual ~CompleteMatrix();

  /**
    @brief Assignment operator
    @param rhs right-hand side of assignment
    @return returns reference to this CompleteMatrix (after assignment)
  */
  CompleteMatrix& operator=(const CompleteMatrix& rhs);

  /**
    @brief access element at (row, column) -- mutator
    @param row element row
    @param column element column
    @return reference to element (lvalue)
  */
  inline BGFLOAT& operator()(int row, int column)
  { return theMatrix[row][column]; }

  /**
    @brief Polymorphic output. Produces text output on stream os. Used by operator<<()
    @param os stream to output to
  */
  virtual void Print(ostream& os) const;

  /**
    @brief Produce XML representation of Matrix in string return value.
    @param name name attribute for XML
  */
  virtual string toXML(string name="") const;

  /** @name Math operations

    For efficiency's sake, these methods will be
    implemented as being "aware" of each other (i.e., using "friend"
    and including the other subclasses' headers).
  */
  //@{

  /**
    @brief Compute the sum of two CompleteMatrices of the same rows and columns.
    @throws Matrix_domain_error
    @param rhs right-hand argument to the addition. Must have same
    dimensions as this.
    @return A new CompleteMatrix, with value equal to the sum of this
    one and rhs and rows and columns the same as both, returned by value.
   */
  virtual const CompleteMatrix operator+(const CompleteMatrix& rhs) const;

  /**
    Matrix product. Number of rows of "rhs" must equal to
    number of columns of this.
    @throws Matrix_domain_error
    @param rhs right-hand argument to the product.
    @return A CompleteMatrix with number of rows equal to this and
    number of columns equal to "rhs".
   */
  virtual const CompleteMatrix operator*(const CompleteMatrix& rhs) const;
  //@}

  friend
  const CompleteMatrix sqrt(const CompleteMatrix& v);

protected:

  /** @name Internal Utilities
   */
  //@{

  /**
    @brief Frees up all dynamically allocated storage
   */
  void clear(void);

  /**
    Performs a deep copy. It is assumed that the storage
    allocate to theMatrix has already been deleted.
    @param source VectorMatrix to copy from
   */
  void copy(const CompleteMatrix& source);

  /**
    @brief Allocates storage for internal Matrix storage
    @throws Matrix_bad_alloc
    @throws MatrixException
    @param rows number of Matrix rows
    @param cols number of Matrix cols
   */
  void alloc(int rows, int cols);

  //@}

  // access adjustment --- allow member functions in this class to
  // access protected member of base class in other objects.
  using Matrix::dimensions;
  using Matrix::rows;
  using Matrix::columns;

private:

  /** Pointer to dynamically allocated 2D array */
  BGFLOAT **theMatrix;

};


