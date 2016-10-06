/**
  @file VectorMatrix.h
  @brief  An efficient implementation of a dynamically-allocated 1D array
  @author Michael Stiber
  @date $Date: 2006/11/22 07:07:35 $
  @version $Revision: 1.2 $
*/

// Written December 2004 by Michael Stiber

// $Log: VectorMatrix.h,v $
// Revision 1.2  2006/11/22 07:07:35  fumik
// DCT growth model first check in
//
// Revision 1.1.1.1  2006/11/18 04:42:32  fumik
// Import of KIIsimulator
//
// Revision 1.4  2005/03/08 19:53:47  stiber
// Modified comments for Doxygen.
//
// Revision 1.3  2005/02/17 15:35:21  stiber
// Modified to support math operations (operator*) with SparseMatrices.
//
// Revision 1.2  2005/02/09 18:55:26  stiber
// "Completely debugged".
//
// Revision 1.1  2004/12/06 20:03:05  stiber
// Initial revision
//


#ifndef _VECTORMATRIX_H_
#define _VECTORMATRIX_H_

#include <string>

#include "Matrix.h"
#include "CompleteMatrix.h"
#include "SparseMatrix.h"
#include "Norm.h"

using namespace std;

// Forward declarations
class VectorMatrix;
class CompleteMatrix;
class SparseMatrix;

const VectorMatrix operator-(BGFLOAT c, const VectorMatrix& rhs);

const VectorMatrix operator/(BGFLOAT c, const VectorMatrix& rhs);

const VectorMatrix operator*(const VectorMatrix& v, const SparseMatrix& m);

const VectorMatrix sqrt(const VectorMatrix& v);

const VectorMatrix exp(const VectorMatrix& v);

/**
  @class VectorMatrix
  @brief An efficient implementation of a dynamically-allocated 1D
  array

  This is a self-allocating and de-allocating 1D array
  that is optimized for numerical computation. A bit of trial and
  error went into this. Originally, the idea was to manipulate
  VectorMatrices using superclass pointers, which would allow generic
  computation on mixtures of subclass objects. However, that doesn't
  work too well with numeric computation, because of the need to have
  "anonymous" intermediate results. So, instead this is implemented as
  most classes would be, with the hope that compilers will optimize
  out unnecessary copying of objects that are intermediate results in
  numerical computations. Note that, while this class keeps track of
  the number of rows and columns it has, no distinction is made
  between row and column vectors, and in fact it is treated as either,
  depending on the context of the mathematical operation.
*/
class VectorMatrix : public Matrix
{
public:

  /**
    Allocate storage and initialize attributes. Either
    "rows" or "columns" must be equal to 1. If "v" is not empty, it
    will be used as a source of data for initializing the vector (and
    must be a list of whitespace separated textual numeric data with the
    same number of elements as this VectorMatrix).
    @throws Matrix_bad_alloc
    @throws Matrix_invalid_argument
    @param t Matrix type
    @param i Matrix initialization
    @param r rows in Matrix
    @param c columns in Matrix
    @param m multiplier used for initialization
    @param v values for initializing VectorMatrix
  */
  VectorMatrix(string t = "complete", string i = "const", int r = 1,
	       int c = 1, BGFLOAT m = 0.0, string v = "");

  /**
    @brief Copy constructor. Performs a deep copy.
    @param oldV The source VectorMatrix
  */
  VectorMatrix(const VectorMatrix& oldV);

  /**
    @brief De-allocate storage
  */
  virtual ~VectorMatrix();


  /**
    @brief Set elements of vector to a constant. Doesn't change its size.
    @param c the constant
  */
  const VectorMatrix& operator=(BGFLOAT c);

  /**
    @brief Assignment operator
    @param rhs right-hand side of assignment
    @return returns reference to this VectorMatrix (after assignment)
  */
  const VectorMatrix& operator=(const VectorMatrix& rhs);

  /**
    @brief Polymorphic output. Produces text output on stream "os"
    @param os stream to output to
  */
  virtual void Print(ostream& os) const;

  /**
    @brief Produce XML representation of vector in string return value.
  */
  virtual string toXML(string name="") const;

  /** @name Accessors
   */
  //@{
  /**
    @brief Access element of a VectorMatrix. Zero-based index; constant-time.
    @param i index
    @return item within this VectorMatrix
  */
  inline const BGFLOAT& operator[](int i) const { return theVector[i]; }

  /**
    @brief access element of a VectorMatrix. Zero-based index; constant-time.
    @param i index
    @return item within this VectorMatrix
  */
  inline const BGFLOAT& at(int i) const { return theVector[i]; }

  /**
    @brief The length of a VectorMatrix elements.
    @return The number of elements in the VectorMatrix
  */
  int Size(void) const { return size; }
  //@}


  /** @name Mutators
   */
  //@{

  /**
    @brief mutate element of a VectorMatrix. Zero-based index; constant time.
    @param i index
    @return Reference to item within this VectorMatrix
  */
  inline BGFLOAT& operator[](int i) { return theVector[i]; }

  /**
    @brief mutate element of a VectorMatrix. Zero-based index.
    @param i index
    @return reference to item within this VectorMatrix
  */
  inline BGFLOAT& at(int i) { return theVector[i]; }
  //@}

  /** @name Math Operations
   */
  //@{

  /**
    @brief Compute the sum of two VectorMatrices of the same length.
    @throws Matrix_domain_error
    @param rhs right-hand argument to the addition. Must be same
    length as this.
    @return A new VectorMatrix, with value equal to the sum of this
    one and rhs and length the same as both, returned by value.
   */
  virtual const VectorMatrix operator+(const VectorMatrix& rhs) const;

  /**
    @brief vector plus a constant. Adds each element of the vector
    plus a constant. Method version (vector on LHS).
    @param c The constant
    @return A vector the same size as the current one
  */
  virtual const VectorMatrix operator+(BGFLOAT c) const;

  /**
    @brief There are two possible vector products. This is an inner product.
    @throws Matrix_domain_error
    @param rhs right-hand argument to the inner product.
    @return A scalar of type BGFLOAT
   */
  virtual BGFLOAT operator*(const VectorMatrix& rhs) const;

  /**
    @brief Vector times a Complete Matrix.

    Treats this vector as a row
    vector; multiplies by matrix with same number of rows as size of
    this vector to produce a new vector the same size as the number of
    columns of the matrix
    @throws Matrix_domain_error
    @param rhs right-hand argument to the vector/matrix product
    @return A vector the same size as the number of columns of rhs
   */
  virtual const VectorMatrix operator*(const CompleteMatrix& rhs) const;

  /**
    @brief Element-by-element multiplication of two vectors.
    @throws Matrix_domain_error
    @param rhs right-hand argument to the vector/matrix product. Must
    be same size as current vector
    @return A vector the same size as the current vector
  */
  virtual const VectorMatrix ArrayMultiply(const VectorMatrix& rhs) const;

  /**
    @brief Constant times a vector.
    Multiplies each element of the vector by a constant. Method
    version (vector on LHS)
    @param c The constant
    @return A vector the same size as the current one
  */
  virtual const VectorMatrix operator*(BGFLOAT c) const;

  /**
    @brief Vector divided by a constant. Divides each element of the vector by a constant.
    @param c The constant
    @return A vector the same size as the current one
  */
  virtual const VectorMatrix operator/(BGFLOAT c) const;

  /**
    @brief Limit values of a vector. Clip values to lie within range.
    @param low lower limit
    @param high upper limit
    @return A vector the same size as the current one
  */
  virtual const VectorMatrix Limit(BGFLOAT low, BGFLOAT high) const;

  /**
    @brief Find minimum value of vector
    @return A scalar
  */
  virtual BGFLOAT Min(void) const;

  /**
    @brief Find maximum value of vector
    @return A scalar
  */
  virtual BGFLOAT Max(void) const;

  /**
    @brief Compute and assign the sum of two VectorMatrices of the same length.
    @throws Matrix_domain_error
    @param rhs right-hand argument to the addition. Must be same
    length as this.
    @return reference to this
   */
  virtual const VectorMatrix& operator+=(const VectorMatrix& rhs);

  /**
    @brief Constant times a vector.

    Multiplies each element of the vector by a constant. Function
    version (constant on LHS)
    @param c The constant
    @param rhs The vector
    @return A vector the same size as the rhs
  */
  friend 
  inline const VectorMatrix operator*(BGFLOAT c, const VectorMatrix& rhs)
  {
    return rhs * c;
  }

  /**
    @brief Vector times sparse matrix.

    Size of v must equal to number of rows of m. Size of resultant
    vector is equal to number Of columns of m.
    @throws Matrix_domain_error
    @return A VectorMatrix with size equal to number of columns of m.
  */
  friend const VectorMatrix operator*(const VectorMatrix& v, const SparseMatrix& m);

  /**
    @brief Constant minus a vector. Subtracts each element of the vector from a constant.
    @param c The constant
    @param v The vector
    @return A vector the same size as the rhs
  */
  friend
  const VectorMatrix operator-(BGFLOAT c, const VectorMatrix& v);

  /**
    @brief Constant divided by a vector. Divides the constant by each element of a vector
    @param c The constant
    @param v The vector
    @return A vector the same size as the rhs
  */
  friend
  const VectorMatrix operator/(BGFLOAT c, const VectorMatrix& v);

  /**
    @brief Constant plus a vector. Adds each element of the vector and a constant
    @param c The constant
    @param rhs The vector
    @return A vector the same size as the rhs
  */
  friend
  inline const VectorMatrix operator+(BGFLOAT c, const VectorMatrix& rhs)
  {
    return rhs + c;
  }

  /**
    @brief Element-wise square root of vector. Computes square root of each element of vector.
    @param v The vector
    @return A vector the same size as the v
  */
  friend
  const VectorMatrix sqrt(const VectorMatrix& v);

  /**
    @brief Element-wise e^x for vector. Computes exp(v[i]) of each element of vector.
    @param v The vector
    @return A vector the same size as the v
  */
  friend
  const VectorMatrix exp(const VectorMatrix& v);
  //@}

protected:

  /** @name Internal Utilities
   */
  //@{

  /**
    @brief Frees up all dynamically allocated storage
   */
  void clear(void);

  /**
    @brief Performs a deep copy
    @param source VectorMatrix to copy from
   */
  void copy(const VectorMatrix& source);

  /**
    @brief Allocates storage for internal Vector storage
    @throws Matrix_bad_alloc
    @throws MatrixException
    @param size number of Vector elements
   */
  void alloc(int size);
  //@}

  // access adjustment --- allow member functions in this class to
  // access protected member of base class in other objects.
  using Matrix::dimensions;
  using Matrix::rows;
  using Matrix::columns;

private:

  /** Pointer to dynamically allocated 1D array */
  BGFLOAT *theVector;

  /** The number of elements in "theVector" */
  int size;

  /** A normal RNG for the whole class */
  static Norm nRng;

};


#endif
