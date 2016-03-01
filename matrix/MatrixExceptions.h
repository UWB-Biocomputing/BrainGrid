/**
  @file MatrixExceptions.h
  @brief Exception class hierarchy for Matrix classes
  @author Michael Stiber
  @date 7/30/14
*/

// Written December 2004 by Michael Stiber

// $Log: KIIexceptions.h,v $
// Revision 1.1.1.1  2006/11/18 04:42:31  fumik
// Import of KIIsimulator
//
// Revision 1.2  2005/03/08 19:52:11  stiber
// Modified comments for Doxygen.
//
// Revision 1.1  2005/02/09 18:50:48  stiber
// Initial revision
//
// Revision 1.1  2004/12/06 20:03:05  stiber
// Initial revision
//


#pragma once

#include <stdexcept>

using namespace std;

/**
  @brief Master base class for Matrix exceptions
*/
class MatrixException : public runtime_error
{
public:
  explicit MatrixException(const string&  __arg) : runtime_error(__arg) {}
};

/**
  @brief Signals memory allocation error for Matrix classes
*/
class Matrix_bad_alloc : public MatrixException
{
public:
  explicit Matrix_bad_alloc(const string&  __arg) : MatrixException(__arg) {}
};

/**
  @brief Signals bad cast among Matrices for Matrix classes
*/
class Matrix_bad_cast : public MatrixException
{
public:
  explicit Matrix_bad_cast(const string&  __arg) : MatrixException(__arg) {}
};

 
/**
  @brief Signals bad function argument for Matrix classes
*/
class Matrix_invalid_argument : public MatrixException
{
public:
  explicit Matrix_invalid_argument(const string&  __arg) : MatrixException(__arg) {}
};

 
/**
  @brief Signals value bad for domain for Matrix classes
*/
class Matrix_domain_error : public MatrixException
{
public:
  explicit Matrix_domain_error(const string&  __arg) : MatrixException(__arg) {}
};


