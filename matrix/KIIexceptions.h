/*!
  @file KIIexceptions.h
  @brief Exception class hierarchy for standalone RKII simulator
  @author Michael Stiber
  @date $Date: 2006/11/18 04:42:31 $
  @version $Revision: 1.1.1.1 $
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

using namespace std;

#ifndef _KIIEXCEPTIONS_H_
#define _KIIEXCEPTIONS_H_

#include <stdexcept>

/*!
  @brief Master base class for KII exceptions
*/
class KII_exception : public runtime_error
{
public:
  explicit KII_exception(const string&  __arg) : runtime_error(__arg) {}
};

/*!
  @brief Signals memory allocation error for KII program
*/
class KII_bad_alloc : public KII_exception
{
public:
  explicit KII_bad_alloc(const string&  __arg) : KII_exception(__arg) {}
};

/*!
  @brief Signals bad cast among Matrices for KII program
*/
class KII_bad_cast : public KII_exception
{
public:
  explicit KII_bad_cast(const string&  __arg) : KII_exception(__arg) {}
};

 
/*!
  @brief Signals bad function argument for KII program
*/
class KII_invalid_argument : public KII_exception
{
public:
  explicit KII_invalid_argument(const string&  __arg) : KII_exception(__arg) {}
};

 
/*!
  @brief Signals value bad for domain for KII program
*/
class KII_domain_error : public KII_exception
{
public:
  explicit KII_domain_error(const string&  __arg) : KII_exception(__arg) {}
};

 
#endif
