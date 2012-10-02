/*!  
  @file SourceVersions.h
  @brief Class for accumulating and outputting source code version in a program
  @author Michael Stiber
  @date $Date: 2006/11/18 04:42:32 $
  @version $Revision: 1.1.1.1 $
*/

// $Log: SourceVersions.h,v $
// Revision 1.1.1.1  2006/11/18 04:42:32  fumik
// Import of KIIsimulator
//
// Revision 1.3  2005/03/08 19:53:22  stiber
// Modified comments for Doxygen.
//
// Revision 1.2  2005/02/18 13:43:22  stiber
// Finally got it working (it wasn't easy).
//
// Revision 1.1  2005/02/17 16:36:44  stiber
// Initial revision
//
//

#ifndef _SOURCEVERSIONS_H_
#define _SOURCEVERSIONS_H_


#include <string>
#include <vector>
#include <iostream>

using namespace std;

/*!
  @class SourceVersions
  This class accumulates all version information for the
  set of source files in a program in a single, class-wide static
  variable. This can be used to produce XML comments that list these
  versions, for incorporation as metadata in program output. To
  use this, do the following in a .cpp file:
  @code
  #include "SourceVersions.h"

  static VersionInfo version("$Id: SourceVersions.h,v 1.1.1.1 2006/11/18 04:42:32 fumik Exp $");
  @endcode
  Use operator<<() on any VersionInfo object to
  output XML comments containing all files' version information.
*/
class SourceVersions {
public:

  /*!
    Adds the given string to the class-global set of
    version information strings. This can be called explicitly.
    @param info source file version information (e.g., from RCS Id keyword).
   */
  void addInfo(string info);

  /*!
    @brief Outputs all of the version information to XML comments.
    @return string holding XML
   */
  string toXML(void);

private:

  /*! Collects all source file version information. Global to class. */
  vector<string> versions;
};

/*! Global object that holds all the version information */
extern SourceVersions* sv;

/*!
  @class VersionInfo
  @brief Class for initialization of and access to the global
  SourceVersions object pointed to by ::sv.
 */
class VersionInfo
{
public:

  /*!
    The constructor dynamically allocates the global
    SourceVersions object if it has not
    yet been created (indicated by the initted class variable). This
    approach is necessary to ensure that the global object is
    initialized the first time any module tries to add version
    information (and that the constructor doesn't run afterwards,
    erasing any previously added information). By allocating the
    global object dynamically, we guarantee that it and its members'
    constructors are run when the first VersionInfo constructor
    executes.
    @param info Version information added to the SourceVersions object
   */
  VersionInfo(string info);

private:

  /*! Indicates whether the global SourceVersions object pointed to by
    sv has initialized.
  */
  static bool initted;
};

/*!
  @brief Stream output operator for VersionInfo.

  This results in the XML for the global SourceVersions object pointed
  to by ::sv being sent to the output stream.
  @param os the output stream
  @param obj the VersionInfo object to send to the output stream
 */
ostream& operator<<(ostream& os, const VersionInfo& obj);

#endif
