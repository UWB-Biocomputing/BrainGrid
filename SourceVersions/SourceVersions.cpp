/*!  
  @file SourceVersions.cpp
  @brief Class for accumulating and outputting source code version in a program
  @author Michael Stiber
  @date $Date: 2006/11/18 04:42:32 $
  @version $Revision: 1.1.1.1 $
*/

// $Log: SourceVersions.cpp,v $
// Revision 1.1.1.1  2006/11/18 04:42:32  fumik
// Import of KIIsimulator
//
// Revision 1.3  2005/03/08 19:56:04  stiber
// Modified comments for Doxygen.
//
// Revision 1.2  2005/02/18 13:41:04  stiber
// Got it working (it wasn't easy).
//
// Revision 1.1  2005/02/17 16:36:44  stiber
// Initial revision
//
//

#include <iterator>

#include "SourceVersions.h"

// Global object that will hold all version information. Dynamically
// allocated to ensure that it and its members' constructors will be
// called at allocation time.
SourceVersions* sv;

bool VersionInfo::initted = false;


/*
  @method addInfo
  @discussion Adds the given string to the class-global set of
  version information strings. This can be called explicitly.
  @param info source file version information (e.g., from RCS Id keyword).
*/
void SourceVersions::addInfo(string info)
{
  versions.push_back(info);
}

/*
  @method toXML
  @discussion Outputs all of the version information to XML
  comments.
  @result string holding XML
*/
string SourceVersions::toXML(void)
{
  string XML;

  for (vector<string>::iterator it=versions.begin(); it != versions.end(); it++)
    XML = XML + "<!-- " + *it + " -->\n";

  return XML;
}

/*
  @method VersionInfo
  @discussion The constructor dynamically allocates the global
  @link SourceVersions SourceVersions object @/link if it has not
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
VersionInfo::VersionInfo(string info)
{ 
  if (!initted) {
    sv = new SourceVersions;
    sv->addInfo("$Id: SourceVersions.cpp,v 1.1.1.1 2006/11/18 04:42:32 fumik Exp $");
    initted = true;
  }

  sv->addInfo(info);
}



/*
  @function operator<<
  @discussion stream output operator for VersionInfo. This results in
  the XML for the global *sv @link SourceVersions SourceVersions
  object @/link being sent to the output
  stream.
  @param os the output stream
  @param obj the VersionInfo object to send to the output stream
 */
ostream& operator<<(ostream& os, const VersionInfo& obj)
{
  os << sv->toXML();
  return os;
}
