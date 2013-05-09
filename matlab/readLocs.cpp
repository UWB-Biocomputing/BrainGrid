/*!
  @file readLocs.cpp
  @brief MATLAB readLocs function
  [x, y] = readLocs(filename)
  @author Michael Stiber
  @date $Date: 2006/11/18 04:42:32 $
  @version $Revision: 1.1.1.1 $
*/

/*
  MATLAB readLocs function
  [x, y] = readLocs(filename)
*/

// $Log: readLocs.cpp,v $
// Revision 1.1.1.1  2006/11/18 04:42:32  fumik
// Import of KIIsimulator
//
// Revision 1.1  2005/03/24 14:59:30  stiber
// Initial revision
//
//

#include <cmath>
#include <cstdlib>
#include <sstream>
#include <iostream>

#include "mex.h"
#include "matrix.h"
#include "tinyxml.h"
#include "SourceVersions.h"

using namespace std;

static VersionInfo version("$Id: readLocs.cpp,v 1.1.1.1 2006/11/18 04:42:32 fumik Exp $");

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{ 
  if (nrhs != 1) {
    mexErrMsgTxt("wrong number of args (must be 1)");
    return;
  }

  if (nlhs != 2) {
    mexErrMsgTxt("wrong number of return values (must be 2)");
    return;
  }

  // Get file name
  int buflen = (mxGetM(prhs[0]) * mxGetN(prhs[0]) * sizeof(mxChar)) + 1;
  char stateFileName[buflen];
  if (mxGetString(prhs[0], stateFileName, buflen)) {
    mexErrMsgTxt("Unable to get file name from MATLAB variable.");
    return;
  }

  TiXmlDocument simDoc(stateFileName);
  if (!simDoc.LoadFile()) {
    char errorMessage[255];
    sprintf(errorMessage, "Unable to open file %s; error was %s on line %d, column %d.\n",
	    stateFileName, simDoc.ErrorDesc(), simDoc.ErrorRow(), simDoc.ErrorCol());
    mexErrMsgTxt(errorMessage);
    return;
  }

  TiXmlElement* simState = NULL;

  if ((simState = simDoc.FirstChildElement("SimParams"))==NULL) {
    mexErrMsgTxt("Could not find <SimParams> in file.");
    return;
  }

  // Get the number of units
  TiXmlElement* temp = NULL;
  int N;
  if ((temp = simState->FirstChildElement("Units")) == NULL)
    mexErrMsgTxt("Unable to find <Units>.");
  if (temp->QueryIntAttribute("number", &N)!=TIXML_SUCCESS)
    mexErrMsgTxt("Unable to read number of units.");

  // Allocate storage
  plhs[0] = mxCreateDoubleMatrix(1, N, mxREAL);
  double* xlocs = mxGetPr(plhs[0]);
  plhs[1] = mxCreateDoubleMatrix(1, N, mxREAL);
  double* ylocs = mxGetPr(plhs[1]);

  // Get the unit locations
  if ((temp = simState->FirstChildElement("UnitLocations")) == NULL)
    mexErrMsgTxt("Unable to find <UnitLocations>.");
      
  // Get the locations by iterating over the child elements with the
  // value "Unit"
  for (TiXmlElement* child = temp->FirstChildElement("Unit"); child != NULL; 
       child=child->NextSiblingElement("Unit")) {
    int number;   // Attributes of each Unit
    double x, y;
    if (child->QueryIntAttribute("number", &number)!=TIXML_SUCCESS)
      mexErrMsgTxt("Unable to get number of a unit.");
    if (child->QueryDoubleAttribute("x", &x)!=TIXML_SUCCESS)
      mexErrMsgTxt("Unable to get unit x location.");
    if (child->QueryDoubleAttribute("y", &y)!=TIXML_SUCCESS)
      mexErrMsgTxt("Unable to get unit y location.");

    xlocs[number] = x;
    ylocs[number] = y;
  }

}

