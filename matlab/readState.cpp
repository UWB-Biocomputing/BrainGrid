/*!
  @file readState.cpp
  @brief MATLAB readState function
  [mState, gState, mRadii, gRadii] = readState(filename)
  @author Michael Stiber
  @date $Date: 2006/11/22 07:07:35 $
  @version $Revision: 1.2 $
*/

/*
  MATLAB readState function
  [mState, gState, mRadii, gRadii] = readState(filename)
*/

// $Log: readState.cpp,v $
// Revision 1.2  2006/11/22 07:07:35  fumik
// DCT growth model first check in
//
// Revision 1.1.1.1  2006/11/18 04:42:32  fumik
// Import of KIIsimulator
//
// Revision 1.5  2005/03/08 19:56:48  stiber
// Modified comments for Doxygen.
//
// Revision 1.4  2005/03/07 16:24:17  stiber
// Improved error messages returned to MATLAB. Modified calls
// to tinyxml Attribute() method to store return value in a
// const char* vaiable, rather than a string, to facilitate checking
// for NULL. Changed calls to generic Attribute() method to
// QueryIntAttribute().
//
// Revision 1.3  2005/02/18 13:42:03  stiber
// Added SourceVersions support.
//
// Revision 1.2  2005/02/09 21:54:24  stiber
// Modified so that it will handle zero-length sequences, returning
// empty (0 by 0) mxArrays.
//
// Revision 1.1  2005/02/09 18:47:31  stiber
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

using namespace std;

// Read a Vector from XML
void ReadVector(TiXmlElement* vector, double dest[], 
		int rows, int columns)
{
  // Get contents of a row of the matrix
  TiXmlHandle vecHandle(vector);
  TiXmlText* valuesNode = vecHandle.FirstChild().Text();
  if (valuesNode == NULL) {
    mexErrMsgTxt("Matrix with empty content.");
    return;
  }
  string values = valuesNode->Value();
  istringstream is(values);
  // Fill the matrix row
  for (int j=0; j<columns; j++)
    is >> dest[j*rows];
}



// Read a sequence of vectors from XML
mxArray* ReadSequence(const TiXmlElement* sequence, int rows)
{
  // An empty sequence is a possibility; return an empty mxArray
  if (rows == 0)
    return mxCreateDoubleMatrix(0, 0, mxREAL);

  mxArray* stateArray;
  
  // A sequence is a sequence of vectors, which we assemble into a
  // 2D MATLAB array
  TiXmlElement* vector = (TiXmlElement*)sequence->FirstChildElement("Matrix");
  if (vector == NULL) {
    mexErrMsgTxt("Number of matrices in Sequence doesn't match attribute.");
    return NULL;
  }
  const char *colCStr = vector->Attribute("columns");
  if (colCStr == NULL) {
    mexErrMsgTxt("Malformed Matrix in state file (no columns).");
    return NULL;
  }
  const int columns = atoi(colCStr);
      
  // Now that we know the number of rows and columns, we can
  // create the MATLAB array
  stateArray = mxCreateDoubleMatrix(rows, columns, mxREAL);
  // Point to the data in it
  double *dataP = mxGetPr(stateArray);
  for (int i=0; i<rows; i++) {
    ReadVector(vector, &dataP[i], rows, columns);

    // Move on to next vector
    vector = vector->NextSiblingElement("Matrix");
  }
  return stateArray;
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{ 
  if (nrhs != 1) {
    mexErrMsgTxt("wrong number of args (must be 1)");
    return;
  }

  if ((nlhs != 2) && (nlhs != 4)) {
    mexErrMsgTxt("wrong number of return values (must be 2 or 4)");
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

  if ((simState = simDoc.FirstChildElement("SimState"))==NULL) {
    mexErrMsgTxt("Could not find <SimState> in file.");
    return;
  }

  // Load mitral and granule radii, if requested
  if (nlhs == 4) {
    for (TiXmlElement* layer = simState->FirstChildElement("Layer"); layer != NULL; 
	 layer=layer->NextSiblingElement("Layer")) {
      string name;
      const char* nameCStr = layer->Attribute("name");
      if (nameCStr == NULL)
	name = "";
      else
	name = nameCStr;
      TiXmlElement* child = layer->FirstChildElement("Radii");
      if (child == NULL) {
	mexErrMsgTxt("Unable to find radii.");
	return;
      }
      child = child->FirstChildElement("Matrix");
      if (child == NULL){
	mexErrMsgTxt("Unable to find radii matrix.");
	return;
      }
      const char *colCStr = child->Attribute("columns");
      if (colCStr == NULL) {
	mexErrMsgTxt("Malformed Matrix in state file (no columns).");
	return;
      }
      const int columns = atoi(colCStr);
      const int rows = 1;    // Assume
    
      // Now that we know the number of rows and columns, we can
      // create the MATLAB array and read the data into it
      if (name == "mitral") {
	plhs[2] = mxCreateDoubleMatrix(rows, columns, mxREAL);
	ReadVector(child, mxGetPr(plhs[2]), rows, columns);
      } else {   // granule radii
	plhs[3] = mxCreateDoubleMatrix(rows, columns, mxREAL);
	ReadVector(child, mxGetPr(plhs[3]), rows, columns);
      }
    }
  }

  // Move to statistics node
  TiXmlElement* curEl = simState->FirstChildElement("Statistics");
  if (curEl == NULL) {
    mexErrMsgTxt("Could not find <Statistics> in file.");
    return;
  }


  // Load sequences of mitral and granule states; each stored in a <Sequence>
  for (TiXmlElement* sequence = curEl->FirstChildElement("Sequence");
       sequence != NULL; sequence=sequence->NextSiblingElement("Sequence")) {
    string name;  // Which Matrix it is
    int rows;

    if (sequence->QueryIntAttribute("items", &rows) != TIXML_SUCCESS) {
      mexErrMsgTxt("Sequence in state file with unknown number of items.");
      return;
    }
    const char* nameCStr = sequence->Attribute("name");
    if (nameCStr == NULL)
      name = "";
    else
      name = nameCStr;
    if (name == "mState")
      plhs[0] = ReadSequence(sequence, rows);
    else if (name == "gState")
      plhs[1] = ReadSequence(sequence, rows);

  }
}

