/*!
  @file readMatrix.cpp
  @brief   MATLAB readMatrix function

  A = readMatrix(filename, matname)

  Reads the Matrix 'matname' from the XML file 'filename'. Only reads
  matrices that have their contents explicitly specified (i.e., no
  constant inits, non-unitary multipliers, etc).
  @author Michael Stiber
  @date $Date: 2006/11/18 04:42:32 $
  @version $Revision: 1.1.1.1 $
*/

/*
  MATLAB readMatrix function
  A = readMatrix(filename, matname)

  Reads the Matrix 'matname' from the XML file 'filename'. Only reads
  matrices that have their contents explicitly specified (i.e., no
  constant inits, non-unitary multipliers, etc).
*/

// $Log: readMatrix.cpp,v $
// Revision 1.1.1.1  2006/11/18 04:42:32  fumik
// Import of KIIsimulator
//
// Revision 1.4  2005/03/22 22:29:31  stiber
// Fixed reading of sparse matrices (so it actually works for them
// now).
//
// Revision 1.3  2005/03/08 19:56:38  stiber
// Modified comments for Doxygen.
//
// Revision 1.2  2005/03/07 16:19:18  stiber
// Immproved code that extracts XML element attributes with const char* variables
// instead of string objects. This allows checking for NULL pointers. Also replaced
// calls to the generic Attribute() method with QueryIntAttribute() and
// QueryDoubleAttribute(). Debugged matrix loading code, especially creation and
// loading of SparseMatrices, taking advantage of new SparseMatrix constructors.
//
// Revision 1.1  2005/02/18 13:42:28  stiber
// Initial revision
//
//

#include <cmath>
#include <cstdlib>
#include <sstream>
#include <iostream>

#include "mex.h"
#include "tinyxml.h"


// #include "SourceVersions.h"

// static VersionInfo version("$Id: readMatrix.cpp,v 1.1.1.1 2006/11/18 04:42:32 fumik Exp $");



using namespace std;

// Read a row of a sparse Matrix
void readSparseRow(TiXmlElement* rowElement, BGFLOAT* arrayData, int rows)
{
  int rowNum;
  if (rowElement->QueryIntAttribute("number", &rowNum)!=TIXML_SUCCESS)
    mexErrMsgTxt("Attempt to read SparseMatrix row without a number");

  // Iterate through the entries, inserting them into arrayData
  for (TiXmlElement* child = rowElement->FirstChildElement("Entry"); 
       child != NULL; child=child->NextSiblingElement("Entry")) {
    int colNum;
    BGFLOAT val;
    if (child->QueryIntAttribute("number", &colNum)!=TIXML_SUCCESS)
      mexErrMsgTxt("Attempt to read SparseMatrix Entry without a number");
    if (child->QueryDoubleAttribute("value", &val)!=TIXML_SUCCESS)
      mexErrMsgTxt("Attempt to read SparseMatrix Entry without a value");
    arrayData[rowNum+colNum*rows] = val;
  }
}


// Read a Matrix from XML
mxArray* ReadMatrix(TiXmlElement* matrix)
{
  int rows, columns;
  BGFLOAT multiplier;
  string init, type;

  // Get matrix type
  const char* typeCStr = matrix->Attribute("type");
  if (typeCStr == NULL) {
    mexErrMsgTxt("Malformed Matrix in state file (no type).");
    return NULL;
  }
  type = typeCStr;

  // Get dimensions
  if (matrix->QueryIntAttribute("columns", &columns)!=TIXML_SUCCESS) {
    mexErrMsgTxt("Malformed Matrix in state file (no columns).");
    return NULL;
  }
  if (matrix->QueryIntAttribute("rows", &rows)!=TIXML_SUCCESS) {
    mexErrMsgTxt("Malformed Matrix in state file (no rows).");
    return NULL;
  }

  // Get initialization information
  if (matrix->QueryDoubleAttribute("multiplier", &multiplier)!=TIXML_SUCCESS) {
    multiplier = 1.0;
  }
  const char* initCStr = matrix->Attribute("init");
  if (initCStr == NULL)
    init = "none";
  else
    init = initCStr;

  // Create the mxArray
  mxArray *theArray = mxCreateDoubleMatrix(rows, columns, mxREAL);
  BGFLOAT *arrayData = mxGetPr(theArray);

  // No init -- get the Matrix contents
  if (init == "none") {
    if (type == "sparse") {
      for (TiXmlElement* rowElement = matrix->FirstChildElement("Row");
	   rowElement != NULL;
	   rowElement = rowElement->NextSiblingElement("Row"))
	readSparseRow(rowElement, arrayData, rows);
    } else {
      TiXmlHandle matHandle(matrix);
      TiXmlText* valuesNode = matHandle.FirstChild().Text();
      if (valuesNode == NULL) {
	mexErrMsgTxt("No contents found for Matrix.");
	return NULL;
      }
      string values = valuesNode->Value();
      istringstream valStream(values);

      for (int i=0; i<rows; i++)
	for (int j=0; j<columns; j++)
	  valStream >> arrayData[i+j*rows];
    }
  } else {      // Initialization via multiplier
    if (type == "diag") {       // diagonal matrix with constant values
      for (int i=0; i<rows; i++)
	for (int j=0; j<columns; j++) {          
	  arrayData[i+j*rows] = 0.0;    // Non-diagonal elements are zero
          if (i == j)
            arrayData[i+j*rows] = multiplier;
        }
    } else if ((type == "complete") || (type == "sparse")) {
      // complete or sparse matrix with constant values
      for (int i=0; i<rows; i++)     
	for (int j=0; j<columns; j++)
          arrayData[i+j*rows] = multiplier;
    } else {
      mexErrMsgTxt("Unrecognized matrix type.");
      return NULL;
    }
  }

  return theArray;
}






void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{ 
  if (nrhs != 2) {
    mexErrMsgTxt("wrong number of args (must be 2)");
    return;
  }

  // Get file name
  int buflen = (mxGetM(prhs[0]) * mxGetN(prhs[0]) * sizeof(mxChar)) + 1;
  char stateFileName[buflen];
  if (mxGetString(prhs[0], stateFileName, buflen)) {
    mexErrMsgTxt("Unable to get file name from MATLAB variable.");
    return;
  }

  // Get Matrix name
  buflen = (mxGetM(prhs[1]) * mxGetN(prhs[1]) * sizeof(mxChar)) + 1;
  char matrixName[buflen];
  if (mxGetString(prhs[1], matrixName, buflen)) {
    mexErrMsgTxt("Unable to get matrix name from MATLAB variable.");
    return;
  }

  // Open file and move to state information
  TiXmlDocument simDoc(stateFileName);
  if (!simDoc.LoadFile()) {
    mexErrMsgTxt("Unable to open file.");
    return;
  }
  TiXmlElement* simState = NULL;
  if ((simState = simDoc.FirstChildElement("SimState"))==NULL) {
    mexErrMsgTxt("Could not find <SimState> in file.");
    return;
  }

  // Loop until we've found the Matrix with the right name
  bool matrixFound = false; 
  for (TiXmlElement* matrix = simState->FirstChildElement("Matrix");
       (matrix != NULL) && !matrixFound; 
       matrix=matrix->NextSiblingElement("Matrix")) {
    string name;
    const char* nameCStr = matrix->Attribute("name");
    if (nameCStr == NULL)
      name = "";
    else
      name = nameCStr;
    if (name == matrixName) {
      plhs[0] = ReadMatrix(matrix);
      matrixFound = true;
    }
  }

  // Not found; return empty mxArray
  if (!matrixFound)
    plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
}

