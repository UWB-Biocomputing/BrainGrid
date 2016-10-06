/*!
  @file MatrixFactory.cpp
  @brief Deserializes Matrices from XML
  @author Michael Stiber
  @date $Date: 2006/11/18 04:42:32 $
  @version $Revision: 1.1.1.1 $
*/

// MatrixFactory.cpp Create Matrix subclass objects from XML
//
// This class deserializes Matrix subclass objects from XML, creating
// the appropriate type of subclass object, based on the XML
// attributes.

// Written December 2004 by Michael Stiber

// $Log: MatrixFactory.cpp,v $
// Revision 1.1.1.1  2006/11/18 04:42:32  fumik
// Import of KIIsimulator
//
// Revision 1.6  2005/03/08 19:55:49  stiber
// Modified comments for Doxygen.
//
// Revision 1.5  2005/03/07 16:14:03  stiber
// Updated creation routines for new SparseMatrix constructors and to improve
// SparseMatrix creation logic.
//
// Revision 1.4  2005/02/22 19:03:23  stiber
// Modified CreateSparse to create sparse matrices from diagonal matrix
// types.
//
// Revision 1.3  2005/02/18 13:40:11  stiber
// Added SourceVersions support.
//
// Revision 1.2  2005/02/09 18:40:23  stiber
// "Competely debugged".
//
// Revision 1.1  2004/12/06 20:03:05  stiber
// Initial revision
//


#include <iostream>
#include "MatrixFactory.h"

#include "SourceVersions.h"

static VersionInfo version("$Id: MatrixFactory.cpp,v 1.1.1.1 2006/11/18 04:42:32 fumik Exp $");

// Get Matrix attributes
// Inputs:
//   matElement: pointer to the Matrix TiXmlElement
// Outputs:
//   type:  "diag" (diagonal matrices), "complete" (all values
//          specified), or "sparse". Required.
//   init:  "none" (initialization data explicitly given, default), 
//          "const" (initialized to muliplier, if present, else 1.0),
//          "random" (random values in the range [0,1]),
//          or "implementation" (must be initialized by caller, not
//          creator). Optional (defaults to "none")
//   rows:  number of matrix rows. Required.
//   columns: number of matrix columns. Required.
//   multiplier: constant multiplier used in initialization, default
//               1.0. Optional (defaults to 1.0).
void MatrixFactory::GetAttributes(TiXmlElement* matElement,
				  string& type, string& init,
				  int& rows, int& columns,
				  FLOAT& multiplier)
{
  const char* temp = NULL;

#ifdef MDEBUG
  cerr << "Getting attributes:" << endl;
#endif
  temp = matElement->Attribute("type");
  if (temp != NULL)
    type = temp;
  else
    type = "undefined";
  if ((type != "diag") && (type != "complete") && (type != "sparse"))
    throw KII_invalid_argument("Illegal matrix type: " + type);
#ifdef MDEBUG
  cerr << "\ttype=" << type << ", ";
#endif

  if (matElement->QueryIntAttribute("rows", &rows)!=TIXML_SUCCESS)
    throw KII_invalid_argument("Number of rows not specified for Matrix.");
#ifdef MDEBUG
  cerr << "\trows=" << rows << ", ";
#endif

  if (matElement->QueryIntAttribute("columns", &columns)!=TIXML_SUCCESS)
    throw KII_invalid_argument("Number of columns not specified for Matrix.");
#ifdef MDEBUG
  cerr << "\tcolumns=" << columns << ", ";
#endif

  if (matElement->QueryFLOATAttribute("multiplier", &multiplier)!=TIXML_SUCCESS) {
    multiplier = 1.0;
  }
#ifdef MDEBUG
  cerr << "\tmultiplier=" << multiplier << ", ";
#endif

  temp = matElement->Attribute("init");
  if (temp != NULL)
    init = temp;
  else
    init = "none";
#ifdef MDEBUG
  cerr << "\tinit=" << init << endl;
#endif
}

// This function creates a Matrix subclass element from the given
// tinyxml Element and its children. The class created is determined
// from the attributes of the Element
//
// Input:
//   matElement: tinyxml DOM node containing a Matrix element
// Postconditions"
//   If no problems, correct Matrix subclass object created and
//   initialized.
// Returns:
//   Pointer to created Matrix (NULL if failure).
Matrix* MatrixFactory::CreateMatrix(TiXmlElement* matElement) 
{
  string type;
  string init;
  int rows, columns;
  FLOAT multiplier;
  Matrix* theMatrix = NULL;
  TiXmlHandle matHandle(matElement);

  GetAttributes(matElement, type, init, rows, columns, multiplier);

#ifdef MDEBUG
  cerr << "Creating Matrix with attributes: " << type << ", " << init
       << ", " << rows << "X" << columns << ", " << multiplier << endl;
#endif

  if (init == "implementation")
    throw KII_invalid_argument("MatrixFactory cannot create implementation-dependent Matrices; client program must perform creation.");

  if (type == "complete") {
    string values;
    // Get the Text node that contains the matrix values, if needed
    if (init == "none") {
      TiXmlText* valuesNode = matHandle.FirstChild().Text();
      if (valuesNode == NULL)
	throw KII_invalid_argument("Contents not specified for Matrix with init='none'.");
      values = valuesNode->Value();
#ifdef MDEBUG
      cerr << "\tData present for initialization: " << values << endl;
#endif
    }
    if ((rows > 1) && (columns > 1))   // Create a 2D Matrix
      theMatrix = new CompleteMatrix(type, init, rows, columns,
				     multiplier, values);
    else                               // Create a 1D Matrix
      theMatrix = new VectorMatrix(type, init, rows, columns,
				   multiplier, values);
  } else if (type == "diag") {   // Implement diagonal matrices as sparse
    if (init == "none") {          // a string of values is present & passed
      TiXmlText* valuesNode = matHandle.FirstChild().Text();
      if (valuesNode == NULL)
	throw KII_invalid_argument("Contents not specified for Sparse Matrix with init='none'.");
      const char* values = valuesNode->Value();
#ifdef MDEBUG
      cerr << "\tData present for initialization: " << values << endl;
#endif
      theMatrix = new SparseMatrix(rows, columns, multiplier, values);
    } else if (init == "const") {   // No string of values or XML row data
      theMatrix = new SparseMatrix(rows, columns, multiplier);
    } else
      throw KII_invalid_argument("Invalid init for sparse matrix");
  } else if (type == "sparse") {
    if (init == "none")             // a sequence of row data nodes is present & passed
      theMatrix = new SparseMatrix(rows, columns, multiplier, matElement);
    else if (init == "const") {     // No row data
      if (multiplier == 0.0)
	theMatrix = new SparseMatrix(rows, columns);
      else
	throw KII_invalid_argument("A sparse matrix can only be initialized to zero with const XML init");
    } else
      throw KII_invalid_argument("A sparse matrix can only be initialized to zero with const XML init");
  } else
    throw KII_invalid_argument("Illegal Matrix type");

  return theMatrix;
}

// This function creates a VectorMatrix from the given tinyxml Element
// and its children.
//
// Input:
//   matElement: tinyxml DOM node containing a Matrix element
// Postconditions"
//   If no problems, VectorMatrix object created and
//   initialized.
// Returns:
//   VectorMatrix object (will be empty if some failure occurs).
VectorMatrix MatrixFactory::CreateVector(TiXmlElement* matElement)
{
  string type;
  string init;
  int rows, columns;
  FLOAT multiplier;
  string values;
  TiXmlHandle matHandle(matElement);

  GetAttributes(matElement, type, init, rows, columns, multiplier);

#ifdef VDEBUG
  cerr << "Creating Vector with attributes: " << type << ", " << init
       << ", " << rows << "X" << columns << ", " << multiplier << endl;
#endif

  // Get the Text node that contains the matrix values, if needed
  if (init == "none") {
    TiXmlText* valuesNode = matHandle.FirstChild().Text();
    if (valuesNode == NULL)
      throw KII_invalid_argument("Contents not specified for Vector with init='none'.");

    values = valuesNode->Value();
#ifdef VDEBUG
    cerr << "\tData present for initialization: " << values << endl;
#endif
  } else if (init == "implementation")
    throw KII_invalid_argument("MatrixFactory cannot create implementation-dependent Matrices; client program must perform creation");

  if (type == "sparse")
    throw KII_invalid_argument("Sparse matrix requested in XML but CreateVector called");

  if ((type == "complete") || (type == "diag")) {
    if ((rows > 1) && (columns > 1)) // Create a 2D Matrix
      throw KII_domain_error("Cannot create Vector with more than one dimension.");
    else                               // Create a 1D Matrix
       return VectorMatrix(type, init, rows, columns, multiplier, values);
  } else if (type == "sparse")
    throw KII_invalid_argument("No such thing as sparse Vectors");
  else
    throw KII_invalid_argument("Illegal Vector type");

  return VectorMatrix();
}

// This function creates a CompleteMatrix from the given tinyxml Element
// and its children. 
//
// Input:
//   matElement: tinyxml DOM node containing a Matrix element
// Postconditions"
//   If no problems, CompleteMatrix subclass object created and
//   initialized.
// Returns:
//   CompleteMatrix object (will be empty if some failure occurs).
CompleteMatrix MatrixFactory::CreateComplete(TiXmlElement* matElement)
{
  string type;
  string init;
  int rows, columns;
  FLOAT multiplier;
  string values;
  TiXmlHandle matHandle(matElement);

  GetAttributes(matElement, type, init, rows, columns, multiplier);

#ifdef MDEBUG
  cerr << "Creating Matrix with attributes: " << type << ", " << init
       << ", " << rows << "X" << columns << ", " << multiplier << endl;
#endif

  // Get the Text node that contains the matrix values, if needed
  if (init == "none") {
    TiXmlText* valuesNode = matHandle.FirstChild().Text();
    if (valuesNode == NULL)
      throw KII_invalid_argument("Contents not specified for Matrix with init='none'.");

    values = valuesNode->Value();
#ifdef MDEBUG
    cerr << "\tData present for initialization: " << values << endl;
#endif
  } else if (init == "implementation")
    throw KII_invalid_argument("MatrixFactory cannot create implementation-dependent Matrices; client program must perform creation.");

  if (type == "sparse")
    throw KII_invalid_argument("Sparse matrix requested by XML but CreateComplete called");

  if ((type == "complete") || (type == "diag"))
    return CompleteMatrix(type, init, rows, columns, multiplier, values);
  else if (type == "sparse")
    throw KII_invalid_argument("No such thing as sparse CompleteMatrices");
  else
    throw KII_invalid_argument("Illegal Vector type");

  return CompleteMatrix();
}

/*
  @method CreateSparse
  @discussion Create a SparseMatrix, based
  on the XML attributes. The object is returned by value.
  @throws KII_invalid_argument
  @param matElement pointer to Matrix XML element
  @result The SparseMatrix object.
*/
SparseMatrix MatrixFactory::CreateSparse(TiXmlElement* matElement)
{
  string type;
  string init;
  int rows, columns;
  FLOAT multiplier;
  TiXmlHandle matHandle(matElement);

  GetAttributes(matElement, type, init, rows, columns, multiplier);

#ifdef MDEBUG
  cerr << "Creating SparseMatrix with attributes: " << type << ", " << init
       << ", " << rows << "X" << columns << ", " << multiplier << endl;
#endif

  if (type == "diag") {
    if (init == "none") {          // a string of values is present & passed
      TiXmlText* valuesNode = matHandle.FirstChild().Text();
      if (valuesNode == NULL)
	throw KII_invalid_argument("Contents not specified for Sparese Matrix with init='none'.");
      const char* values = valuesNode->Value();
#ifdef MDEBUG
      cerr << "\tData present for initialization: " << values << endl;
#endif
      return SparseMatrix(rows, columns, multiplier, values);
    } else if (init == "const") {   // No string of values or XML row data
      if (multiplier == 0.0)
	return SparseMatrix(rows, columns);
      else
	throw KII_invalid_argument("A sparse matrix can only be initialized to zero with const XML init");
    } else
      throw KII_invalid_argument("Invalid init for sparse matrix");
  } else if (type == "sparse") {
    if (init == "none")             // a sequence of row data nodes is present & passed
      return SparseMatrix(rows, columns, multiplier, matElement);
    else if (init == "const") {     // No row data
      if (multiplier == 0.0)
	return SparseMatrix(rows, columns);
      else
	throw KII_invalid_argument("A sparse matrix can only be initialized to zero with const XML init");
    } else
      throw KII_invalid_argument("A sparse matrix can only be initialized to zero with const XML init");
  }

  // If we get here, then something is really wrong
  throw KII_invalid_argument("Invalid type specified for sparse matrix.");

}


