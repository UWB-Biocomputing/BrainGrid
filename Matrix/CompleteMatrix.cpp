/*!
  @file CompleteMatrix.cpp
  @brief An efficient implementation of a dynamically-allocated 2D array.
  @author Michael Stiber
  @date $Date: 2006/11/22 07:07:34 $
  @version $Revision: 1.2 $
*/

// CompleteMatrix.cpp 2D Matrix with all elements present
//
// An efficient implementation of a dynamically-allocated 2D
// array. Self-allocating and de-allocating.

// Written December 2004 by Michael Stiber

// $Log: CompleteMatrix.cpp,v $
// Revision 1.2  2006/11/22 07:07:34  fumik
// DCT growth model first check in
//
// Revision 1.1.1.1  2006/11/18 04:42:31  fumik
// Import of KIIsimulator
//
// Revision 1.4  2005/03/08 19:54:36  stiber
// Modified comments for Doxygen.
//
// Revision 1.3  2005/02/18 13:38:53  stiber
// Added SourceVersions support.
//
// Revision 1.2  2005/02/09 18:34:39  stiber
// "Completely" debugged implementation.
//
// Revision 1.1  2004/12/06 20:03:05  stiber
// Initial revision
//


#include <iostream>
#include <sstream>

#include "KIIexceptions.h"
#include "CompleteMatrix.h"

#include "SourceVersions.h"

static VersionInfo version("$Id: CompleteMatrix.cpp,v 1.2 2006/11/22 07:07:34 fumik Exp $");

// Create a complete 2D Matrix
CompleteMatrix::CompleteMatrix(string t, string i, int r,
			       int c, FLOAT m, string values)
  : Matrix(t, i, r, c, m), theMatrix(NULL)
{
#ifdef MDEBUG
  cerr << "Creating CompleteMatrix, size: ";
#endif

  // Bail out if we're being asked to create nonsense
  if (!((rows > 0) && (columns > 0)))
    throw KII_invalid_argument("CompleteMatrix::CompleteMatrix(): Asked to create zero-size");

  // We're a 2D Matrix, even if only one row or column
  dimensions = 2;

#ifdef MDEBUG
  cerr << rows << "X" << columns << ":" << endl;
#endif

  // Allocate storage
  alloc(rows, columns);

  if (values != "") {     // Initialize from the text string
    istringstream valStream(values);
    if (type == "diag") {       // diagonal matrix with values given
      for (int i=0; i<rows; i++)
	for (int j=0; j<columns; j++) {
	  theMatrix[i][j] = 0.0;    // Non-diagonal elements are zero
	  if (i == j) {
	    valStream >> theMatrix[i][j];
	    theMatrix[i][j] *= multiplier;
	  }
	}
    } else if (type == "complete") { // complete matrix with values given
      for (int i=0; i<rows; i++)
	for (int j=0; j<columns; j++) {
	  valStream >> theMatrix[i][j];
	  theMatrix[i][j] *= multiplier;
	}
    } else {
      clear();
      throw KII_invalid_argument("Illegal type for CompleteMatrix with 'none' init: " + type);
    }
  } else if (init == "const") {
    if (type == "diag") {       // diagonal matrix with constant values
      for (int i=0; i<rows; i++)
	for (int j=0; j<columns; j++) {
	  theMatrix[i][j] = 0.0;    // Non-diagonal elements are zero
	  if (i == j)
	    theMatrix[i][j] = multiplier;
	}
    } else if (type == "complete") { // complete matrix with constant values
      for (int i=0; i<rows; i++)
	for (int j=0; j<columns; j++)
	  theMatrix[i][j] = multiplier;
    } else {
      clear();
      throw KII_invalid_argument("Illegal type for CompleteMatrix with 'none' init: " + type);
    }
  }
  //  else if (init == "random")
#ifdef MDEBUG
    cerr << "\tInitialized " << type << " matrix" << endl;
#endif
}


// "Copy Constructor"
CompleteMatrix::CompleteMatrix(const CompleteMatrix& oldM) : theMatrix(NULL)
{
#ifdef MDEBUG
  cerr << "CompleteMatrix copy constructor:" << endl;
#endif
  copy(oldM);
}

// Destructor
CompleteMatrix::~CompleteMatrix()
{
#ifdef MDEBUG
  cerr << "Destroying CompleteMatrix" << endl;
#endif
  clear();
}


// Assignment operator
CompleteMatrix& CompleteMatrix::operator=(const CompleteMatrix& rhs)
{
  if (&rhs == this)
    return *this;

#ifdef MDEBUG
  cerr << "CompleteMatrix::operator=" << endl;
#endif

  clear();
#ifdef MDEBUG
  cerr << "\t\tclear() complete, ready to copy." << endl;
#endif
  copy(rhs);
#ifdef MDEBUG
  cerr << "\t\tcopy() complete; returning by reference." << endl;
#endif
  return *this;
}

// Clear out storage
void CompleteMatrix::clear(void)
{
#ifdef MDEBUG
  cerr << "\tclearing " << rows << "X" << columns << " CompleteMatrix...";
#endif

  if (theMatrix != NULL) {
    for (int i=0; i<rows; i++)
      if (theMatrix[i] != NULL) {
	delete [] theMatrix[i];
	theMatrix[i] = NULL;
      }
    delete [] theMatrix;
    theMatrix = NULL;
  }
#ifdef MDEBUG
  cerr << "done." << endl;
#endif
}


// Copy matrix to this one
void CompleteMatrix::copy(const CompleteMatrix& source)
{
#ifdef MDEBUG
  cerr << "\tcopying " << source.rows << "X" << source.columns
       << " CompleteMatrix...";
#endif

  SetAttributes(source.type, source.init, source.rows,
		source.columns, source.multiplier, source.dimensions);

  alloc(rows, columns);

  for (int i=0; i<rows; i++)
    for (int j=0; j<columns; j++)
      theMatrix[i][j] = source.theMatrix[i][j];
#ifdef MDEBUG
  cerr << "\t\tdone." << endl;
#endif
}


// Allocate internal storage
void CompleteMatrix::alloc(int rows, int columns)
{
  if (theMatrix != NULL)
    throw KII_exception("Attempt to allocate storage for non-cleared Matrix");

  if ((theMatrix = new FLOAT*[rows]) == NULL)
    throw KII_bad_alloc("Failed allocating storage to copy Matrix.");

  for (int i=0; i<rows; i++)
    if ((theMatrix[i] = new FLOAT[columns]) == NULL)
      throw KII_bad_alloc("Failed allocating storage to copy Matrix.");
#ifdef MDEBUG
  cerr << "\tStorage allocated for "<< rows << "X" << columns << " Matrix." << endl;
#endif

}


// Polymorphic output
void CompleteMatrix::Print(ostream& os) const
{
  for (int i=0; i<rows; i++) {
    for (int j=0; j<columns; j++)
      os << theMatrix[i][j] << " ";
    os << endl;
  }
}

// convert Matrix to XML string
string CompleteMatrix::toXML(string name) const
{
  stringstream os;

  os << "<Matrix ";
  if (name != "")
    os << "name=\"" << name << "\" ";
  os << "type=\"complete\" rows=\"" << rows
     << "\" columns=\"" << columns
     << "\" multiplier=\"1.0\">" << endl;
  os << "   " << *this << endl;
  os << "</Matrix>";

  return os.str();
}


// Math operations. For efficiency's sake, these methods will be
// implemented as being "aware" of each other (i.e., using "friend"
// and including the other subclasses' headers).

const CompleteMatrix CompleteMatrix::operator+(const CompleteMatrix& rhs) const
{
  if ((rhs.rows != rows) || (rhs.columns != columns)) {
    throw KII_domain_error("Illegal matrix addition: dimension mismatch");
  }
  // Start with this
  CompleteMatrix result(*this);
  // Add in rhs
  for (int i=0; i<rows; i++)
    for (int j=0; j<columns; j++)
    result.theMatrix[i][j] += rhs.theMatrix[i][j];

  return result;
}


// Multiply the rhs into the current object
const CompleteMatrix CompleteMatrix::operator*(const CompleteMatrix& rhs) const
{
  throw KII_domain_error("CompleteMatrix product not yet implemented");
}

// Element-wise square root of a vector
const CompleteMatrix sqrt(const CompleteMatrix& m)
{
  // Start with vector
  CompleteMatrix result(m);

  for (int i=0; i<result.rows; i++)
	  for (int j=0; j < result.columns; j++)
		result.theMatrix[i][j] = sqrt(result.theMatrix[i][j]);

  return result;
}
