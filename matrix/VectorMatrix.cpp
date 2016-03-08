/**
 @file VectorMatrix.cpp
 @brief  An efficient implementation of a dynamically-allocated 1D array
 @author Michael Stiber
 @date January 2016
 @version 2
 */

// VectorMatrix.cpp 1D Matrix with all elements present
//
// An efficient implementation of a dynamically-allocated 1D
// array. Self-allocating and de-allocating.

// Written December 2004 by Michael Stiber

// $Log: VectorMatrix.cpp,v $
// Revision 1.1.1.1  2006/11/18 04:42:32  fumik
// Import of KIIsimulator
//
// Revision 1.5  2005/03/08 19:56:25  stiber
// Modified comments for Doxygen.
//
// Revision 1.4  2005/02/18 13:41:42  stiber
// Added SourceVersions support.
//
// Revision 1.3  2005/02/17 15:26:28  stiber
// Minor modifications of comments because of support for Sparse Matrices
// elsewhere.
//
// Revision 1.2  2005/02/09 18:45:26  stiber
// "Completely debugged".
//
// Revision 1.1  2004/12/06 20:03:05  stiber
// Initial revision
//


#include <iostream>
#include <sstream>

#include "Global.h"
#include "VectorMatrix.h"

// Classwide normal RNG
Norm VectorMatrix::nRng;

/*
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
VectorMatrix::VectorMatrix(string t, string i, int r, int c, BGFLOAT m, string values) :
	Matrix(t, i, r, c, m), theVector(NULL) {
	DEBUG_VECTOR(cerr << "Creating VectorMatrix, size: ";)

	// Bail out if we're being asked to create nonsense
	if (!((rows == 1) || (columns == 1)) || (rows == 0) || (columns == 0))
		throw Matrix_invalid_argument("VectorMatrix: Asked to create 2D or zero-size");

	// We're a 1D Matrix
	dimensions = 1;
	size = (rows > columns) ? rows : columns;

	DEBUG_VECTOR(cerr << rows << "X" << columns << ":" << endl;)

	alloc(size);

	if (values != "") { // Initialize from the text string
		istringstream valStream(values);
		if (type == "complete") { // complete matrix with values given
			for (int i = 0; i < size; i++) {
				valStream >> theVector[i];
				theVector[i] *= multiplier;
			}
		} else {
			clear();
			throw Matrix_invalid_argument("Illegal type for VectorMatrix with 'none' init: " + type);
		}
	} else if (init == "const") {
		if (type == "complete") { // complete matrix with constant values
			for (int i = 0; i < size; i++)
				theVector[i] = multiplier;
		} else {
			clear();
			throw Matrix_invalid_argument("Illegal type for VectorMatrix with 'none' init: " + type);
		}
	} else if (init == "random") {
		// Initialize with normally distributed random numbers with zero
		// mean and unit variance
		for (int i = 0; i < size; i++) {
			theVector[i] = nRng();
		}
	} else {
		clear();
		throw Matrix_invalid_argument("Illegal initialization for VectorMatrix: " + init);
	}
	DEBUG_VECTOR(cerr << "\tInitialized " << type << " vector to " << *this << endl;)
}

// Copy constructor
VectorMatrix::VectorMatrix(const VectorMatrix& oldV) :
	theVector(NULL) {
	copy(oldV);
}

// Assignment operator: set elements of vector to constant
const VectorMatrix& VectorMatrix::operator=(BGFLOAT c) {
	for (int i = 0; i < size; i++)
		theVector[i] = c;

	return *this;
}

// Assignment operator
const VectorMatrix& VectorMatrix::operator=(const VectorMatrix& rhs) {
	if (&rhs == this)
		return *this;

	clear();
	copy(rhs);
	return *this;
}

// Destructor
VectorMatrix::~VectorMatrix() {
	clear();
}

// Clear out storage
void VectorMatrix::clear(void) {
	if (theVector != NULL) {
		delete[] theVector;
		theVector = NULL;
	}
}

// Copy vector to this one
void VectorMatrix::copy(const VectorMatrix& source) {
	size = source.size;
	SetAttributes(source.type, source.init, source.rows, source.columns, source.multiplier,
			source.dimensions);

	alloc(size);

	for (int i = 0; i < size; i++)
		theVector[i] = source.theVector[i];
}

// Allocate internal storage
void VectorMatrix::alloc(int size) {
	if (theVector != NULL)
		throw MatrixException("Attempt to allocate storage for non-cleared Vector.");

	if ((theVector = new BGFLOAT[size]) == NULL) {
		throw Matrix_bad_alloc("Failed allocating storage of Vector copy.");
	}

	DEBUG_VECTOR(cerr << "\tStorage allocated for "<< size << " element Vector." << endl;)

}

// Polymorphic output
void VectorMatrix::Print(ostream& os) const {
	for (int i = 0; i < size; i++)
		os << theVector[i] << " ";
}

// convert vector to XML string
string VectorMatrix::toXML(string name) const {
	stringstream os;

	os << "<Matrix ";
	if (name != "")
		os << "name=\"" << name << "\" ";
	os << "type=\"complete\" rows=\"1\" columns=\"" << size << "\" multiplier=\"1.0\">" << endl;
	os << "   " << *this << endl;
	os << "</Matrix>";

	return os.str();
}

// The math operations

const VectorMatrix VectorMatrix::operator+(const VectorMatrix& rhs) const {
	if (rhs.size != size) {
		throw Matrix_domain_error("Illegal vector sum. Vectors must be equal length.");
	}

	// Start with this
	VectorMatrix result(*this);

	// Add in rhs
	for (int i = 0; i < size; i++)
		result.theVector[i] += rhs.theVector[i];

	return result;
}

// Vector plus a constant
const VectorMatrix VectorMatrix::operator+(BGFLOAT c) const {
	// Start with this
	VectorMatrix result(*this);

	for (int i = 0; i < size; i++)
		result.theVector[i] += c;

	return result;
}

// There are two possible products. This is an inner product.
BGFLOAT VectorMatrix::operator*(const VectorMatrix& rhs) const {
	if (rhs.size != size) {
		throw Matrix_domain_error("Illegal vector inner product. Vectors must be equal length.");
	}

	// the result is scalar
	BGFLOAT result;

	result = theVector[0] * rhs.theVector[0];

	for (int i = 1; i < size; i++)
		result += theVector[i] * rhs.theVector[i];

	return result;
}

// Vector times a Complete matrix
const VectorMatrix VectorMatrix::operator*(const CompleteMatrix& rhs) const {
	if (rhs.rows != size) {
		throw Matrix_domain_error(
				"Illegal vector/matrix product. Rows of matrix must equal vector size.");
	}

	// the result is a vector the same size as rhs columns
	VectorMatrix result("complete", "const", 1, rhs.columns, 0.0, "");

	for (int i = 0; i < result.size; i++)
		// Compute each element of the result
		for (int j = 0; j < size; j++)
			result.theVector[i] += theVector[j] * rhs.theMatrix[j][i];

	return result;
}

const VectorMatrix VectorMatrix::ArrayMultiply(const VectorMatrix& rhs) const {
	if (rhs.size != size) {
		throw Matrix_domain_error("Illegal array product. Vectors must be equal length.");
	}

	// Start with this
	VectorMatrix result(*this);

	// Multiply elements of rhs
	for (int i = 0; i < size; i++)
		result.theVector[i] *= rhs.theVector[i];

	return result;
}

// Vector times a constant
const VectorMatrix VectorMatrix::operator*(BGFLOAT c) const {
	// Start with this
	VectorMatrix result(*this);

	for (int i = 0; i < size; i++)
		result.theVector[i] *= c;

	return result;
}

// Vector divided by a constant
const VectorMatrix VectorMatrix::operator/(BGFLOAT c) const {
	// Start with this
	VectorMatrix result(*this);

	for (int i = 0; i < size; i++)
		result.theVector[i] /= c;

	return result;
}

// Constant minus a vector
const VectorMatrix operator-(BGFLOAT c, const VectorMatrix& v) {
	// Start with vector
	VectorMatrix result(v);

	for (int i = 0; i < result.size; i++)
		result.theVector[i] = c - result.theVector[i];

	return result;
}

// Constant divided by a vector
const VectorMatrix operator/(BGFLOAT c, const VectorMatrix& v) {
	// Start with vector
	VectorMatrix result(v);

	for (int i = 0; i < result.size; i++)
		result.theVector[i] = c / result.theVector[i];

	return result;
}

// Limit values of a vector
const VectorMatrix VectorMatrix::Limit(BGFLOAT low, BGFLOAT high) const {
	// Start with this
	VectorMatrix result(*this);

	for (int i = 0; i < size; i++) {
		if (result.theVector[i] < low)
			result.theVector[i] = low;
		if (result.theVector[i] > high)
			result.theVector[i] = high;
	}

	return result;
}

// Find minimum value
BGFLOAT VectorMatrix::Min(void) const {
	BGFLOAT min = theVector[0];

	for (int i = 1; i < size; i++)
		if (theVector[i] < min)
			min = theVector[i];

	return min;
}

// Find maximum value
BGFLOAT VectorMatrix::Max(void) const {
	BGFLOAT max = theVector[0];

	for (int i = 1; i < size; i++)
		if (theVector[i] > max)
			max = theVector[i];

	return max;
}

// Element-wise square root of a vector
const VectorMatrix sqrt(const VectorMatrix& v) {
	// Start with vector
	VectorMatrix result(v);

	for (int i = 0; i < result.size; i++)
		result.theVector[i] = sqrt(result.theVector[i]);

	return result;
}

// Element-wise e^x for vector
const VectorMatrix exp(const VectorMatrix& v) {
	// Start with vector
	VectorMatrix result(v);

	for (int i = 0; i < result.size; i++)
		result.theVector[i] = exp(result.theVector[i]);

	return result;
}

const VectorMatrix& VectorMatrix::operator+=(const VectorMatrix& rhs) {
	if (rhs.size != size) {
		throw Matrix_domain_error("Illegal vector sum. Vectors must be equal length.");
	}

	// Add in rhs
	for (int i = 0; i < size; i++)
		theVector[i] += rhs.theVector[i];

	return *this;
}

