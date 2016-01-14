/**
 @file SparseMatrix.cpp
 @brief An efficient implementation of a dynamically-allocated 2D sparse array.
 @author Michael Stiber
 @date January 2016
 @version 2
 */

// SparseMatrix.cpp 2D Sparse Matrix
//
// An efficient implementation of a dynamically-allocated sparse 2D
// array. Self-allocating and de-allocating.

// Written February 2005 by Michael Stiber

// $Log: SparseMatrix.cpp,v $
// Revision 1.1.1.1  2006/11/18 04:42:32  fumik
// Import of KIIsimulator
//
// Revision 1.8  2006/10/09 15:11:30  stiber
// Added unary minus operator.
//
// Revision 1.7  2005/03/31 12:46:58  stiber
// Fixed hash table overflow.
//
// Revision 1.6  2005/03/08 19:56:15  stiber
// Modified comments for Doxygen.
//
// Revision 1.5  2005/03/07 16:14:38  stiber
// A moderate amount of debugging and code improvement. Split constructor
// into multiple, simpler ones. Reorganized code for clearing, allocating,
// resizing, and copying SparseMatrices, along with copy constructor and
// assignment operator. Reworked and simplified logic for HashTable search
// operations. Added try...catch blocks to generate extra diagnostic output
// in the event of internal runtime errors. This class probably still needs
// work, but I am much more confident that it is working correctly.
//
// Revision 1.4  2005/02/22 20:03:26  stiber
// Now outputs correct XML for empty SparseMatrices.
//
// Revision 1.3  2005/02/18 13:41:35  stiber
// Added SourceVersions support.
//
// Revision 1.2  2005/02/17 15:25:34  stiber
// All basic functionality for KIIgrowth simulator working.
//
// Revision 1.1  2005/02/16 15:31:20  stiber
// Initial revision
//
//


#include <iostream>
#include <sstream>
#include <algorithm>

#include "Global.h"
#include "SparseMatrix.h"

extern bool debugSparseMatrix;

// hash table methods and static members
/*
 @class HashTable
 @brief Specialized hash table for pointers to Elements.
 
 Implemented using linear probing. Because of this choice, we cannot delete storage taken
 up by elements that get zeroed out. These elements can be re-used, of course. Consequently,
 a SparseMatrix never shrinks.
 */

/* A special Element to mark deleted table locations */
SparseMatrix::Element SparseMatrix::HashTable::deleted(-1, -1, -1.0);


/*
 @param s table capacity
 @param c number of columns in containing SparseMatrix
 @abstract resize hash table and initialize elements to NULL
 */
void SparseMatrix::HashTable::resize(int s, int c)
{
    // If nothing has changed, just clear the table to NULL
    if ((s == capacity) && (c == columns)) {
        DEBUG_SPARSE(cerr << "\t\t\tSM::HT::resize(): table capacity unchanged; filling with NULL." << endl;)
        fill(table.begin(), table.end(), static_cast<Element*>(NULL));
        size = 0;
    } else {
        DEBUG_SPARSE(cerr << "\t\t\tSM::HT::resize(): table capacity changed." << endl;)
        capacity = s;
        columns = c;
        table.resize(capacity, static_cast<Element*>(NULL));
        size = 0;
    }
    
	DEBUG_SPARSE(cerr << "\t\t\tAllocated " << capacity << " locations for hash table" << endl;)
}

/*
 @method clear
 @abstract removes all elements from the hash table
 */
void SparseMatrix::HashTable::clear(void)
{
    // Use resize to clean up the table
    resize(capacity, columns);
}

/*
 @method insert
 @discussion Inserts Element into the hash table using linear
 probing. If the Element is already in the table, an exception is
 thrown (the update method should be used to update a value).
 @param el pointer to the Element to insert
 @throws Matrix_bad_alloc
 @throws Matrix_invalid_argument
 */
void SparseMatrix::HashTable::insert(Element* el)
{
    // Don't overfill the table
    if (capacity == size+1)
        throw Matrix_invalid_argument("SparseMatrix hash table full");
    
    int start = hash(el);
    int loc = start;
    
	DEBUG_SPARSE(cerr << "\tInserting value " << el->value << " at ("
                 << el->row << ", " << el->column << ")" << endl;)
    
    // Find first location to insert (if Element isn't in the table)
    while ((table[loc] != &deleted) && (table[loc] != NULL)) {
        if (*table[loc] == *el)
            throw Matrix_invalid_argument("Attempt to insert duplicate Element into SparseMatrix hash table [1]");
        loc = (loc+1)%capacity;
        if (loc == start)
            throw Matrix_invalid_argument("Hashtable insert wraparound: unable to insert Element [1]");
    }
    
    // If that first location is a deleted entry, then we need to
    // continue searching to ensure the Element isn't in the table
    if (table[loc] == &deleted) {
        int loc2 = (loc+1)%capacity;
        while (table[loc2] != NULL) {
            if (loc2 == start)
                throw Matrix_invalid_argument("Hashtable insert wraparound: unable to insert Element [2]");
            loc2 = (loc2+1)%capacity;
            if (*table[loc2] == *el)
                throw Matrix_invalid_argument("Attempt to insert duplicate Element into SparseMatrix hash table [2]");
        }
    }
    
    // At this point, either table[loc] is deleted or NULL. Note that
    // loc2 was only used to ensure we aren't inserting duplicates;
    // table[loc] is the first available storage location.
    table[loc] = el;
    size++;
    
	DEBUG_SPARSE(cerr << "\tInserted at table location " << loc << endl;)
}


/*
 @method retrieve
 @discussion retrieves Element from the hash table using linear
 probing. If Element isn't in the table, returns NULL. Any
 zero-value elements found in the hash table are deleted along
 the way.
 @param r row coordinate of Element
 @param c column coordinate of Element
 @result pointer to Element (NULL if not in table)
 */
SparseMatrix::Element* SparseMatrix::HashTable::retrieve(int r, int c)
{
    int loc = hash(r, c);
    
    // Keep probing while we haven't found the Element (or finished the
    // probing). Also, eliminate any Elements found with zero value.
    while ((table[loc] != NULL) && (!table[loc]->is_at(r, c))) {
        if (table[loc]->value == 0.0) {
            theMatrix->remove_lists(table[loc]);
            table[loc] = &deleted;
            size--;
        }
        loc = (loc+1)%capacity;
    }
    
    // At this point, we either have the pointer to the Element or the
    // NULL pointer
    return table[loc];
}

/*
 @method update
 @discussion Updates the Element already in the table using linear
 probing. If Element isn't in the table, throws an exception
 (use insert to add new Elements to the table).
 @param r row coordinate of Element
 @param c column coordinate of Element
 @param v new value for Element
 @throws Matrix_invalid_argument
 */
void SparseMatrix::HashTable::update(int r, int c, BGFLOAT v)
{
    Element* el = retrieve(r, c);
    
    if (el == NULL)
        throw Matrix_invalid_argument("Attempt to update Element not in SparseMatrix hash table");
    el->value = v;
}




/*
 @method SparseMatrix
 @discussion Allocate storage and initialize attributes for a
 sparse matrix with explicit row data. The parameter e is used as a
 source of data for initializing the matrix (and must be the
 pointer to the Matrix element in the XML).
 @throws Matrix_bad_alloc
 @throws Matrix_invalid_argument
 @param r rows in Matrix
 @param c columns in Matrix
 @param m multiplier used for initialization
 @param e pointer to Matrix element in XML
 */
SparseMatrix::SparseMatrix(int r, int c, BGFLOAT m, TiXmlElement* e)
: Matrix("sparse", "none", r, c, m), theRows(NULL), theColumns(NULL),
theElements(MaxElements(r,c), c, this)
{
	DEBUG_SPARSE(cerr << "Creating SparseMatrix, size: ";)
    
    // Bail out if we're being asked to create nonsense
    if (!((rows > 0) && (columns > 0)))
        throw Matrix_invalid_argument("SparseMatrix::SparseMatrix(): Asked to create zero-size");
    
    // We're a 2D Matrix, even if only one row or column
    dimensions = 2;
    
	DEBUG_SPARSE(cerr << rows << "X" << columns << ":" << endl;)
    
    // Allocate storage for row and column lists (hash table already
    // allocated at initialization time; see initializer list, above).
    if ((theRows = new list<Element*>[rows]) == NULL)
        throw Matrix_bad_alloc("Failed allocating storage for SparseMatrix.");
    if ((theColumns = new list<Element*>[columns]) == NULL)
        throw Matrix_bad_alloc("Failed allocating storage for SparseMatrix.");
    
    // Initialize from the XML
    for (TiXmlElement* rowElement = e->FirstChildElement("Row");
         rowElement != NULL;
         rowElement = rowElement->NextSiblingElement("Row"))
        rowFromXML(rowElement);
    
	DEBUG_SPARSE(cerr << "\tInitialized " << type << " matrix" << endl;)
}


/*
 @method SparseMatrix
 @discussion Allocate storage and initialize attributes for a
 diagonal sparse matrix with explicit row data. The parameter v is
 used as a source of data for initializing the matrix (and must be
 a string of numbers equal to the number of rows or columns).
 @throws Matrix_bad_alloc
 @throws Matrix_invalid_argument
 @param r rows in Matrix
 @param c columns in Matrix
 @param m multiplier used for initialization
 @param v string of initialization values
 */
SparseMatrix::SparseMatrix(int r, int c, BGFLOAT m, const char* v)
: Matrix("sparse", "none", r, c, m), theRows(NULL), theColumns(NULL),
theElements(MaxElements(r,c), c, this)
{
	DEBUG_SPARSE(cerr << "\tCreating diagonal sparse matrix" << endl;)
    // Bail out if we're being asked to create nonsense
    if (!((rows > 0) && (columns > 0)))
        throw Matrix_invalid_argument("SparseMatrix::SparseMatrix(): Asked to create zero-size");
    
    // We're a 2D Matrix, even if only one row or column
    dimensions = 2;
    
	DEBUG_SPARSE(cerr << rows << "X" << columns << ":" << endl;)
    
    // Allocate storage for row and column lists (hash table already
    // allocated at initialization time; see initializer list, above).
    if ((theRows = new list<Element*>[rows]) == NULL)
        throw Matrix_bad_alloc("Failed allocating storage for SparseMatrix.");
    if ((theColumns = new list<Element*>[columns]) == NULL)
        throw Matrix_bad_alloc("Failed allocating storage for SparseMatrix.");
    
    if (multiplier == 0.0)  // If we're empty, then we're done.
        return;
    
    if (v != NULL) {     // Initialize from string of numeric data
        istringstream valStream(v);
        for (int i=0; i<rows; i++) {
            Element* el;
            BGFLOAT val;
            valStream >> val;
            if ((el = new Element(i, i, val*multiplier)) == NULL)
                throw Matrix_bad_alloc("Failed allocating storage for SparseMatrix.");
            theRows[i].push_back(el);
            theColumns[i].push_back(el);
            try {
                theElements.insert(el);
            } catch(Matrix_invalid_argument e) {
                cerr << "Failure during SparseMatrix string constructor: " << e.what() << endl;
                exit(-1);
            }
        }
    } else {             // No string of data; initialize from multipler only
        for (int i=0; i<rows; i++) {
            Element* el;
            if ((el = new Element(i, i, multiplier)) == NULL)
                throw Matrix_bad_alloc("Failed allocating storage for SparseMatrix.");
            theRows[i].push_back(el);
            theColumns[i].push_back(el);
            try {
                theElements.insert(el);
            } catch(Matrix_invalid_argument e) {
                cerr << "Failure during SparseMatrix multiplier only constructor: " << e.what() << endl;
                exit(-1);
            }
        }
    }
}


/*
 @method SparseMatrix
 @discussion Allocate storage and initialize attributes for an
 empty sparse matrix. This is also the default constructor.
 @throws Matrix_bad_alloc
 @throws Matrix_invalid_argument
 @param r rows in Matrix
 @param c columns in Matrix
 */
SparseMatrix::SparseMatrix(int r, int c)
: Matrix("sparse", "none", r, c, 0.0), theRows(NULL), theColumns(NULL),
theElements(MaxElements(r,c), c, this)
{
	DEBUG_SPARSE(cerr << "\tCreating empty sparse matrix: ";)
    // Bail out if we're being asked to create nonsense
    if (!((rows > 0) && (columns > 0)))
        throw Matrix_invalid_argument("SparseMatrix::SparseMatrix(): Asked to create zero-size");
    
    // We're a 2D Matrix, even if only one row or column
    dimensions = 2;
    
	DEBUG_SPARSE(cerr << rows << "X" << columns << ":" << endl;)
    
    // Allocate storage for row and column lists (hash table already
    // allocated at initialization time; see initializer list, above).
    if ((theRows = new list<Element*>[rows]) == NULL)
        throw Matrix_bad_alloc("Failed allocating storage for SparseMatrix.");
    if ((theColumns = new list<Element*>[columns]) == NULL)
        throw Matrix_bad_alloc("Failed allocating storage for SparseMatrix.");
    
    // And that's all, folks!
}




// Copy Constructor
SparseMatrix::SparseMatrix(const SparseMatrix& oldM)
: Matrix("sparse", "none", oldM.rows, oldM.columns, oldM.multiplier),
theRows(NULL), theColumns(NULL),
theElements(MaxElements(oldM.rows,oldM.columns), oldM.columns, this)
{
	DEBUG_SPARSE(cerr << "SparseMatrix copy constructor:" << endl;)
    
    // We're a 2D Matrix, even if only one row or column
    dimensions = 2;
    
	DEBUG_SPARSE(cerr << rows << "X" << columns << ":" << endl;)
    
    // Allocate storage for row and column lists (hash table already
    // allocated at initialization time; see initializer list, above).
    if ((theRows = new list<Element*>[rows]) == NULL)
        throw Matrix_bad_alloc("Failed allocating storage for SparseMatrix.");
    if ((theColumns = new list<Element*>[columns]) == NULL)
        throw Matrix_bad_alloc("Failed allocating storage for SparseMatrix.");
    
    try {
        copy(oldM);
    } catch(Matrix_invalid_argument e) {
        cerr << "Failure during SparseMatrix copy constructor\n"
        << "\tError was: " << e.what() << endl;
        exit(-1);
    }
}

// Destructor
SparseMatrix::~SparseMatrix()
{
	DEBUG_SPARSE(cerr << "Destroying SparseMatrix" << endl;)
	clear();
}

// Assignment operator
SparseMatrix& SparseMatrix::operator=(const SparseMatrix& rhs)
{
    if (&rhs == this)
        return *this;
    
	DEBUG_SPARSE(cerr << "SparseMatrix::operator=" << endl;)
    
    clear();
	DEBUG_SPARSE(cerr << "\t\tclear() complete, setting data member values." << endl;)
    
    SetAttributes(rhs.type, rhs.init, rhs.rows,
                  rhs.columns, rhs.multiplier, rhs.dimensions);
    
	DEBUG_SPARSE(cerr << "\t\tvalues set, ready to allocate." << endl;)
    
    alloc();
    
	DEBUG_SPARSE(cerr << "\t\talloc() complete, ready to copy." << endl;)
    
    try {
        copy(rhs);
    } catch(Matrix_invalid_argument e) {
        cerr << "\tFailure during SparseMatrix assignment operator\n"
        << "\tError was: " << e.what() << endl;
        exit(-1);
    }
	DEBUG_SPARSE(cerr << "\t\tcopy() complete; returning by reference." << endl;)
    return *this;
}

// Clear out storage
void SparseMatrix::clear(void)
{
	DEBUG_SPARSE(cerr << "\tclearing " << rows << "X" << columns << " SparseMatrix...";)
    
    // Since each Element is only allocated once (and shared among the
    // row and column lists and the hash table), we only need to
    // traverse all of the row lists and delete the Elements. This
    // invalidates all of the column list and hash table pointers, but
    // those lists will go away when the column list destructors are
    // called in response to us deleting the column array and when
    // the hash table is cleared.
    if (theRows != NULL) {
        for (int i = 0; i < rows; i++)
            for (list<Element*>::iterator it = theRows[i].begin();
                 it != theRows[i].end(); it++)
                delete *it;
        delete [] theRows;
    }
    
    if (theColumns != NULL)
        delete [] theColumns;
    
    theElements.clear();
    
    theRows = NULL;
    theColumns = NULL;
    
	DEBUG_SPARSE(cerr << "done." << endl;)
}


/*
 @method remove_lists
 @discussion Remove an Element from the SparseMatrix lists (but not
 the hash table). This is meant to be called from a HashTable
 method.
 */
void SparseMatrix::remove_lists(Element* el)
{
    theRows[el->row].remove(el);
    theColumns[el->column].remove(el);
    
    delete el;
}

// Copy matrix to this one
void SparseMatrix::copy(const SparseMatrix& source)
{
	DEBUG_SPARSE(cerr << "\t\t\tcopying " << source.rows << "X" << source.columns
                 << " SparseMatrix...";)
    
    // We will access the source row-wise, inserting new Elements into
    // the current SparseMatrix's row and column lists.
    for (int i=0; i<rows; i++) {
        for (list<Element*>::iterator it = source.theRows[i].begin();
             it != source.theRows[i].end(); it++) {
            Element* el = new Element(i, (*it)->column, (*it)->value);
            theRows[i].push_back(el);
            theColumns[el->column].push_back(el);
            try {
                theElements.insert(el);
            } catch(Matrix_invalid_argument e) {
                cerr << "\nFailure during SparseMatrix copy() for element "
                << el->value << " at (" << el->row << "," << el->column << ")" << endl;
                cerr << "\twith " << theElements.size << " elements already copied at i="
                << i << ", hashed to " << theElements.hash(el) << " in table with capacity "
                << theElements.capacity << endl;
                cerr << "\tSource was: " << source << endl << endl;
                throw e;
            }
        }
    }
    
	DEBUG_SPARSE(cerr << "\t\tdone." << endl;)
}

// Read row from XML and add items to SparseMatrix
void SparseMatrix::rowFromXML(TiXmlElement* rowElement)
{
    int rowNum;
    if (rowElement->QueryIntAttribute("number", &rowNum)!=TIXML_SUCCESS)
        throw Matrix_invalid_argument("Attempt to read SparseMatrix row without a number");
    
    // Iterate through the entries, inserting them into the row, column,
    // and hash table
    for (TiXmlElement* child = rowElement->FirstChildElement("Entry");
         child != NULL; child=child->NextSiblingElement("Entry")) {
        int colNum;
        BGFLOAT val;
        if (child->QueryIntAttribute("number", &colNum)!=TIXML_SUCCESS)
            throw Matrix_invalid_argument("Attempt to read SparseMatrix Entry without a number");
        if (child->QueryFLOATAttribute("value", &val)!=TIXML_SUCCESS)
            throw Matrix_invalid_argument("Attempt to read SparseMatrix Entry without a value");
        Element* el = new Element(rowNum, colNum, val);
        theRows[rowNum].push_back(el);
        theColumns[colNum].push_back(el);
        try {
            theElements.insert(el);
        } catch(Matrix_invalid_argument e) {
            cerr << "Failure during SparseMatrix rowFromXML: " << e.what() << endl;
            exit(-1);
        }
    }
}


// Allocate internal storage
void SparseMatrix::alloc(void)
{
    if (theRows != NULL)
        throw MatrixException("Attempt to allocate storage for non-cleared SparseMatrix");
    
    // Allocate the 1D array
    if ((theRows = new list<Element*>[rows]) == NULL)
        throw Matrix_bad_alloc("Failed allocating storage for SparseMatrix.");
    
    if (theColumns != NULL)
        throw MatrixException("Attempt to allocate storage for non-cleared SparseMatrix");
    
    // Allocate the 1D array
    if ((theColumns = new list<Element*>[columns]) == NULL)
        throw Matrix_bad_alloc("Failed allocating storage for SparseMatrix.");
    
    // Set the hash table capacity
    theElements.resize(MaxElements(rows, columns), columns);
    
	DEBUG_SPARSE(cerr << "\t\tStorage allocated for "<< rows << " row by "
                 << columns << " column SparseMatrix." << endl;)
    
}


// Polymorphic output
void SparseMatrix::Print(ostream& os) const
{
    if (theElements.size == 0) // must catch this before here; not XML
        os << "empty";
    
    for (int i=0; i<rows; i++) {
        if (theRows[i].begin() == theRows[i].end())
            continue;
        os << "   <Row number=\"" << i << "\">" << endl;
        for (list<Element*>::iterator it = theRows[i].begin();
             it != theRows[i].end(); it++) {
            // Prune out any left over zero elements
            if ((*it)->value != 0.0)
                os << "      <Entry number=\"" << (*it)->column
                << "\" value=\"" << (*it)->value << "\"/>" << endl;
        }
        os << "   </Row>" << endl;
    }
}

// convert Matrix to XML string
string SparseMatrix::toXML(string name) const
{
    stringstream os;
    
    os << "<Matrix ";
    if (name != "")
        os << "name=\"" << name << "\" ";
    os << "type=\"sparse\" rows=\"" << rows
    << "\" columns=\"" << columns << "\" ";
    if (theElements.size == 0) {           // Empty sparse matrix has special XML
        os << "init=\"const\" multiplier=\"0.0\"/>";
    } else {                               // Non-empty: output contents' XML
        os << "multiplier=\"1.0\">" << endl;
        os << "   " << *this << endl;
        os << "</Matrix>";
    }
    
    return os.str();
}


// Mutator
/*
 @method operator()
 @discussion Access value of element at (row, column) -- mutator. Constant
 time as long as number of items in the N x M SparseMatrix is less
 than 4*sqrt(N*M). If there is no element at the given location,
 then a zero value one is created.
 @param r element row
 @param c element column
 @result value of element at that location
 @throws Matrix_bad_alloc
 */
BGFLOAT& SparseMatrix::operator()(int r, int c)
{
    Element* el = theElements.retrieve(r, c);
    
    // Because we're a mutator, we need to insert a zero-value element
    // if the element wasn't found. We will rely on other methods to
    // eliminate zero elements "on the fly".
    if (el == NULL) {
        if ((el = new  Element(r, c, 0.0)) == NULL)
            throw Matrix_bad_alloc("Failed allocating storage for SparseMatrix.");
        theRows[r].push_back(el);
        theColumns[c].push_back(el);
        try {
            theElements.insert(el);
        } catch(Matrix_invalid_argument e) {
            cerr << "Failure during SparseMatrix operator() at row "
            << r << " column " << c << ": " << e.what() << endl;
            exit(-1);
        }
    }
    
    return el->value;
}

/**
 Unary minus. Negate all elements of the SparseMatrix.
 @return A new SparseMatrix, with same size as the current one.
 */
const SparseMatrix SparseMatrix::operator-() const
{
    SparseMatrix result(*this);
    
    // Iterate over all elements, negating their values
    for (int i=0; i<result.rows; i++)
        for (list<Element*>::iterator it = result.theRows[i].begin();
             it != result.theRows[i].end(); it++)
            (*it)->value = - (*it)->value;
    
    return result;
}


// Math operations. For efficiency's sake, these methods will be
// implemented as being "aware" of each other (i.e., using "friend"
// and including the other subclasses' headers).



const SparseMatrix SparseMatrix::operator+(const SparseMatrix& rhs) const
{
    throw Matrix_domain_error("SparseMatrix addition not yet implemented");
}


// Multiply the rhs into the current object
const SparseMatrix SparseMatrix::operator*(const SparseMatrix& rhs) const
{
    throw Matrix_domain_error("SparseMatrix product not yet implemented");
}


// Vector times a Sparse matrix
const VectorMatrix operator*(const VectorMatrix& v, const SparseMatrix& m)
{
    if (m.rows != v.size) {
        throw Matrix_domain_error("Illegal vector/matrix product. Rows of matrix must equal vector size.");
    }
    
    // the result is a vector the same size as m columns
    VectorMatrix result("complete", "const", 1, m.columns, 0.0, "");
    
    // To get each element of result, we will iterate down a column of m,
    // multiplying each element found by the element in v at position
    // equal to the row position of the m element. The result is the sum
    // of those products.
    for (int col=0; col<m.columns; col++) {
        BGFLOAT sum = 0.0;
        for (list<SparseMatrix::Element*>::iterator el=m.theColumns[col].begin();
             el!=m.theColumns[col].end(); el++)
            sum += (*el)->value * v[(*el)->row];
        result[col] = sum;
    }
    
    return result;
    
}

