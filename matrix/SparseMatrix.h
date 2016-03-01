/**
 @file SparseMatrix.h
 @brief An efficient implementation of a dynamically-allocated 2D sparse array.
 @author Michael Stiber
 @date 7/31/14
 @version 2
 */

// Written February 2005 by Michael Stiber

// $Log: SparseMatrix.h,v $
// Revision 1.1.1.1  2006/11/18 04:42:32  fumik
// Import of KIIsimulator
//
// Revision 1.5  2006/10/09 15:17:09  stiber
// Updated for unary minus and to reflect max size being rows*columns.
//
// Revision 1.4  2005/03/08 19:53:35  stiber
// Modified comments for Doxygen.
//
// Revision 1.3  2005/03/07 16:29:16  stiber
// Modified Element op== and op!= to use is_at method. Modified clear
// method. Added getSize method. Replaced single constructor with
// multiple, simpler ones. Added SparseMatrix size() method. Added
// HashTable MaxElements() method.
//
// Revision 1.2  2005/02/17 15:34:28  stiber
// All KIIgrowth simulator functionality working.
//
// Revision 1.1  2005/02/16 15:31:20  stiber
// Initial revision
//
//


#ifndef _SPARSEMATRIX_H_
#define _SPARSEMATRIX_H_

#include <cmath>
#include <string>
#include <list>
#include <vector>

#include "Matrix.h"
#include "VectorMatrix.h"

using namespace std;

// Forward declarations
class SparseMatrix;
class VectorMatrix;

const VectorMatrix operator*(const VectorMatrix& v, const SparseMatrix& m);

/**
 @class SparseMatrix
 @brief An efficient implementation of a dynamically-allocated sparse 2D array
 
 This is a self-allocating and de-allocating 2D array
 that is optimized for numerical computation. It is modified from
 CompleteMatrix, but with a totally different, sparse
 implementation, including optimization of the math operations to
 take advantage of sparseness. More specifically, if the size of the
 array is NxN and the average number of non-zero entries in a row or
 column is M, then this implementation is intended to provide O(M)
 time scanning of all non-zero entries in a row or column and O(1)
 time access of the item at arbitrary location (i,j). The maximum
 number of items that a SparseMatrix can hold is sqrt(rows * columns).
 
 This magic is accomplished by storing non-zero elements as part of three data
 structures simultaneously:
 1. an element of a linked list of items in row i
 2. an element of a linked list of items in column j
 3. an element of a hash table with hashing function that uses both i and j.
 As a result, we can linearly scan a particular row or column to retrieve all of its
 elements, thus facilitating efficient math operations. We can also use the hash table
 to provide constant time random access by row and column, like a full 2D array would provide.
 */
class SparseMatrix: public Matrix {
    
	/**
	 @struct Element
	 @brief This structure is used to hold a non-zero element of a sparse matrix.
	 */
	struct Element {
		/**
		 @brief Convenience function initializes struct member
		 @param r row number
		 @param c column number
		 @param v value
		 */
		Element(int r = 0, int c = 0, BGFLOAT v = 0.0) :
        row( r ), column( c ), value( v ) {
		}
        
		/**
		 @brief Two Elements are equal if they have the same (row, column) coordinates
		 @return true if two Elements are at the same coordinates
		 */
		bool operator==(Element rhs) const {
			return is_at( rhs.row, rhs.column );
		}
        
		/**
		 @brief Two Elements are not equal if they have different (row, column) coordinates
		 @return true if two Elements are at different coordinates
		 */
		bool operator!=(Element rhs) const {
			return !is_at( rhs.row, rhs.column );
		}
        
		/**
		 @brief Checks to see if Element is at particular (row, column) coordinates
		 @return true if Element is at specified coordinates
		 */
		bool is_at(int r, int c) const {
			return ( row == r ) && ( column == c );
		}
        
		/** Row location of item */
		int row;
        
		/** Column location of item */
		int column;
        
		/** The value at the current row, column */
		BGFLOAT value;
	};
    
	/**
	 @class HashTable
	 @brief Specialized hash table for pointers to Elements.
     
	 Implemented using linear probing. Because of this choice, we cannot delete storage taken
     up by elements that get zeroed out. These elements can be re-used, of course. Consequently,
     a SparseMatrix never shrinks.
	 */
	class HashTable {
		friend class SparseMatrix;
	public:
        
		/**
		 @brief allocates and initializes hash table of given capacity
		 @param s table capacity
		 @param c number of columns in enclosing SparseMatrix
		 @param m pointer to enclosing SparseMatrix
		 */
		HashTable(int s = 0, int c = 0, SparseMatrix* m = NULL) :
        capacity( s ), size( 0 ), columns( c ), table( s, static_cast<Element*> ( NULL ) ), theMatrix( m ) {
		}
		/**
		 @brief resize hash table and initialize elements
		 @param s the new table capacity
		 @param c the new number of columns
		 */
		void resize(int s, int c);
        
		/**
		 @brief removes all elements from the hash table
		 */
		void clear(void);
        
		/**
		 Inserts Element into the hash table using linear
		 probing. If the Element is already in the table, an exception is
		 thrown  (the update method should be used to update a value).
		 @param el pointer to the Element to insert
		 @throws Matrix_bad_alloc
		 @throws Matrix_invalid_argument
		 */
		void insert(Element* el);
        
		/**
		 Retrieves Element from the hash table using linear
		 probing. If Element isn't in the table, returns NULL. Any
		 zero-value elements found in the hash table are deleted along
		 the way.
		 @param r row coordinate of Element
		 @param c column coordinate of Element
		 @return pointer to Element (NULL if not in table)
		 */
		Element* retrieve(int r, int c);
        
		/**
		 Updates the Element already in the table using linear
		 probing. If Element isn't in the table, throws and exception
		 (use insert to add new Elements to the table).
		 @param r row coordinate of Element
		 @param c column coordinate of Element
		 @param v new value for Element
		 @throws Matrix_invalid_argument
		 */
		void update(int r, int c, BGFLOAT v);
        
		/**
		 @brief returns the number of elements currently in the hash table.
		 @return size
		 */
		int getSize(void) const {
			return size;
		}
        
	private:
        
		/**
		 @brief Hashes Elements into hash table
		 */
		int hash(Element* el) const {
			return hash( el->row, el->column );
		}
        
		/**
		 @brief computes initial hash location based on row and column coordinates.
		 */
		int hash(int r, int c) const {
			return ( r * columns + c ) % capacity;
		}
        
		/** A special Element to mark deleted table locations */
		static Element deleted;
        
		/** Number of items the table can hold */
		int capacity;
        
		/** Number of items actually in the table */
		int size;
        
		/** Number of columns in the enclosing SparseMatrix */
		int columns;
        
		/** The hash table itself */
		vector<Element*> table;
        
		/** Pointer to container object, for calling its methods */
		SparseMatrix* theMatrix;
	};
    
public:
    
	/**
	 Allocate storage and initialize attributes for a
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
	SparseMatrix(int r, int c, BGFLOAT m, TiXmlElement* e);
    
	/**
	 Allocate storage and initialize attributes for a
	 diagonal sparse matrix with explicit row data. The parameter v is
	 used as a source of data for initializing the matrix (and must be
	 a string of numbers equal to the number of rows or columns). If v
	 is NULL, then the multiplier is used to initialize the diagonal
	 elements.
	 @throws Matrix_bad_alloc
	 @throws Matrix_invalid_argument
	 @param r rows in Matrix
	 @param c columns in Matrix
	 @param m multiplier used for initialization (and must be non-zero)
	 @param v string of initialization values
	 */
	SparseMatrix(int r, int c, BGFLOAT m, const char* v = NULL);
    
	/**
	 Allocate storage and initialize attributes for an
	 empty sparse matrix. This is also the default constructor.
	 @throws Matrix_bad_alloc
	 @throws Matrix_invalid_argument
	 @param r rows in Matrix
	 @param c columns in Matrix
	 */
	SparseMatrix(int r = 1, int c = 1);
    
	/**
	 @brief Copy constructor. Performs a deep copy.
	 @param oldM The source SparseMatrix
	 */
	SparseMatrix(const SparseMatrix& oldM);
    
	/**
	 @brief De-allocate storage
	 */
	virtual ~SparseMatrix();
    
	/**
	 @brief Assignment operator
	 @param rhs right-hand side of assignment
	 @return returns reference to this SparseMatrix (after assignment)
	 */
	SparseMatrix& operator=(const SparseMatrix& rhs);
    
	/**
	 @brief Access value of element at (row, column) -- mutator.
     
	 Constant time as long as number of items in the N x M SparseMatrix
	 is less than 4*sqrt(N*M). **If there is no element at the given
	 location, then a zero value one is created.** This is done because this
     method is usable as a mutator: it returns a reference to the value in the
     specified element. If this becomes a problem in other code, then the solution
     would be to add an accessor that either throws an exception if an element
     doesn't exist or that returns both the element value and a success/failure
     flag.
	 @param r element row
	 @param c element column
	 @return reference to value of element at that location
	 */
	BGFLOAT& operator()(int r, int c);
    
	/**
	 @brief Returns the number of elements in the sparse matrix.
	 @return number of elements
	 */
	int size(void) const {
		return theElements.getSize( );
	}
    
	/**
	 @brief Polymorphic output. Produces text output on stream "os". Output is by row.
	 @param os stream to output to
	 */
	virtual void Print(ostream& os) const;
    
	/**
	 @brief Produce XML representation of Matrix in string return value.
	 */
	virtual string toXML(string name = "") const;
    
	/** @name Math Operations
     
	 Math operations. For efficiency's sake, these methods will be
	 implemented as being "aware" of each other (i.e., using "friend"
	 and including the other subclasses' headers).
	 */
	//@{
    
	/**
	 Unary minus. Negate all elements of the SparseMatrix.
	 @return A new SparseMatrix, with same size as the current one.
	 */
	virtual const SparseMatrix operator-() const;
    
	/**
	 Compute the sum of two SparseMatrices of the same
	 rows and columns.
	 @throws Matrix_domain_error
	 @param rhs right-hand argument to the addition. Must have same
	 dimensions as this.
	 @return A new SparseMatrix, with value equal to the sum of this
	 one and rhs and rows and columns the same as both, returned by value.
	 */
	virtual const SparseMatrix operator+(const SparseMatrix& rhs) const;
    
	/**
	 Matrix product. Number of rows of "rhs" must equal to
	 number of columns of this.
	 @throws Matrix_domain_error
	 @param rhs right-hand argument to the product.
	 @return A SparseMatrix with number of rows equal to this and
	 number of columns equal to "rhs".
	 */
	virtual const SparseMatrix operator*(const SparseMatrix& rhs) const;
    
	/**
	 Matrix times vector. Size of v must equal to
	 number of rows of m. Size of resultant vector is equal to
	 number Of columns of m.
	 @throws Matrix_domain_error
	 @return A VectorMatrix with size equal to number of columns of m.
	 */
	friend const VectorMatrix operator*(const VectorMatrix& v, const SparseMatrix& m);
	//@}
    
protected:
    
	/** @name Internal Utilities
	 */
	//@{
    
	/**
	 @brief Frees up all dynamically allocated storage
	 */
	void clear(void);
    
	/**
	 Remove an Element from the SparseMatrix lists (but not
	 the hash table). This is meant to be called from a HashTable
	 method.
	 */
	void remove_lists(Element* el);
    
	/**
	 Performs a deep copy. It is assumed that the correct
	 storage has already been allocated and all "simple" data members
	 have already had their values copied.
	 @param source VectorMatrix to copy from
	 */
	void copy(const SparseMatrix& source);
    
	/**
	 Allocates storage for internal storage. Assumes that
	 no storage has already been allocated (i.e., must call clear()
	 before this if not in a constructor) and that all
	 simple data members (specifically, #rows and
	 #columns) have been correctly set.
	 @throws Matrix_bad_alloc
	 @throws MatrixException
	 */
	void alloc(void);
    
	/**
	 @brief Reads a row of the sparse Matrix from XML
	 @param rowElement pointer to Row XML element
	 @throws Matrix_invalid_argument
	 */
	void rowFromXML(TiXmlElement* rowElement);
	//@}
    
	// access adjustment --- allow member functions in this class to
	// access protected member of base class in other objects.
    using Matrix::dimensions;
    
private:
    
	/**
	 Determines the maximum number of elements that a
	 SparseMatrix can hold, based on its dimensions.
	 @param r number of rows in matrix
	 @param c number of columns in matrix
	 @return maximum capacity of matrix
	 */
	int MaxElements(int r, int c)
	{	return r * c;}
    
	/** 1D array of lists of elements in a row */
	list<Element*> *theRows;
    
	/** 1D array of lists of elements in a column */
	list<Element*> *theColumns;
    
	/** Hash table used to access elements by coordinates */
	HashTable theElements;
    
};

#endif

