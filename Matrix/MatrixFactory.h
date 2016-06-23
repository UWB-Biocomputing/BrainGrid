/*!
  @file MatrixFactory.h
  @brief Deserializes Matrices from XML
  @author Michael Stiber
  @date $Date: 2006/11/18 04:42:32 $
  @version $Revision: 1.1.1.1 $
*/


// Written December 2004 by Michael Stiber

// $Log: MatrixFactory.h,v $
// Revision 1.1.1.1  2006/11/18 04:42:32  fumik
// Import of KIIsimulator
//
// Revision 1.5  2005/03/08 19:53:01  stiber
// Modified comments for Doxygen.
//
// Revision 1.4  2005/02/22 19:05:12  stiber
// Modified CreateSparse to create sparse matrices from diagonal matrix
// types.
//
// Revision 1.3  2005/02/17 15:33:49  stiber
// Added CreateSparse method to create SparseMatrices from XML.
//
// Revision 1.2  2005/02/09 18:52:40  stiber
// "Completely debugged".
//
// Revision 1.1  2004/12/06 20:03:05  stiber
// Initial revision
//

#ifndef _MATRIXFACTORY_H_
#define _MATRIXFACTORY_H_

#include <string>

#include "CompleteMatrix.h"
#include "VectorMatrix.h"

using namespace std;

/*!
  @class MatrixFactory
  @brief Deserializes Matrix objects from XML

  This class deserializes Matrix subclass objects from
  XML. It is a helper class with all static methods. MatrixFactory
  methods are used solely as class methods to create Matrix subclass
  objects. MatrixFactory instances are never created.
*/
class MatrixFactory
{
public:

  /*!
    Create the appropriate type of subclass object, based
    on the XML attributes. The object is allocated on the heap, and it
    is the responsibility of the client to delete it.
    @throws KII_invalid_argument
    @param matElement pointer to Matrix XML element
    @return A pointer to the Matrix subclass object.
  */
  static Matrix* CreateMatrix(TiXmlElement* matElement);

  /*!
    Create a VectorMatrix, based
    on the XML attributes. The object is returned by value. Why is
    this here, rather than just an alternative CompleteMatrix
    constructor? To isolate the Matrix class hierarchy from tinyxml.
    @throws KII_invalid_argument
    @throws KII_domain_error
    @param matElement pointer to Matrix XML element
    @return The VectorMatrix object.
  */
  static VectorMatrix CreateVector(TiXmlElement* matElement);

  /*!
    Create a CompleteMatrix, based
    on the XML attributes. The object is returned by value. Why is
    this here, rather than just an alternative CompleteMatrix
    constructor? To isolate the Matrix class hierarchy from tinyxml.
    @throws KII_invalid_argument
    @param matElement pointer to Matrix XML element
    @return The CompleteMatrix object.
    @throws KII_invalid_argument
  */
  static CompleteMatrix CreateComplete(TiXmlElement* matElement);

  /*!
    Create a SparseMatrix, based
    on the XML attributes. The object is returned by value.
    @throws KII_invalid_argument
    @param matElement pointer to Matrix XML element
    @return The SparseMatrix object.
  */
  static SparseMatrix CreateSparse(TiXmlElement* matElement);

private:

  /*!
    @brief Read attributes from a Matrix XML element.
    @throws KII_invalid_argument
    @param matElement Matrix XML element
    @param type Matrix type: "diag" (diagonal matrices), "complete" (all values
    specified), or "sparse". Required.
    @param init Matrix initialization: "none" (initialization data
    explicitly given, default),  "const" (initialized to muliplier, if
    present, else 1.0), "random" (random values in the range [0,1]),
    or "implementation" (must be initialized by caller, not
    creator). Optional (defaults to "none").
    @param rows Number of rows in Matrix. Required.
    @param columns Number of columns columns in Matrix. Required
    @param multiplier multiplier used for initialization, default 1.0. Optional (defaults to 1.0).
  */
  static void GetAttributes(TiXmlElement* matElement,
			    string& type, string& init,
			    int& rows, int& columns,
			    FLOAT& multiplier);

  /*!
    @brief By making this private, I prevent construction of MatrixFactories by clients.
    @throws KII_exception
  */
  MatrixFactory()
  { throw KII_exception("Private MatrixFactory constructor called"); }

};


#endif
