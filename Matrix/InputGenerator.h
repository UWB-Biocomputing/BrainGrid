/*!
  @file InputGenerator.h
  @brief Provides sequence of inputs to simulation from XML file
  @date $Date: 2006/11/18 04:42:31 $
  @version $Revision: 1.1.1.1 $
*/

// $Log: InputGenerator.h,v $
// Revision 1.1.1.1  2006/11/18 04:42:31  fumik
// Import of KIIsimulator
//
// Revision 1.3  2005/03/08 19:51:56  stiber
// Modified comments for Doxygen.
//
// Revision 1.2  2005/02/22 19:04:37  stiber
// Added capability to loop through the sequence of inputs, rather than
// just outputting the last one infinitely.
//
// Revision 1.1  2005/02/18 20:47:41  stiber
// Initial revision
//
//

#ifndef _INPUTGENERATOR_H_
#define _INPUTGENERATOR_H_


#include <iterator>
#include <list>

#include "tinyxml.h"
#include "VectorMatrix.h"

/*!
  @class InputGenerator
  InputGenerator objects act as wrappers around XML files
  that define sequences of vectors to be used as simulation input. A
  nested iterator class is used to move through the sequence. The XML
  file format is:
  @verbatim
  <?xml version="1.0" standalone=no?>
  <InputSequence interval="100" loop="no">
     <Matrix type="complete" rows="1" columns="25" multiplier="1.0">
        1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     </Matrix>
     <Matrix type="complete" rows="1" columns="25" init="const" multiplier="2.0"/>
     <Matrix type="complete" rows="1" columns="25" init="const" multiplier="0.0"/>
  </InputSequence>
  @endverbatim
  The \<InputSequence\> attributes are the interval (in simulation
  iterations) between successive inputs (each input vector is
  presented for that many steps) and whether the inputs cycle around
  until the simulation ends (loop="yes") or the last input is used
  until the end of simulation (loop="no"). The default is "no".
 */
class InputGenerator {
public:
  class iterator;
  friend class iterator;

  /*!
    @class iterator
    Iterates through the sequence of inputs and provides efficient
    access to the current input. It does this by updating its copy of
    the current input vector every updateInterval increments and
    returning a const reference to this internal copy when
    dereferenced. If InputGenerator::loop is false, the InputGenerator
    is modeled as infinite length, so the final vector will be
    returned by the dereferencing operator regardless of the number of
    times the iterator is incremented. If InputGenerator::loop is
    true, the iterator starts over at the beginning of the input list
    when it hits the end.
   */
  class iterator {
  public:

    /*!
      Initializes the iterator by associating it with an
      InputGenerator object, starting the internal vector list
      iterator at the first input vector, and loading the first input
      vector.
      @param ig Reference to the associated InputGenerator
     */
    iterator(InputGenerator& ig); 

    /*!
      @brief Copy constructor
      @param it The other iterator
     */
    iterator(const iterator& it) 
      : currentInput(it.currentInput), currentStep(it.currentStep),
	currentVector(it.currentVector), theGenerator(it.theGenerator) {}

    /*!
      Updates the iterator's internals so that it will
      return the input vector for the next simulation time
      step. Prefix form
      @return The input vector after update
     */
    VectorMatrix operator++();

    /*!
      Updates the iterator's internals so that it will
      return the input vector for the next simulation time
      step. Postfix form
      @return The input vector before the update
     */
    VectorMatrix operator++(int);
    
    /*!
      @brief Returns the current input vector
     */
    const VectorMatrix& operator*() { return currentInput; }

  private:

    /*! The current input vector */
    VectorMatrix currentInput;

    /*! The current step number */
    int currentStep;

    /*! Iterator into the InputGenerator's list of vectors */
    list<VectorMatrix>::iterator currentVector;

    /*! The InputGenerator object this iterator is associated with */
    InputGenerator& theGenerator;
  };

  /*!
    @brief Creates an empty InputGenerator, which must later be loaded by Load().
  */
  InputGenerator() : updateInterval(-1), loop(false) {}

  /*!
    Constructor parses the XML input pointed to by the
    parameter. The parameter <i>must</i> point to an XML InputSequence
    element.
    @param is Pointer to XML InputSequence element
    @throws KII_invalid_argument
  */
  InputGenerator(TiXmlElement* is);

  /*!
    Parse the XML input pointed to by the
    parameter. The parameter <i>must</i> point to an XML InputSequence
    element.
    @param is Pointer to XML InputSequence element
    @throws KII_invalid_argument
  */
  void Load(TiXmlElement* is);

  /*!
    Returns iterator that presents input vectors starting
    with simulation step 0.
    @return iterator initialized to step 0 for this InputGenerator
   */
  iterator begin(void) { return iterator(*this); }

private:

  /*! List of the input vectors */
  list<VectorMatrix> inputs;

  /*! Number of simulation steps in between input changes */
  int updateInterval;

  /*! Whether or not to start over again at teh beginning when at last input */
  bool loop;
};

#endif
