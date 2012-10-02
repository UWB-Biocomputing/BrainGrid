/*!
  @file InputGenerator.cpp
  @brief Provides sequence of inputs to simulation from XML file
  @date $Date: 2006/11/18 04:42:31 $
  @version $Revision: 1.1.1.1 $
*/

// InputGenerator.cpp  Provides sequence of inputs to simulation from
// XML file

// Written February 2005 by Michael Stiber

// $Log: InputGenerator.cpp,v $
// Revision 1.1.1.1  2006/11/18 04:42:31  fumik
// Import of KIIsimulator
//
// Revision 1.4  2005/03/08 19:54:59  stiber
// Modified comments for Doxygen.
//
// Revision 1.3  2005/03/07 16:08:10  stiber
// Fixed problem where the tinyxml Attribute() method returns a C string, which
// was being assigned to a C++ string object. The returned const pointer is now
// assigned to a const char* variable, which is then checked to see if it is a
// NULL pointer.
//
// Revision 1.2  2005/02/22 19:01:18  stiber
// Added capability to loop through the sequence of inputs, rather than
// just outputting the last one infinitely.
//
// Revision 1.1  2005/02/18 20:46:58  stiber
// Initial revision
//
//

#include "InputGenerator.h"
#include "KIIexceptions.h"
#include "MatrixFactory.h"

#include "SourceVersions.h"

static VersionInfo version("$Id: InputGenerator.cpp,v 1.1.1.1 2006/11/18 04:42:31 fumik Exp $");


/*
  @method iterator
  @discussion Initializes the iterator by associating it with an
  InputGenerator object, starting the internal vector list
  iterator at the first input vector, and loading the first input
  vector.
  @param ig Reference to the associated InputGenerator
*/
InputGenerator::iterator::iterator(InputGenerator& ig) : theGenerator(ig), currentStep(0) 
{ 
  currentVector = theGenerator.inputs.begin();
  currentInput = *currentVector;
}


/*
  @method operator++
  @discussion Updates the iterator's internals so that it will
  return the input vector for the next simulation time
  step. Prefix form
  @result The input vector after update
*/
VectorMatrix InputGenerator::iterator::operator++()
{
  currentStep++;

  // Time to get the next input vector?
  if (((currentStep % theGenerator.updateInterval) == 0)
      && (currentVector != theGenerator.inputs.end())) {
    currentVector++;
    if (currentVector != theGenerator.inputs.end())
      currentInput = *currentVector;
    else if (theGenerator.loop) {        // We hit the end. Shall we loop?
      currentVector = theGenerator.inputs.begin();
      currentInput = *currentVector;
    }
  }
  return currentInput;
}

/*
  @method operator++
  @discussion Updates the iterator's internals so that it will
  return the input vector for the next simulation time
  step. Postfix form
  @result The input vector before update
*/
VectorMatrix InputGenerator::iterator::operator++(int)
{
  VectorMatrix savedInput = currentInput;

  currentStep++;

  // Time to get the next input vector?
  if (((currentStep % theGenerator.updateInterval) == 0)
      && currentVector != theGenerator.inputs.end()) {
    currentVector++;
    if (currentVector != theGenerator.inputs.end())
      currentInput = *currentVector;
  }
  return savedInput;
}

/*
  @method InputGenerator
  @discussion Constructor parses the XML input pointed to by the
  parameter. The parameter <i>must</i> point to an XML InputSequence
  element.
  @param is Pointer to XML InputSequence element
  @throws KII_invalid_argument
*/
InputGenerator::InputGenerator(TiXmlElement* is)
{
  Load(is);
}

/*
  @method Load
  @discussion Parse the XML input pointed to by the
  parameter. The parameter <i>must</i> point to an XML InputSequence
  element.
  @param is Pointer to XML InputSequence element
  @throws KII_invalid_argument
*/
void InputGenerator::Load(TiXmlElement* is)
{
  string loopStr;

  if (is->QueryIntAttribute("interval", &updateInterval)!=TIXML_SUCCESS)
    throw KII_invalid_argument("Update interval not specified for InputSequence in XML.");

  // Looping defaults to "no"
  const char* loopCStr = is->Attribute("loop");
  if (loopCStr == NULL)
    loopStr = "no";
  else
    loopStr = loopCStr;
  if (loopStr == "no")
    loop = false;
  else
    loop = true;

  // Load all the vectors into the list
  for (TiXmlElement* vector = is->FirstChildElement("Matrix"); 
       vector != NULL; vector = vector->NextSiblingElement("Matrix"))
    inputs.push_back(MatrixFactory::CreateVector(vector));

  // If no vectors were read, we need to bail, because we don't know
  // the input vector size
  if (inputs.size() == 0)
    throw KII_invalid_argument("No vectors in InputSequence in XML.");
}


