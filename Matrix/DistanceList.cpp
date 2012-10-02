/*!
  @file DistanceList.cpp
  @brief Specialized class for storing information for circle overlap computation
  @author Michael Stiber
  @date $Date: 2006/11/18 04:42:31 $
  @version $Revision: 1.1.1.1 $
*/

// $Log: DistanceList.cpp,v $
// Revision 1.1.1.1  2006/11/18 04:42:31  fumik
// Import of KIIsimulator
//
// Revision 1.6  2005/03/08 19:54:46  stiber
// Modified comments for Doxygen.
//
// Revision 1.5  2005/03/07 16:05:43  stiber
// Added code to track the number of overlapping units, mostly for additional debuggin information
// but also for some minor optimization. Also modified to take into account restructuring of
// SparseMatrix constructors.
//
// Revision 1.4  2005/02/18 19:39:01  stiber
// Added RCS Log keyword.
//
//

#include <cmath>

#include "DistanceList.h"

#include "SourceVersions.h"

static VersionInfo version("$Id: DistanceList.cpp,v 1.1.1.1 2006/11/18 04:42:31 fumik Exp $");

using namespace std;



/*
  @method DistanceList
  @discussion Allocate storage for an n-unit unitList.
  @throws KII_bad_alloc
*/
DistanceList::DistanceList(const FLOAT xlocs[], const FLOAT ylocs[], 
			   const VectorMatrix& radii) 
  : unitList(NULL), numUnits(radii.Size()), numOverlap(0)
{
  // Since this is a symmetric array, there's no need to store
  // information for unit n (all that information is already in the
  // lists for units 1..n-1
  if ((unitList = new ListItem[numUnits-1]) == NULL)
    throw KII_bad_alloc("Failed allocating memory for DistanceList.");

  // Fill the lists. For unit i, we need only store information for
  // connections to units i+1..n, as earlier units already have the
  // information for their connections with i.
  FLOAT tempDist, tempDist2, tempDelta, deltaX, deltaY;
  for (int u1=0; u1<numUnits-1; u1++) {
    unitList[u1].radius = radii[u1];
    for (int u2=u1+1; u2<numUnits; u2++) {
      deltaX = xlocs[u1]-xlocs[u2];
      deltaY = ylocs[u1]-ylocs[u2];
      tempDist2 = deltaX*deltaX + deltaY*deltaY;
      tempDist = sqrt(tempDist2);
      tempDelta = tempDist - (radii[u1]+radii[u2]);
      if (tempDelta < 0.0) {
	unitList[u1].overlapping.push_back(SublistItem(u2, radii[u2], tempDist,
						       tempDist2, tempDelta));
	numOverlap++;
      } else {
	unitList[u1].nonOverlapping.push_back(SublistItem(u2, radii[u2],
							  tempDist, tempDist2, tempDelta));
      }
    }
  }
}


/*
  @method Update
  @discussion Update the DistanceList information based on the given
  new radii.
  @param radii unit connectivity radii must have same number of
  elements as numUnits
  @throws KII_invalid_argument
*/
void DistanceList::Update(const VectorMatrix& radii)
{
  if (radii.Size() != numUnits)
    throw KII_invalid_argument("Wrong number of elements in radii for distance update.");

#ifdef DEBUG2
  cerr << "Updating DistanceList with radii " << radii << endl;
#endif
  FLOAT tempDelta;
  SublistItem tempItem;
  list<SublistItem>::iterator u2iter;
  // Iterate through all units
  for (int u1=0; u1<numUnits-1; u1++) {
#ifdef DEBUG2
    cerr << u1 << ", ";
#endif
    // Update unit connection radius
    unitList[u1].radius = radii[u1];
    // Iterate over all overlapping and non-overlapping other units
    // First, overlapping:
    u2iter = unitList[u1].overlapping.begin();
    while (u2iter != unitList[u1].overlapping.end()) {
      // Compute and update other unit's information
      u2iter->radius = radii[u2iter->otherUnit];
      tempDelta = u2iter->dist - (unitList[u1].radius+u2iter->radius);
      u2iter->Delta = tempDelta;
      // Move it if non-overlapping, if necessary (Delta>=0). If the
      // move occurs, the iterator will point to the next item. If
      // not, then the iterator must be advanced explicitly.
      if (tempDelta >= 0) {
	tempItem = *u2iter;
	u2iter = unitList[u1].overlapping.erase(u2iter);
	unitList[u1].nonOverlapping.push_back(tempItem);
	numOverlap--;
      } else
	u2iter++;
    }
    // Then, non-overlapping:
    u2iter = unitList[u1].nonOverlapping.begin();
    while (u2iter != unitList[u1].nonOverlapping.end()) {
      // Compute and update other unit's information
      u2iter->radius = radii[u2iter->otherUnit];
      tempDelta = u2iter->dist - (unitList[u1].radius+u2iter->radius);
      u2iter->Delta = tempDelta;
      // Move it if overlapping, if necessary (Delta<0). If the
      // move occurs, the iterator will point to the next item. If
      // not, then the iterator must be advanced explicitly.
      if (tempDelta < 0) {
	tempItem = *u2iter;
	u2iter = unitList[u1].nonOverlapping.erase(u2iter);
	unitList[u1].overlapping.push_back(tempItem);
	numOverlap++;
#ifdef DLDEBUG
	cerr << "\tDL::Update(): overlap of " << u1 << " and " << tempItem.otherUnit
	     << " with radii " << unitList[u1].radius << " and " << tempItem.radius
	     << ", respectively" << endl;
#endif
      } else
	u2iter++;
    }
  }  // end for (int u1-0; ...)
#ifdef DEBUG2
  cerr << "done." << endl;;
#endif
}


/*
  @method ComputeAreas
  @discussion Computes the areas of overlap of the units and returns
  a matrix of those values.
  @result Matrix of areas of overlap
*/
SparseMatrix DistanceList::ComputeAreas(void) const
{
#ifdef DEBUG2
  cerr << "Computing areas of overlap: " << endl;;
#endif
  // Initialize NxN matrix to all zero
  SparseMatrix areas(numUnits, numUnits);

  // A minor optimization
  if (numOverlap == 0)
    return areas;

  // We only need to iterate over the overlapping units; all other
  // overlap areas are, of course, zero
  list<SublistItem>::iterator u2iter;
  FLOAT lenAB, lenAB2, r1, r2, // Information from storage
    r12, r22,                   // Radii squared
    cosCBA, angCBA, angCBD,     // Angles to intersection points
    cosCAB, angCAB, angCAD,
    area;                       // Computed (scalar) area of overlap
  for (int u1=0; u1<numUnits-1; u1++) {
#ifdef DEBUG2
    cerr << "\n\t" << u1 << ": ";
#endif
    r1 = unitList[u1].radius;
    u2iter = unitList[u1].overlapping.begin();
    while (u2iter != unitList[u1].overlapping.end()) {
      lenAB = u2iter->dist;
      r2 = u2iter->radius;
      area = 0.0;
      // Compute area for completely overlapping connection circles
      if (lenAB+fmin(r1,r2) <= fmax(r1,r2)) {
#ifdef DLDEBUG
	cerr << "\tDL::ComputeAreas(): complete overlap of " << u1
	     << " and " << u2iter->otherUnit << " found." << endl;
#endif
	area = M_PI * fmin(r1,r2)*fmin(r1,r2);
      } else {                    // Compute area of partial overlap
#ifdef DLDEBUG
	cerr << "\tDL::ComputeAreas(): partial overlap of " << u1
	     << " and " << u2iter->otherUnit << " found." << endl;
#endif
	lenAB2 = u2iter->dist2;
	r12 = r1*r1;
	r22 = r2*r2;
	cosCBA = (r22 + lenAB2 - r12)/(2*r2*lenAB);
	angCBA = acos(cosCBA);
	angCBD = 2 * angCBA;

	cosCAB = (r12 + lenAB2 - r22)/(2*r1*lenAB);
	angCAB = acos(cosCAB);
	angCAD = 2 * angCAB;

	area = 0.5 * (r22 * (angCBD - sin(angCBD)) + r12 * (angCAD - sin(angCAD)));
      }
#ifdef DEBUG2
      cerr << "(" << u2iter->otherUnit << ": " << area << "), ";
#endif
      areas(u1, u2iter->otherUnit) = area;
      areas(u2iter->otherUnit, u1) = area;
      u2iter++;
    }
  }
#ifdef DEBUG2
  cerr << "\n\tdone." << endl;
#endif
  return areas;
}

    
/*
  @method Print
  @discussion Output text representation os object to stream
  @param os stream to output to
*/
void DistanceList::Print(ostream& os) const
{
  for (int u1=0; u1<numUnits-1; u1++) {
    os << u1 << "r" << unitList[u1].radius << ": [" 
       << unitList[u1].overlapping << "]; (" 
       << unitList[u1].nonOverlapping << ")" << endl;
  }
}

/*
  @function operator<<
  @discussion convenience function for stream output of DistanceLists
  @param os stream to output to
  @param dl DistanceList to output
*/
ostream& operator<<(ostream& os, const DistanceList& dl) 
{
  dl.Print(os); 
  return os;
}

/*
  @method Print
  @discussion produce text representation of item to stream
  @param os stream to output to
*/
void DistanceList::SublistItem::Print(ostream& os) const
{
  os << otherUnit << "r" << radius;
}

/*
  @function operator<<
  @discussion convenience function for stream output of SublistItems
  @param os stream to output to
  @param sli SublistItem to output
*/
ostream& operator<<(ostream& os, 
		    const DistanceList::SublistItem& sli)
{
  sli.Print(os);
  return os;
}


/*
  @function operator<<
  @discussion convenience function for stream output of lists of SublistItems
  @param os stream to output to
  @param l list to output
*/
ostream& operator<<(ostream& os, 
		    list<DistanceList::SublistItem>& l)
{
  for (list<DistanceList::SublistItem>::iterator iter = l.begin();
       iter != l.end(); iter++) {
    os << *iter << "|->";
  }
  return os;
}


