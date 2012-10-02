/*!
  @file DistanceList.h
  @brief Specialized class for storing information for circle overlap computation
  @author Michael Stiber
  @date $Date: 2006/11/18 04:42:31 $
  @version $Revision: 1.1.1.1 $
*/

// $Log: DistanceList.h,v $
// Revision 1.1.1.1  2006/11/18 04:42:31  fumik
// Import of KIIsimulator
//
// Revision 1.5  2005/03/08 19:51:32  stiber
// Modified comments for Doxygen.
//
// Revision 1.4  2005/03/07 16:27:25  stiber
// Added data member to track number of units with overlap.
//
// Revision 1.3  2005/02/18 19:40:42  stiber
// Added RCS Log keyword.
// Added #ifndef...#endif wrapping, which was originally forgotten.
//
//

#ifndef _DISTANCELIST_H_
#define _DISTANCELIST_H_


#include <iterator>
#include <list>

#include "VectorMatrix.h"
#include "SparseMatrix.h"

//fmin() and fmax() are not available in WIN32, min() and max() are.
//fmin()/fmax() deal with doubles min()/max() is a template class.
#ifdef _WIN32	
#define fmin __min	
#define fmax __max	
#ifndef M_PI	//M_PI is not provided by all WIN32
#define M_PI 3.14159265358979323846
#endif
#endif

/*!
  @class DistanceList

  This is a specialized class used to aid in efficient
  update of the area of overlap of all pairs of a set of circles. It
  is something like an adjacency list. It is assumed that the location
  of each unit and its circle's radius is stored externally. The basic
  principles of this class are:

  -# Segregate units based on whether there is any overlap at all
  between their circles into two sublists. This allows the area of
  overlap algorithm to focus on pairs that have overlap.

  -# Quickly scan the two sublists to update values and shift items
  between the two lists.
*/
class DistanceList {
  struct SublistItem;
  friend struct SublistItem;

  /*!
    @struct SublistItem

    This structure holds all of the information needed
    for the efficient computation of circle overlap for a pair of
    points. Basically, this is all of the information that doesn't
    change during program execution or that can be recomputed as the
    change from the previous value.
  */
  struct SublistItem {

    /*!
      @brief initializes struct members
      @param ou other unit number
      @param r other unit's radius
      @param d distance between this unit and other unit
      @param d2 distance squared
      @param D Difference between inter-unit distance and sum of two radii
    */
    SublistItem(int ou=0, FLOAT r=0.0, FLOAT d=0.0, FLOAT d2=0.0, FLOAT D=0.0)
      : otherUnit(ou), radius(r), dist(d), dist2(d2), Delta(D) {}

    /*!
      @brief produce text representation of item to stream
      @param os stream to output to
    */
    void Print(ostream& os) const;

    /*! The unit number of the other unit of the pair [0,n] */
    int otherUnit;

    /*! Radius of other unit */
    FLOAT radius;

    /*! Distance between this and the other unit */
    FLOAT dist;

    /*! Distance squared between this and the other unit */
    FLOAT dist2;

    /*! Difference between distance and sum of the two radii */
    FLOAT Delta;
  };

  struct ListItem;
  friend struct ListItem;

  /*!
    @struct ListItem
    This stucture keeps track of two lists of other units'
    information: units whose circles overlap with this one and units
    with non-overlapping circles.
  */
  struct ListItem{

    /*! List of information for units with overlapping circles */
    list<SublistItem> overlapping;

    /*! List of information for units with non-overlapping circles */
    list<SublistItem> nonOverlapping;

     /*! The unit's connection radius (other units' are within SublistItems) */
    FLOAT radius;
  };

  friend ostream& operator<<(ostream& os, const DistanceList::SublistItem& sli);
  friend ostream& operator<<(ostream& os, 
			     list<DistanceList::SublistItem>& l);

public:

  /*!
    @brief Allocate storage and initialize an n-unit unitList.
    @param xlocs unit X locations
    @param ylocs unit Y locations
    @param radii unit connectivity radii (can determine number of
    units from this)
    @throws KII_bad_alloc
  */
  DistanceList(const FLOAT xlocs[], const FLOAT ylocs[], 
	       const VectorMatrix& radii);

  /*!
    @brief Update the DistanceList information based on the given new radii.
    @param radii unit connectivity radii must have same number of elements as numUnits
    @throws KII_invalid_argument
  */
  void Update(const VectorMatrix& radii);

  /*!
    Computes the areas of overlap of the units and returns
    a sparse matrix of those values.
    @return Matrix of areas of overlap
  */
  SparseMatrix ComputeAreas(void) const;

  /*!
    @brief Output text representation os object to stream
    @param os stream to output to
  */
  void Print(ostream& os) const;

private:

  /*!
    @brief Declared private to prevent creation of empty DistanceLists
    @throws KII_exception
  */
  DistanceList() { throw KII_exception("Attempt to create empty DistanceList"); }

  /*! Dynamically allocated array of numUnits-1 ListItems */
  ListItem* unitList;

  /*! Number of units */
  int numUnits;

  /*! Number of units with overlapping connection circles */
  int numOverlap;
};

/*!
  @brief Convenience function for stream output of DistanceLists
  @param os stream to output to
  @param dl DistanceList to output
*/
ostream& operator<<(ostream& os, const DistanceList& dl);

/*!
  @brief Convenience function for stream output of SublistItems
  @param os stream to output to
  @param sli SublistItem to output
*/
ostream& operator<<(ostream& os, 
		    const DistanceList::SublistItem& sli);


/*!
  @brief Convenience function for stream output of lists of SublistItems
  @param os stream to output to
  @param l list to output
*/
ostream& operator<<(ostream& os, 
		    list<DistanceList::SublistItem>& l);



#endif
