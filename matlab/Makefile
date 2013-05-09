#
# Source directories
#
XMLDIR = ../tinyxml
SVDIR = ../SourceVersions

#
# Build tools
#

CXX = /usr/local/MATLAB/R2010b/bin/mex

CXXFLAGS = -I../include -I$(XMLDIR) -I$(SVDIR) -DTIXML_USE_STL 

CFLAGS = -Wall

#
# here are all the objects
#

OBJS = readLocs.mexglx readMatrix.mexglx readState.mexglx

XMLOBJ = $(XMLDIR)/tinyxml.o $(XMLDIR)/tinyxmlerror.o $(XMLDIR)/tinyxmlparser.o
SVOBJ = $(SVDIR)/SourceVersions.o

.SUFFIXES: .mexglx

#
# how to make the target
#

all: $(OBJS)

#
# some default targets
#
clean:
	rm -f *.mexglx

.cpp.mexglx:	
	$(CXX) $(CXXFLAGS) CFLAGS="$(CFLAGS)" $<  $(XMLOBJ) $(SVOBJ)


