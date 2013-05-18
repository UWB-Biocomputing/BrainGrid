#
# Source directories
#
XMLDIR = ../tinyxml
# SVDIR = ../SourceVersions

#
# Build tools
#

CXX = /usr/local/MATLAB/R2010b/bin/mex

CXXFLAGS = -I../include -I$(XMLDIR) -DTIXML_USE_STL  # -I$(SVDIR) 

CFLAGS = -Wall

#
# here are all the objects
#
# Note that the extension is different on different platforms; that allows multiple
# executables to co-exist in the same directory. .mexglx is for Linux; .mexmaci64 is
# 64-bit Mac OS X.
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
	$(CXX) $(CXXFLAGS) CFLAGS="$(CFLAGS)" $<  $(XMLOBJ) # $(SVOBJ)


