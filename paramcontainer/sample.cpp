#include <stdio.h>
#include <math.h>

#ifdef WIN32
#include <windows.h>
#endif

#include "ParamContainer.h"

#define _PI 3.14159265

using namespace std;

const char *projsignature="ParamContainer Sample Project";

/*
Class representing coords, which can be initialized in either Cartesian or polar manner
*/
class Coords {
	double x,y;
public:
	// This static method sets ParamContainer to accept Coords values
	static string initParam(ParamContainer &p) {
		ParamContainer xy;
		xy.initOptions(false);
		xy.setHelpString("Point coordinates: xy = Cartesian coordinates. Syntax:");
		xy.addParam("x", 0, xy.noname|xy.required, "X coordinate of point");
		xy.addParam("y", 0, xy.noname|xy.required, "Y coordinate of point");
		p.addParamType("xy", xy);

		ParamContainer polar;
		polar.initOptions(false);
		polar.setHelpString("Point coordinates: polar = polar coordinates. Syntax:");
		polar.addParam("r", 0, polar.noname|polar.required, "Norm");
		polar.addParam("phi", 0, polar.noname, "Azimuth (0 by default)", "0");
		p.addParamType("polar", polar);
		return "xy|polar";
	}
	Coords():x(0),y(0) {}
	// Construct the object from ParamContainer
	Coords(string ctype, const ParamContainer &p) {
		if(ctype=="xy") {
			x=atof(p["x"].c_str());
			y=atof(p["y"].c_str());
		} else {
			x=atof(p["r"].c_str())*cos(atof(p["phi"].c_str()));
			y=atof(p["r"].c_str())*sin(atof(p["phi"].c_str()));
		}
	}
	// get coordinates
	double getX() const {return x;}
	double getY() const {return y;}
};

/*
Interface representing shape
*/
class Shape {
public:
	// calculate shape area
	virtual double getArea() = 0;
};

/*
Square
*/
class Square:public Shape {
	Coords c1, c2;
public:
	static string initParam(ParamContainer &p) {
		ParamContainer p1;
		// spdump=false, we don't need to see several explanations of coords initialization in 
		// our help screen. Change it to p1.initOptions(false, true), run program without arguments
		// and see, what's the difference.
		p1.initOptions(false, false);
		p1.setHelpString("Shape: square. Syntax:");
		// Initializing subtypes of coordinates representation
		string ctypes=Coords::initParam(p1);
		p1.addParam("lefttop", 0, p1.noname|p1.required, "Coords of left-top corner", "", ctypes);
		p1.addParam("rightbottom", 0, p1.noname|p1.required, "Coords of right-bottom corner", "", ctypes);
		p.addParamType("square", p1);
		return "square";
		
	}
	// constructing
	Square(const ParamContainer &p) {
		c1=Coords(p["lefttop"], p.child("lefttop"));
		c2=Coords(p["rightbottom"], p.child("rightbottom"));
	}
	virtual double getArea() {
		return fabs((c2.getX()-c1.getX())*(c2.getY()-c1.getY()));
	}
};

/*
Circle
*/
class Circle:public Shape {
	Coords c;
	double r;
public:
	static string initParam(ParamContainer &p) {
		ParamContainer p1;
		p1.initOptions(false, false);
		p1.setHelpString("Shape: circle. Syntax:");
		string ctypes=Coords::initParam(p1);
		p1.addParam("center", 'c', p1.required, "Coords of the center", "", ctypes);
		p1.addParam("radius", 'r', p1.required, "Radius");
		p.addParamType("circle", p1);
		return "circle";
		
	}	
	Circle(const ParamContainer &p) {
		c=Coords(p["center"], p.child("center"));
		r=atof(p["radius"].c_str());
	}
	virtual double getArea() {
		return _PI*r*r;
	}
};

int main(int argc, char* argv[])
{
	int width;
#ifdef WIN32
		// Get window width to output help and erro messages better
		CONSOLE_SCREEN_BUFFER_INFO csbi;
		GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
		if(csbi.dwSize.X<=0)
		{
			GetConsoleScreenBufferInfo(GetStdHandle(STD_ERROR_HANDLE), &csbi);
			if(csbi.dwSize.X<=0) width=78; else width=csbi.dwSize.X-1;
		}
		else width=csbi.dwSize.X-1;
#else
		// Add here code to obtain width in your system
		width=78;
#endif

	// ParamContainer initialization
	ParamContainer p;
	p.initOptions(false);	// Don't allow unknown parameters
	p.setHelpString(string("Test ParamContainer application\nUsage: ")+argv[0]+" ");

	// Initializing subtypes of shape representation
	// maybe it'll look better inside some Shape Factory class
	string shapesquare=Square::initParam(p);
	string shapecircle=Circle::initParam(p);
	string shapetypes=shapesquare+"|"+shapecircle;

	string coordstypes=Coords::initParam(p);	// Initializing subtypes of coordinates representation

	// Creating parameter list
	p.addParam("startpoint", 'p', p.required, "Start point coordinates", "", coordstypes);
	p.addParam("shape", 's', p.required, "Shape", "", shapetypes);
	p.addParam("projfile", 0, p.noname, "Project file name to load parameters from");
	p.addParam("save", 'o', p.novalue, "Save parameters to the project file instead of loading");

	// No arguments: show help and quit
	if(argc==1)
	{
		p.dumpHelp(stdout, true, width);
		return 2;
	}
	// Parsing
	ParamContainer::errcode err=p.parseCommandLine(argc, argv);
	// Projectfile specified: loading data from it
	if((err==p.errOk || err==p.errRequiredParameterMissing) && p["projfile"]!="" && p["save"]=="")
		err=p.loadParams(p["projfile"], projsignature);
	// Parse error: print error message and exit
	if(err!=p.errOk)
	{
		int errpos;
		printf("%s\n", p.getErrCmdLine(errpos, width-5).c_str());
		for(;errpos-->0;) printf(" ");
		printf("^\n");
		printf("%s\n", p.getErrorMessage(err).c_str());
		return 1;
	}
	// All seems to be ok: creating objects
	Coords pt(p["startpoint"], p.child("startpoint"));

	Shape *s;

	// Creating shape
	// maybe it'll look better inside some Shape Factory class
	if(p["shape"]==shapecircle) s=new Circle(p.child("shape"));
	if(p["shape"]==shapesquare) s=new Square(p.child("shape"));

	// Saving project if needed
	if(p["projfile"]!="" && p["save"]!="")
	{
		std::string fname=p["projfile"];
		p.unsetParam("save");
		p.unsetParam("projfile");
		p.saveParams(fname, projsignature);
	}

	// Some code using our objects
	printf("Point coords: (%g, %g)\n", pt.getX(), pt.getY());
	printf("Shape area: %g\n", s->getArea());

	// Cleaning up
	delete s;
	return 0;
}
