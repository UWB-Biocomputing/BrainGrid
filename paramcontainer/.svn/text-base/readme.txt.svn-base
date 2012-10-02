Contacts
========

Support  e-mail: support@biorainbow.com
Author's e-mail: lan@biorainbow.com

ParamContainer overview
=======================

ParamContainer is simple and easy to use C++ class to parse command line
parameters. Parameters format are derived from UNIX getopt_long() function syntax
but may contain nested parameters as well. It was developed to fit requirenments
of our projects, but we'll be glad if it will be useful for somebody else. Main
features of ParamContainer are:

* Easy to use
* Structure of command line conforms object hierarchy.
* Adding/changing parameters is really easy. You don't need to modify class
interfaces and anything outside of the class which new parameter
corresponds to.
* Parameters can be saved to the project file and loaded later.
* When command line contains additional file names, they paths will be
converted to relative in project file, so you can freely move project with
all required files to the different location.
* ParamContainer can be used as internal interface between presentation
(GUI) and logic parts of the project. You can use the same logic part in
graphics/command-line versions of your project.
* Dynamically generated help screen
* Powerful error handling
* Portability between Win32 and Unix systems (on Win32 systems there must
be WIN32 preprocessor definition).

Quick start
===========
Here is the simple way to use ParamContainer in your project:
* Add ParamContainer.cpp and ParamContainer.h into project
* Include ParamContainer.h in your main cpp file:

        #include "ParamContainer.h"

* Create ParamContainer object in your main() function

        ParamContainer p;

* Add some parameters using addParam()

        p.addParam("long-name", 'n', ParamContainer::regular, "parameter description", "default_value");        
        //"long-name" - long parameter name
        //'n' - short parameter name
        //ParamContainer::regular - parameter type (regular means that parameter is not required and has an argument)
        //"parameter description" - description
        //"default_value" - default value

* call parseCommandLine()

        p.parseCommandLine(argc, argv[]);

* obtain parameter values via []

        cout << p["long-name"];

* compile and run your program, specifying your argument:
        programname --long-name=value
or
        programname -n value
or
        programname -n "value"

Your program will look like this:

// We're writing program to list files in the directory
#include "ParamContainer.h"

main(int argc, char *argv[])
{
        // Create ParamContainer object
        ParamContainer p;
        // Add some parameters
        p.addParam("dir", 0, ParamContainer::noname);   // directory to list files in - parameter has no name and not required
        p.addParam("long-listing", 'l', ParamContainer::novalue); // -l = long listing
        p.addParam("sort", 's', ParamContainer::regular, "", "name"); // -s = sort mode, by default sorting by name
        // Done. Now parsing command line
        if(p.parseCommandLine(argc, argv[])!=p.errOk) {
                fprintf(stderr, "Error in arguments!\n");
                return -1;
        }
        // Command line parsed. Now using parameters
        if(p["long-listing"]!="") {     // -l or --long-listing is specified
                ....
        }
        if(p["sort"]=="name") ....      // -s name, sorting by name
        else if(p["sort"]=="date") ...  // -s date, sorting by date/time
        else if(p["sort"]=="size") ...  // -s size, sorting by size
        else {
                fprintf(stderr, "Invalid sort mode!\n");
                return -1;
        }
        doListing(p["dir"]);    // list directory
        return 0;
}

More complex example with nested parameters, error handling, help and load/save features
is provided in sample.cpp and described at the end of this file.

ParamContainer interface
========================
Initializing:

* ParamContainer();
        Default constructor
* void initOptions(bool aup, bool sphelp=true);
        Set parameters
        aup = whether or not allow unknown parameters
        sphelp = whether or not dump help of subparameters
* static void setMsgList(char **_messages);
        Set message list (mainly error messages)
        Default is parammessages array in ParamContainer.cpp.
        You can create message lists in different languages and change
        them via this function.
* void setHelpString(std::string _helptopic);
        Set description of this ParamContainer (see sample.cpp)
        
Parameter list control:
* errcode addParam(std::string pname, char abbr=0, int type=novalue, std::string helptopic="", std::string defvalue="", std::string allowedtypes="");
        Add parameter into ParamContainer
        pname = parameter name
        abbr = one-character abbreviation of the parameter
        int = type and flags:
                ParamContainer::regular = regular parameter (with an argument)
                ParamContainer::novalue = parameter have no value (argument)
                ParamContainer::required = parameter should be present
                ParamContainer::noname = parameter have no name and identified
                        by position in command line (see sample.cpp)
                ParamContainer::filename = parameter is filename and it's path
                        will be converted to relative when saving project
                filename, noname and required may be combined
        helptopic = description string
        defvalue = default parameter value
        allowedtypes = either empty string (for simple parameters) or allowed types
                separated by '|' (for nested parameters)

* errcode addParamType(std::string tname, const ParamContainer &p);
        Add new type of nested parameter
                tname = type name
                p = nested ParamContainer (see sample.cpp)
* errcode delParam(std::string pname);
        Delete specified parameter

Parsing:
* errcode parseCommandLine(int argc, char *argv[]);
        Parse command line in argc/argv format (argv[0] is ignored)
* errcode parseCommandLine(std::string cmdline);
        Parse command line represented as string
* errcode unsetParam(std::string pname);
        Unset parameter after parsing (use this to exclude some parameters
        before saving the project - see sample.cpp)

Loading/saving:
* errcode loadParams(std::string filename, std::string signature="");
        Loads params from file with signature checking
* errcode saveParams(std::string filename, std::string signature="");
        Save parameters to file with specified signature
        This will convert filenames to relative! Be careful not to use
        them after saving. If you still need it, make a copy of ParamContainer

Obtain values:
* std::string &operator [](std::string pname);
* std::string operator [](std::string pname) const;
        Get value of specified parameter
* const ParamContainer &child(std::string pname) const;
        Get nested ParamContainer
        This function doesn't check errors. Behavior will be unpredictable if
        there's no such parameter or it's not nested.

* void dumpHelp(FILE *f, bool showparamlist=true, unsigned int width=0, std::string subtopic="") const;
        dumps help to the file stream f.
        showparamlist = obsolete, should be true
        width = number of columns in the terminal window
        subtopic = should be empty string
* static void printHelpTopic(FILE *f, std::string topic, int indent, int width, bool linebreak=true);
        helper-function to output help. May be also useful for other output.
        outputs topic to stream f with left-indent, supposing terminal window
        width is width. linebreak points whether or not put the '\n' at the end
        of output

Error handling:
* std::string getErrCmdLine(int &pos, int maxlength=70) const;
        Get command line fragment with last error (see sample.cpp)
* std::string getErrorMessage(errcode err) const
        Get error message by error code

Error codes
===========
Many functions of ParamContainer return ParamContainer::errcode enumeration value.
These values are:
        errOk                           All ok

addParam/addParamType errors:
        errParameterNameLengthExceed    Too long parameter name
        errParameterNameInvalid         Parameter contains invalid symbols
        errDublicateParameter           Parameter already exists
        errDublicateAbbreviation        Abbreviation symbol is already in use
        errDublicateType                Nested type already exists

Load/save errors:
        errInvalidSignature             Invalid file signature
        errUnableToOpenFile             Unable to open file

Parsing errors:
        errInvalidParameter             Invalid parameter (also returned by delParam() and unsetParam())
        errRequiredParameterMissing     Required parameter missing
        errParameterArgumentMissing     Parameter argument missing
        errInvalidSyntax                Command line syntax invalid
        errMissingEnclosingQuotemark    Missing enclosing quotemark
        errMissingEnclosingBracket      Missing enclosing bracket
        errInvalidType                  Invalid parameter type



Sample application
==================

Provided sample shows some features of ParamContainer in action. It contains
class Coords to represent 2D-coordinates (which can be initialized either in
Cartesian or in polar manner), interface Shape with pure virtual function
to calculate shape area and two classes (Circle and Square) which are
implements interface Shape. These classes can be easily initialized through
ParamContainer as shown in main function.

main() function creates single Coords object and single Shape object depending
on command line arguments. It calculates area of Shape object and prints
Cartesian coordinates of Coords object. Also it can save parameters to a file
and load them from. After compiling try following examples:

./sample --shape=square[xy[0 0] xy[5 5]] --startpoint=polar[10 3.1]

./sample --shape=square[xy[0 0] xy[5 5]] --startpoint=polar[10 3.1] -o test.prj

./sample test.prj

./sample --shape=square[xy[0 0] polar[5 1]] --startpoint=xy[3 5]

./sample -s circle[-c polar[5 1] -r 10] -p xy[3 5]

Also check error handling:

./sample --shape=circle[--center=polar[5 1] --radius=10 --bug] --startpoint=xy[3 5]

And automatically generated help screen:

./sample

