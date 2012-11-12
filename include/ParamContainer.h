/*
ParamContainer.h
ParamContainer class declaration header file
(c) 2004 BioRainbow group [ http://www.biorainbow.com/ ]

Last modification: 2004-09-14
*/
#ifndef __PARAMCONTAINER_H
#define __PARAMCONTAINER_H

// Turn off troublesome warning about very long debug symbols
#ifdef _WIN32
#pragma warning(disable:4786)
#endif

/*
Include some required STL classes
*/
#include <stdio.h>
#include <map>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <vector>
#ifdef _WIN32
#include <direct.h>
#else
#include <unistd.h>
#define stricmp strcasecmp
#endif


/*
ParamContainer class declaration
*/
class ParamContainer {
	friend class TestUtils;
public:
	// Parameters flags
	enum {
		novalue=1,	// Parametr has no value (shoudn't be used together with other flags)
		regular=0,	// Regular parameter
		required=2,	// Required parameter
		noname=4,	// Parameter has no name, will be defined by it's position in command line
		filename=8	// Parameter is a file name (path will be translated when loading/saving)
	};
	// Error codes
	enum errcode {
		errOk,
			// addParam() error codes
		errParameterNameLengthExceed,
		errParameterNameInvalid,
		errDublicateParameter,
		errDublicateAbbreviation,
			// addParamType() error codes
		errDublicateType,
			// parseCommandLine() error codes
		errInvalidParameter,			// See errinfo, also can be returned by delParam()
		errRequiredParameterMissing,	// See errinfo
		errParameterArgumentMissing,	// See errinfo
		errInvalidSyntax,				// See errinfo
		errMissingEnclosingQuotemark,
		errInvalidType,					// See errinfo
		errMissingEnclosingBracket,
			// Additional loadParams()/saveParams() messages
		errInvalidSignature,
		errUnableToOpenFile,
		errLast
	};
	// Text message codes
	enum {
		msgRequiredParameter=errLast
	};
	struct param {
		/*
		When parameter has no value, then param::value is position identifier:
		bigger values mean that parameter was met later in command line
		Useful when you have parameters which are mask each other
		*/
		std::string value;
		int pflag;
		std::string help;
		char abbr;
		bool wasset;
		std::string defvalue;
		std::string allowedtypes;
		ParamContainer *p;
		param():p(NULL) {}
		~param() {if(p) delete(p);}
	};
	std::string lastcmdline;
	std::string helptopic;
private:
	// Maximum parameter name length
	static const size_t maxpnamelength;
	// Empty parameter (returned when there is no such parameter)
	static param dumbparam;
	// Association list of parameters
	std::map<std::string, param> plist;
	// Abbreviations index for parameters list
	std::map<char, std::string> alist;
	// Parameters indexed in the order of addParam() calls (used for help output)
	std::vector<std::string> vlist;
	// Helper class to manage nested parameter types (in fact this is a workaround for old compilers like MSVC 6)
	class TList {
		std::map<std::string, ParamContainer> *tlist;
	public:
		TList();
		~TList();
		TList(const TList &);
		TList &operator =(const TList &);
		ParamContainer &operator[](std::string);
		std::map<std::string, ParamContainer> *operator->() {return tlist;}
		std::map<std::string, ParamContainer> *operator->() const {return tlist;}
	};
	// Association list to represent nested parameters types
	TList tlist;
//	std::map<std::string, ParamContainer> tlist;
	// Types indexed in the order of addParamType() calls (used for help output)
	std::vector<std::string> tindex;
	// Allow unregistered parameters
	// (will ignore unknown parameters if they are encountered; kinda obsolete and shouldn't be used;
	// may be removed in future releases)
	bool allowunknownparams;
	// Dump help for nested parameters
	bool dumpsubparamhelp;

	// Command line lexical convolution
	struct cmdlineel {
		enum elflag {
			abbrparam,
			param,
			equal,
			value,
			leftbracket,
			rightbracket,
			eol
		} flag;
		std::string val;
		int pos;
		cmdlineel(elflag _flag, int _pos, std::string _val=""):flag(_flag),val(_val) ,pos(_pos){}
	};
	std::vector<cmdlineel> lexconv;

	// Command line lexical analysis
	errcode lexicalAnalysis(std::string s);
	// Command line syntax and semantic analysis
	errcode syntaxAndSemanticsAnalysis(bool fromproject=false);
	// Complex type parsing
	errcode parseComplexType(int &i, std::string name, std::string value);
	// Reconstruct command line after parsing
	std::string getCmdLine();

	// Help output helper fucntions
	std::string paramhelp(std::string param) const;
	std::string typelist(std::string param) const;
	// Reset parameter values (before second parsing)
	void reset(void);
	// Additional error info
	std::string errinfo;
	// Error position
	int errpos;
	// Text messages
	static const char** messages;

	// Relative paths translation
	void convertFilePaths(std::string dirfrom, std::string dirto);
public:
	/************
	Initializing
	************/

	// Default constructor
	ParamContainer():allowunknownparams(true),dumpsubparamhelp(true) {}

	// Parameter initialization (why it was not done in constructor?.. cannot remember)
	void initOptions(bool aup, bool sphelp=true)
	{
		allowunknownparams=aup;
		dumpsubparamhelp=sphelp;
	}
        // Set message list (mainly error messages)
        // Default is parammessages array in ParamContainer.cpp.
        // You can create message lists in different languages and change
        // them via this function.
	static void setMsgList(const char **_messages) {messages=_messages;}

	/*
	Set description of this ParamContainer
	*/
	void setHelpString(std::string _helptopic)
	{
		helptopic=_helptopic;
	}

	/*********************
	Parameter list control
	*********************/
	/*
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
	*/
	errcode addParam(std::string pname, char abbr=0, int type=novalue,
		std::string helptopic="", std::string defvalue="", std::string allowedtypes="");
	/*
	Add new type of nested parameter
	*/
	errcode addParamType(std::string tname, const ParamContainer &p);
	/*
	Delete specified parameter
	*/
	errcode delParam(std::string pname);

	/******
	Parsing
	******/
	/*
	Parse command line in argc/argv format (argv[0] is ignored)
	*/
	errcode parseCommandLine(int argc, char *argv[]);
	/*
	Parse command line represented as string
	*/
	errcode parseCommandLine(std::string cmdline);
	/*
	Repeated parse of last command line (obsolete)
	*/
	errcode parseCommandLine()
	{
		return parseCommandLine(lastcmdline);
	}
	// Unset parameter after parsing (use this to exclude some parameters
	// before saving the project)
	errcode unsetParam(std::string pname);

	/*************
	Loading/saving
	*************/
	/*
	Load params from file with signature checking
	*/
	errcode loadParams(FILE *f, std::string signature="");
	/*
	Load parameters from file stream
	*/
	errcode loadParams(std::string filename, std::string signature="");

	/*
	Save parameters to file stream
	*/
	void saveParams(FILE *f, std::string signature="") const;
	/*
	Save parameters to file with specified signature
	This will convert filenames to relative! Be careful not to use
        them after saving. If you still need it, make a copy of ParamContainer
	*/
	errcode saveParams(std::string filename, std::string signature="");

	/************
	Obtain values
	************/
	/*
	Get value of specified parameter ("" if no such parameter)
	*/
	std::string &operator [](std::string pname)
	{
		if(plist.find(pname)==plist.end()) return dumbparam.value;
		return plist.find(pname)->second.value;
	}
	std::string operator [](std::string pname) const
	{
		if(plist.find(pname)==plist.end()) return "";
		return plist.find(pname)->second.value;
	}
	/*
        Get nested ParamContainer
        This function doesn't check errors. Behavior will be unpredictable if
        there's no such parameter or it's not nested.
	*/
	const ParamContainer &child(std::string pname) const
	{
		return *(plist.find(pname)->second.p);
	}

	/**************
	Help management
	**************/
	/*
        dumps help to the file stream f.
        showparamlist = obsolete, should be true
        width = number of columns in the terminal window (0 = unlimited; try not to use widths<20)
        subtopic = should be empty string
	*/
	void dumpHelp(FILE *f, bool showparamlist=true, unsigned int width=0, std::string subtopic="") const;
	/*
        helper-function to output help. May be also useful for other output.
        outputs topic to stream f with left-indent, supposing terminal window
        width is width. linebreak points whether or not put the '\n' at the end
        of output
	*/
	static void printHelpTopic(FILE *f, std::string topic, int indent, int width, bool linebreak=true);

	/*************
	Error handling
	*************/
	/*
	Get command line fragment with last error (see sample.cpp)
	*/
	std::string getErrCmdLine(int &pos, int maxlength=70) const;
	/*
	Get error message by error code
	*/
	std::string getErrorMessage(errcode err) const
	{
		std::string s=messages[err];
		int tpos=(int)s.find("%s");
		return tpos<(int)s.length()?s.substr(0, tpos)+errinfo+s.substr(tpos+2):s;
	}
};

#endif /* __PARAMCONTAINER_H */
