#ifndef _PARSEPARAMERROR_H_
#define _PARSEPARAMERROR_H_

#include <iostream>

using namespace std;

class ParseParamError {
    public:
        ParseParamError(const string paramName, const string errorMessage);
        void printError(ostream &output) const;
    
    private:
        const string m_paramName;
        const string m_errorMessage;
};
