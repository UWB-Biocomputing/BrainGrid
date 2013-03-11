#include "ParseParamError.h"

ParseParamError::ParseParamError(const string paramName, const string errorMessage) :
     m_paramName(paramName)
    ,m_errorMessage(errorMessage)
{
    // Constructor
}

void ParseParamError::print(ostream &output) const
{
    output << "ERROR :: Failed to parse parameter \"" << m_paramName << "\".";
    output << " Cause: " << m_errorMessage;
}
