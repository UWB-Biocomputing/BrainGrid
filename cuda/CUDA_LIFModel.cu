#include "CUDA_LIFModel.h"
#include "MersenneTwisterCUDA.h"
#include "../tinyxml/tinyxml.h"

#include "ParseParamError.h"
#include "Util.h"


CUDA_LIFModel::CUDA_LIFModel()
{
}

CUDA_LIFModel::~CUDA_LIFModel()
{
	LIFModel::~LIFModel();
}

