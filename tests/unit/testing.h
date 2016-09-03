
#ifndef _UNIT_TESTING_H
#define _UNIT_TESTING_H 1

#include <math.h>
#include <float.h>

#if defined(FLOAT32)
#include <armas/smatrix.h>

#define _EPS FLT_EPSILON
#define _EPS2 FLT_EPSILON*FLT_EPSILON

#else

#define _EPS  DBL_EPSILON
#define _EPS2 DBL_EPSILON*DBL_EPSILON

#include <armas/dmatrix.h>


#endif

// type dependent mapping
#include "dtype.h"
#include "dlpack.h"

// helper functions
#include "helper.h"

#endif //_UNIT_TESTING_H

// Local Variables:
// indent-tabs-mode: nil
// c-basic-offset: 4
// End:
