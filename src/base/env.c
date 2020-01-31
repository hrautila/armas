

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#define ARMAS_PROVIDES 1
// this file requires external public functions
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

#include "armas.h"

#define CMEMSIZE 256*1024L

static struct armas_env __env_config = {
    .mb = 64,             // MB (row count)
    .nb = 64,            // NB (col count)
    .kb = 160,            // KB (row/col count)
    .lb = 64,             // LB (lapack blocking size)
    .blas1min = 1024,
    .blas2min = 256,      // Blas2 min length to recursive functions.
    .cmem = CMEMSIZE,     // default per-thread cache memory size
    .l1mem = CMEMSIZE/8,
    .fixed = 0
};

static struct armas_conf __default_conf = {
    .error = 0,           // last error (output)
    .optflags = 0,        // opt flags
    .tolmult = 10,        // error tolerance multiplier (tolerance = tolmult*EPSILON)
    .cbuf = (struct armas_cbuf *)0,
    .accel = (struct armas_accel *)0,
    .maxiter = 0,         // max iterators
    .gmres_m = 0,         //
    .numiters = 0,        // last iterations executed (output)
    .stop = 0.0,          // absolute stopping criteria
    .smult = 0.0,         // relative stopping critedia multiplier
    .residual = 0.0       // last residual computed (output)
};

static int armas_x_read_environment = 0;


#ifndef ENV_ARMAS_CONFIG
#define ENV_ARMAS_CONFIG "ARMAS_CONFIG"
#endif
#ifndef ENV_ARMAS_CACHE
#define ENV_ARMAS_CACHE  "ARMAS_CACHE"
#endif
#ifndef ENV_ARMAS_DEBUG
#define ENV_ARMAS_DEBUG  "ARMAS_DEBUG"
#endif

/**
 */
static
void armas_read_environment()
{
    char *cstr, *tok;
    int n, val;

    if (armas_x_read_environment)
        return;

    cstr = getenv(ENV_ARMAS_CONFIG);
    // parse string: "MB,NB,KB,LB,BLAS1MIN,BLAS2MIN,ISFIXED"
    for (n = 0, tok = strsep(&cstr, ","); tok; tok = strsep(&cstr, ","), n++) {
        val = atoi(tok);
        switch (n) {
        case 0:
            if (val > 0)
                __env_config.mb = val;
            break;
        case 1:
            if (val > 0)
                __env_config.nb = val;
            break;
        case 2:
            if (val > 0)
                __env_config.kb = val;
            break;
        case 3:
            if (val > 0)
                __env_config.lb = val;
            break;
        case 4:
            if (val > 0)
                __env_config.blas1min = val;
            break;
        case 5:
            if (val > 0)
                __env_config.blas2min = val;
            break;
        case 6:
        default:
            __env_config.fixed = tok && tolower(*tok) == 'y';
            break;
        }
    }
    cstr = getenv(ENV_ARMAS_DEBUG);
    if (cstr && tolower(*cstr) == 'y')
        __env_config.fixed = 1;

    // get cache memory size; L2MEM,L1MEM
    cstr = getenv(ENV_ARMAS_CACHE);
    if (cstr) {
        char *endptr;
        size_t val;
        for (n = 0, tok = strsep(&cstr, ","); tok; tok = strsep(&cstr, ","), n++) {
            endptr = (char *)0;
            val = strtoul(tok, &endptr, 0);
            if (endptr) {
                switch (toupper(*endptr)) {
                case 'K':
                    val *= 1024;
                    break;
                case 'M':
                    val *= 1024*1024;
                    break;
                default:
                    break;
                }
            }
            switch (n) {
            case 0:
                if (val > 0)
                    __env_config.cmem = val;
                break;
            case 1:
                if (val > 0)
                    __env_config.l1mem = val;
                break;
            }
        }
    }

    armas_x_read_environment = 1;
}

struct armas_env *armas_getenv()
{
    armas_read_environment();
    return &__env_config;
}

struct armas_conf *armas_conf_default()
{
    armas_read_environment();
    return &__default_conf;
}

#else
#warning "No active code"
#endif // __ARMAS_PROVIDES && __ARMAS_REQUIRES
