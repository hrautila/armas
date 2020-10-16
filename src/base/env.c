
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

#include "armas.h"

/**
 * @file
 * Environment functions.
 * @addtogroup other
 * @{
 */

#define CMEMSIZE 256*1024L
#define CMEM_MIN 4096
#define L1MEM_MIN 1024

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
    .work = (struct armas_wbuf *)0,
    .accel = (struct armas_accel *)0,
    .maxiter = 0,         // max iterators
    .gmres_m = 0,         //
    .numiters = 0,        // last iterations executed (output)
    .stop = 0.0,          // absolute stopping criteria
    .smult = 0.0,         // relative stopping critedia multiplier
    .residual = 0.0       // last residual computed (output)
};

static int has_read_environment = 0;


#ifndef ENV_ARMAS_CONFIG
#define ENV_ARMAS_CONFIG "ARMAS_CONFIG"
#endif
#ifndef ENV_ARMAS_CACHE
#define ENV_ARMAS_CACHE  "ARMAS_CACHE"
#endif
#ifndef ENV_ARMAS_DEBUG
#define ENV_ARMAS_DEBUG  "ARMAS_DEBUG"
#endif

static
void armas_read_environment()
{
    char *cstr, *tok;
    int n, val;

    if (has_read_environment)
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
                    __env_config.cmem =  val < CMEM_MIN ? CMEM_MIN : val;;
                break;
            case 1:
                if (val > 0)
                    __env_config.l1mem =  val < L1MEM_MIN ? L1MEM_MIN : val;;
                break;
            }
        }
    }

    has_read_environment = 1;
}

/**
 * @brief Get blocking configurations.
 *
 * @return Pointer to blocking and memory configuration.
 */
struct armas_env *armas_getenv()
{
    armas_read_environment();
    return &__env_config;
}

/**
 * @brief Get default configuration block.
 *
 * Library global configuration structure. This is used whenever null pointer to configuration block is
 * provided as function argument.
 */
struct armas_conf *armas_conf_default()
{
    armas_read_environment();
    return &__default_conf;
}

//! @}
