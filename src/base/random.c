
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "dtype.h"
// ------------------------------------------------------------------------------
// this file provides following type independent functions
#if defined(armas_uniform) && defined(armas_normal) && defined(armas_uniform_at) \
    && defined(armas_normal_at)
#define ARMAS_PROVIDES 1
#endif
// this this requires no external public functions
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"

static long _seed_value = 0;

static inline
void init_seed()
{
    if (!_seed_value) {
        _seed_value = (long) time(0);
        srand48(_seed_value);
    }
}

static
DTYPE unitrand(int i, int j)
{
    init_seed();
    return (DTYPE) drand48();
}

// std random variable from unit random with box-muller transformation
static
DTYPE stdrand(int i, int j)
{
    double s, u, v;
    init_seed();
    do {
        u = 2.0 * drand48() - 1.0;
        v = 2.0 * drand48() - 1.0;
        s = u * u + v * v;
    } while (s == 0.0 || s >= 1.0);
    return (DTYPE) (u * sqrt(-2.0 * log(s) / s));
}

/**
 * @brief Set seed value for pseudo-random number generator.
 *
 * @param seed
 *    New seed value. If value is zero current seed value is returned and new
 *    seed is not set.
 *
 * @returns
 *    Current seed if seed value is zero otherwise the new seed value.a
 */
long armas_seed(long seed)
{
    if (!seed)
        return _seed_value;

    _seed_value = seed;
    srand48(seed);
    return seed;
}

/**
 * @brief Return a value from uniform distrubution [0.0, 1.0)
 */
DTYPE armas_uniform_at(int r, int c)
{
    return unitrand(r, c);
}

/**
 * @brief Return a value from uniform distrubution [0.0, 1.0)
 */
DTYPE armas_uniform(void)
{
    return unitrand(0, 0);
}

/**
 * @brief Return a value from normal distribution.
 */
DTYPE armas_normal_at(int r, int c)
{
    return stdrand(r, c);
}

/**
 * @brief Return a value from normal distribution.
 */
DTYPE armas_normal(void)
{
    return stdrand(0, 0);
}

#else
#warning "Missing type names! No code."
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
