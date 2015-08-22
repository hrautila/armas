
// Copyright (c) Harri Rautila, 2015

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef __AVXFUNCS_H
#define __AVXFUNCS_H 1

#include <immintrin.h>

/*
 * Horizontal sum if 4 double precision floats (in 4 instructions)
 */
static inline
double hsum256_f64(__m256d A)
{
    register __m256d B;
    B  = _mm256_hadd_pd(A, A);
    B += _mm256_permute2f128_pd(B, B, 1);
    return B[0];
}

/*
 * Assume A > B > C > D
 *
 * 1. swap low/high 128bit and compare
 *       {ABCD} : {CDAB} = {ABAB}
 * 2. swap 64 bit values pairwise and compare
 *       {ABAB} : {BABA} = {AAAA}
 */

static inline
double max256_f64(__m256d A)
{
    register __m256d B;
    B = _mm256_max_pd(A, _mm256_permute2f128_pd(A, A, 1));
    B = _mm256_max_pd(B, _mm256_permute_pd(B, 0b0101));
    return B[0];
}

static inline
double min256_f64(__m256d A)
{
    register __m256d B;
    B = _mm256_min_pd(A, _mm256_permute2f128_pd(A, A, 1));
    B = _mm256_min_pd(B, _mm256_permute_pd(B, 0b0101));
    return B[0];
}

/*
 * Horizontal sum of 256bit packed single precision floats (8)
 */
static inline
float hsum256_f32(__m256 A)
{
    register __m256 B;
    B  = _mm256_hadd_ps(A, A);
    B += _mm256_permute2f128_ps(B, B, 1);
    B  = _mm256_hadd_ps(B, B);
    return B[0];
}

/*
 * Assume A > B > ... > G > H
 *
 * 1. swap low/high 128bit and compare
 *       {ABCDEFGH} : {EFGHABCD} = {ABCDACBD}
 * 2. swap 32 values pairwise and compare
 *       {ABCDABCD} : {BADCBACD} = {AACCAACC}
 * 3. swap 64 bits pairwise and compare
 *       {AACCAACC} : {CCAACCAA} = {AAAAAAAA}
 */

/**
 * Max of packed single precision vector
 */
static inline
float max256_f32(__m256 A)
{
    register __m256 B;
    B = _mm256_max_ps(A, _mm256_permute2f128_ps(A, A, 1));
    B = _mm256_max_ps(B, _mm256_permute_ps(B, 0b10110001));
    B = _mm256_max_ps(B, _mm256_permute_ps(B, 0b00011011));
    return B[0];
}

/**
 * Min of packed single precision vector
 */
static inline
float min256_f32(__m256 A)
{
    register __m256 B;
    B = _mm256_min_ps(A, _mm256_permute2f128_ps(A, A, 1));
    B = _mm256_min_ps(B, _mm256_permute_ps(B, 0b10110001));
    B = _mm256_min_ps(B, _mm256_permute_ps(B, 0b00011011));
    return B[0];
}

#endif

// Local Variables:
// indent-tabs-mode: nil
// c-basic-offset: 4
// End:
