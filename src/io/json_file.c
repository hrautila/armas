
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \cond
#include <stdio.h>
//! \endcond

#include "dtype.h"
// ------------------------------------------------------------------------------
// this file provides following type independent functions
#if defined(armas_json_load)
#define ARMAS_PROVIDES 1
#endif
// this this requires no external type dependnet public functions
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// ------------------------------------------------------------------------------

//! \cond
#include "matrix.h"
//! \endcond

/**
 * @brief Initialize matrix from JSON serialized file stream
 *
 * @param[out] A
 *    On entry, uninitialized matrix. On exit deserialized matrix
 *    from JSON stream.
 * @param[in] fp
 *    Open file pointer.
 * 
 *  Reads JSON serialization of matrix from defined stream starting from
 *  current position. On exit stream is positioned at first character of JSON 
 *  serialization of matrix.
 *
 * @ingroup matrix
 */
int armas_json_load(armas_dense_t **A, FILE *fp)
{
    armas_iostream_t reader;

    armas_ios_file(&reader, fp);
    return armas_json_read(A, &reader);
}

/**
 * @brief Serialize matrix in JSON format to file stream.
 *
 * @param [out] fp Open file stream
 * @param [in]  A Matrix to serialize
 * @param [in]  flags Matrix shape flags
 *
 * @ingroup matrix
 */
int armas_json_dump(FILE *fp, const armas_dense_t *A, int flags)
{
    armas_iostream_t writer;

    armas_ios_file(&writer, fp);
    return armas_json_write(&writer, A, flags);
}
#else
#warning "Missing defines. No code."
#endif
