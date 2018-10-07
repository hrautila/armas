
// Copyright (c) Harri Rautila, 2018

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.


#include <stdio.h>
#include <armas/armas.h>

//! \cond
#include <stdio.h>

#include "dtype.h"
//! \endcond
// ------------------------------------------------------------------------------
// this file provides following type independent functions
#if defined(armas_x_json_load)
#define __ARMAS_PROVIDES 1
#endif
// this this requires no external type dependnet public functions
#define __ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

//! \cond
#include "matrix.h"
//! \endcond

static
int file_getc(void *ptr)
{
    FILE *fp = (FILE *)ptr;
    return fgetc(fp);
}

static
void file_ungetc(void *ptr, int c)
{
    FILE *fp = (FILE *)ptr;
    ungetc(c, fp);
}

static
int file_putc(void *ptr, int c)
{
    FILE *fp = (FILE *)ptr;
    return fputc(c, fp);
}

static
armas_iostream_vtable_t file_vtable = (armas_iostream_vtable_t){
    .get_char = file_getc,
    .unget_char = file_ungetc,
    .put_char = file_putc
};

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
 */
int armas_x_json_load(armas_x_dense_t *A, FILE *fp)
{
    armas_iostream_t reader;

    armas_iostream_init(&reader, &file_vtable, fp);
    return armas_x_json_read(A, &reader);
}


int armas_x_json_dump(FILE *fp, const armas_x_dense_t *A, int flags)
{
    armas_iostream_t writer;

    armas_iostream_init(&writer, &file_vtable, fp);
    return armas_x_json_write(&writer, A, flags);
}



#endif

// Local Variables:
// indent-tabs-mode: nil
// c-basic-offset: 4
// End:
