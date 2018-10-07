
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
#if defined(armas_x_json_write) && defined(armas_x_json_read)
#define __ARMAS_PROVIDES 1
#endif
// this this requires no external type dependent public functions
#define __ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

//! \cond
#include "matrix.h"
//! \endcond


#define JSON_ONERROR(func)                        \
    do { if ((func) < 0) { return -1; } } while (0);

/**
 * @brief Serialize matrix as JSON stream.
 *
 * @param[out] ios
 *     Simple I/O stream to receive JSON serialization.
 * @param[in]  A
 *     Matrix to serialize.
 * @param[in] flags
 *     Serialization control bits. 
 *
 * 
 */
int armas_x_json_write(armas_iostream_t *ios, const armas_x_dense_t *A, int flags)
{
    if (!A || !ios)
        return -1;

    JSON_ONERROR(armas_json_write_simple_token('{', ios));
    JSON_ONERROR(armas_json_write_token(ARMAS_JSON_STRING, "rows", 4, ios));
    JSON_ONERROR(armas_json_write_simple_token(':', ios));
    JSON_ONERROR(armas_json_write_token(ARMAS_JSON_INT, &A->rows, sizeof(A->rows), ios));
    JSON_ONERROR(armas_json_write_simple_token(',', ios));

    JSON_ONERROR(armas_json_write_token(ARMAS_JSON_STRING, "cols", 4, ios));
    JSON_ONERROR(armas_json_write_simple_token(':', ios));
    JSON_ONERROR(armas_json_write_token(ARMAS_JSON_INT, &A->cols, sizeof(A->cols), ios));
    JSON_ONERROR(armas_json_write_simple_token(',', ios));

    if (flags != 0) {
        JSON_ONERROR(armas_json_write_token(ARMAS_JSON_STRING, "flags", 5, ios));
        JSON_ONERROR(armas_json_write_simple_token(':', ios));
        JSON_ONERROR(armas_json_write_token(ARMAS_JSON_INT, &flags, sizeof(flags), ios));
        JSON_ONERROR(armas_json_write_simple_token(',', ios));
        // nnz here if write only lower triangular part ...
    }

    JSON_ONERROR(armas_json_write_token(ARMAS_JSON_STRING, "data", 4, ios));
    JSON_ONERROR(armas_json_write_simple_token(':', ios));
    JSON_ONERROR(armas_json_write_simple_token('[', ios));

    
    int n = 0;
    for (int j = 0; j < A->cols; j++) {
        for (int i = 0; i < A->rows; i++, n++) {
            if (n > 0)
                JSON_ONERROR(armas_json_write_simple_token(',', ios));

            double val = armas_x_get_unsafe(A, i, j);
            JSON_ONERROR(armas_json_write_token(ARMAS_JSON_NUMBER, &val, sizeof(val), ios));
        }
    }

    JSON_ONERROR(armas_json_write_simple_token(']', ios));
    JSON_ONERROR(armas_json_write_simple_token('}', ios));

    return 0;
}



enum {
    HAVE_ROWS = 1,
    HAVE_COLS = 2,
    HAVE_FLAGS = 4,
    HAVE_NNZ = 8,
    MEMBER_KEY = 100,
    MEMBER_SEP,
    MEMBER_ROW_SEP,
    MEMBER_COL_SEP,
    MEMBER_NNZ_SEP,
    MEMBER_FLG_SEP,
    MEMBER_DTA_SEP,
    MEMBER_ROW_VAL,
    MEMBER_COL_VAL,
    MEMBER_NNZ_VAL,
    MEMBER_FLG_VAL,
    MEMBER_DTA_VAL
};

/**
 * @brief Read and initialize matrix from JSON serialized stream.
 *
 * @param[out] A
 *    On entry, uninitialized matrix. On exit deserialized matrix
 *    from JSON stream.
 * @param[in] ios
 *    Simple JSON stream.
 * 
 *  Reads JSON serialization of matrix from defined stream starting from
 *  current position. On exit stream is positioned at first character of JSON 
 *  serialization of matrix.
 *
 */
int armas_x_json_read(armas_x_dense_t *A, armas_iostream_t *ios)
{
    char iob[64];
    int tok;
    int member_bits = 0;
    int state = MEMBER_KEY;
    int rows, cols, nnz, flags;

    rows = cols = nnz = flags = 0;
    
    tok = armas_json_read_token(iob, sizeof(iob), ios);
    if (tok != '{') {
        return -1;
    }
    // expect: rows|cols|flags|nnz ':' val
    for (; state != MEMBER_DTA_VAL ;) {
        tok = armas_json_read_token(iob, sizeof(iob), ios);
        switch (state) {
        case MEMBER_KEY:
            if (tok != ARMAS_JSON_STRING)
                return -1;
            if (strncmp(iob, "rows", 4) == 0) {
                if ((member_bits & HAVE_ROWS) != 0)
                    return -4;
                state = MEMBER_ROW_SEP;
            }
            else if (strncmp(iob, "cols", 4) == 0) {
                if ((member_bits & HAVE_COLS) != 0)
                    return -4;
                state = MEMBER_COL_SEP;
            }
            else if (strncmp(iob, "data", 4) == 0) {
                state = MEMBER_DTA_SEP;
            }
            else if (strncmp(iob, "nnz", 3) == 0) {
                if ((member_bits & HAVE_NNZ) != 0)
                    return -4;
                state = MEMBER_NNZ_SEP;
            }
            else if (strncmp(iob, "flags", 5) == 0) {
                if ((member_bits & HAVE_FLAGS) != 0)
                    return -4;
                state = MEMBER_FLG_SEP;
            }
            else {
                // unexpected member
                return -2;
            }
            break;

        case MEMBER_SEP:
            if (tok != ',')
                return -1;
            state = MEMBER_KEY;
            break;
            
        case MEMBER_ROW_SEP:
            if (tok != ':')
                return -1;
            state = MEMBER_ROW_VAL;
            break;
        case MEMBER_ROW_VAL:
            if (tok != ARMAS_JSON_INT)
                return -1;
            rows = atoi(iob);
            member_bits |= HAVE_ROWS;
            state = MEMBER_SEP;
            break;

        case MEMBER_COL_SEP:
            if (tok != ':')
                return -1;
            state = MEMBER_COL_VAL;
            break;
        case MEMBER_COL_VAL:
            if (tok != ARMAS_JSON_INT)
                return -1;
            cols = atoi(iob);
            member_bits |= HAVE_COLS;
            state = MEMBER_SEP;
            break;

        case MEMBER_NNZ_SEP:
            if (tok != ':')
                return -1;
            state = MEMBER_NNZ_VAL;
            break;
        case MEMBER_NNZ_VAL:
            if (tok != ARMAS_JSON_INT)
                return -1;
            nnz = atoi(iob);
            member_bits |= HAVE_NNZ;
            state = MEMBER_SEP;
            break;

        case MEMBER_FLG_SEP:
            if (tok != ':')
                return -1;
            state = MEMBER_FLG_VAL;
            break;
        case MEMBER_FLG_VAL:
            if (tok != ARMAS_JSON_INT)
                return -1;
            flags = atoi(iob);
            member_bits |= HAVE_FLAGS;
            state = MEMBER_SEP;
            break;

        case MEMBER_DTA_SEP:
            if (tok != ':')
                return -1;           
            state = MEMBER_DTA_VAL;
            break;

        default:
            break;
        }
    }
    // here we need to have at least rows and cols
    if ((member_bits & (HAVE_ROWS|HAVE_COLS)) != (HAVE_ROWS|HAVE_COLS)) {
        return -5;
    }
    // for the time being; ignore flags and nnz
    // here read next data elements;
    tok = armas_json_read_token(iob, sizeof(iob), ios);
    if (tok != '[') {
        return -1;
    }

    armas_x_init(A, rows, cols);
    int n = 0;
    double dval;
    for (int j = 0; j < cols; j++) {
        for (int i = 0; i < rows; i++, n++) {
            if (n > 0) {
                if ((tok = armas_json_read_token(iob, sizeof(iob), ios)) != ',') 
                    return -1;
            }
            tok = armas_json_read_token(iob, sizeof(iob), ios);
            if (tok != ARMAS_JSON_NUMBER && tok != ARMAS_JSON_INT)
                return -1;
            switch (tok) {
            case ARMAS_JSON_INT:
                dval = (double)atoi(iob);
                break;
            default:
                dval = strtod(iob, (char **)0);
                break;
            }
            armas_x_set_unsafe(A, i, j, dval);
        }
    }
    // end of array
    if (armas_json_read_token(iob, sizeof(iob), ios) != ']')
        return -1;
    // end of object
    if (armas_json_read_token(iob, sizeof(iob), ios) != '}')
        return -1;
    return 0;
}




#endif

// Local Variables:
// indent-tabs-mode: nil
// c-basic-offset: 4
// End:
