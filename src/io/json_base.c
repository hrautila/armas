
// Copyright (c) Harri Rautila, 2018-2020

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
// ----------------------------------------------------------------------------
// this file provides following type independent functions
#if defined(armas_x_json_write) && defined(armas_x_json_read)
#define ARMAS_PROVIDES 1
#endif
// this this requires no external type dependent public functions
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ----------------------------------------------------------------------------

//! @cond
#include <stdio.h>
#include "matrix.h"
//! @endcond


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
 * @retval   0 Success
 * @retval  <0 Error
 *
 * @ingroup matrix
 */
int armas_x_json_write(armas_iostream_t *ios, const armas_x_dense_t *A, int flags)
{
    if (!ios)
        return -1;

    if (!A) {
        JSON_ONERROR(armas_json_write_simple_token(ARMAS_JSON_NULL, ios));
        return 0;
    }

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
 *  current position. On exit stream is positioned at first character after the JSON 
 *  serialization of matrix.
 *
 * TODO: null matrix? is it: null | "{}"? Is it same as {"rows":0, "cols":0,...}
 * Should we return pointer to new matrix? Maybe A is armas_x_dense_t ** and
 * if *A == null we allocate, otherwise we deserialize into provided space.
 * 
 * @retval  0  Success
 * @retval <0  Failure
 *
 * @ingroup matrix
 */
int armas_x_json_read(armas_x_dense_t **A, armas_iostream_t *ios)
{
    char iob[64];
    int tok;
    int member_bits = 0;
    int state = MEMBER_KEY;
    int rows, cols, nnz, flags, ntok;
    armas_x_dense_t *aa;

    rows = cols = nnz = flags = 0;

    aa = *A;
    tok = armas_json_read_token(iob, sizeof(iob), ios);
    if (tok == ARMAS_JSON_NULL) {
        // if pointer to matrix provided set it to null.
        if (aa)
            *A = (armas_x_dense_t *)0;
        return 0;
    }
    if (tok != '{') {
        return -1;
    }
    // expect: rows|cols|flags|nnz ':' val
    for (ntok = 0; state != MEMBER_DTA_VAL ; ntok++) {
        tok = armas_json_read_token(iob, sizeof(iob), ios);
        switch (state) {
        case MEMBER_KEY:
            if (tok != ARMAS_JSON_STRING) {
                if (ntok == 0 && tok == '}') {
                    // we have null matrix
                    if (aa)
                        *A = (armas_x_dense_t *)0;
                    return 0;
                }
                return -1;
            }
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
            if (tok != ARMAS_JSON_NUMBER)
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
            if (tok != ARMAS_JSON_NUMBER)
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
            if (tok != ARMAS_JSON_NUMBER)
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
            if (tok != ARMAS_JSON_NUMBER)
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

    int have_new = aa ? 0 : 1;
    if (have_new) {
        aa = (armas_x_dense_t *)calloc(1, sizeof(armas_x_dense_t));
        if (!aa)
            return -1;
    }
    armas_x_init(aa, rows, cols);

    int n = 0;
    double dval;
    for (int j = 0; j < cols; j++) {
        for (int i = 0; i < rows; i++, n++) {
            if (n > 0) {
                if ((tok = armas_json_read_token(iob, sizeof(iob), ios)) != ',')
                    goto error_exit;
            }
            tok = armas_json_read_token(iob, sizeof(iob), ios);
            if (tok != ARMAS_JSON_NUMBER && tok != ARMAS_JSON_INT)
                goto error_exit;
            switch (tok) {
            case ARMAS_JSON_INT:
                dval = (double)atoi(iob);
                break;
            default:
                dval = strtod(iob, (char **)0);
                break;
            }
            armas_x_set_unsafe(aa, i, j, dval);
        }
    }
    // end of array
    if (armas_json_read_token(iob, sizeof(iob), ios) != ']')
        goto error_exit;

    // end of object
    if (armas_json_read_token(iob, sizeof(iob), ios) == '}') {
        // return the new matrix to caller
        if (have_new)
            *A = aa;
        return 0;
    }

 error_exit:
    // release reserved space
    if (have_new)
        armas_x_free(aa);
    else
        armas_x_release(aa);
    return -1;
}
#else
#warning "Missing defines. No code"
#endif
