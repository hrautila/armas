
#ifndef ARMAS_ACCEL_H
#define ARMAS_ACCEL_H 1

#include <stdint.h>
#include "dtype.h"
#include "matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

enum armas_ac_opcodes {
    ARMAS_AC_GEMM = 1,
    ARMAS_AC_TRMM = 2,
    ARMAS_AC_TRSM = 3,
    ARMAS_AC_SYMM = 4,
    ARMAS_AC_GEMV = 5,
    ARMAS_AC_TRSV = 6,
    ARMAS_AC_TRMV = 7,
    ARMAS_AC_DOT = 8,
    ARMAS_AC_AXPBY = 9,
    ARMAS_AC_NRM2 = 10
};

#define ARMAS_AC_TYPEMASK  0xFFFFL

enum armas_ac_types {
    ARMAS_AC_BLAS1 = 0x1 << 16,
    ARMAS_AC_BLAS2 = 0x2 << 16,
    ARMAS_AC_BLAS3 = 0x3 << 16
};

struct armas_ac_blas1 {
    size_t tag;
    DTYPE result;
    DTYPE beta;
    DTYPE alpha;
    struct armas_x_dense *y;
    const struct armas_x_dense *x;
    int flags;
};

struct armas_ac_blas2 {
    size_t tag;
    DTYPE beta;
    struct armas_x_dense *y;
    DTYPE alpha;
    const struct armas_x_dense *A;
    const struct armas_x_dense *x;
    int flags;
    char fill[8];
};

struct armas_ac_blas3 {
    size_t tag;
    DTYPE beta;
    struct armas_x_dense *C;
    DTYPE alpha;
    const struct armas_x_dense *A;
    const struct armas_x_dense *B;
    int flags;
    char fill[16];
};


static inline
size_t armas_ac_blas2_tag()
{
    return (sizeof(struct armas_ac_blas2) | ARMAS_AC_BLAS2);
}

static inline
size_t armas_ac_blas3_tag()
{
    return (sizeof(struct armas_ac_blas3) | ARMAS_AC_BLAS3);
}

static inline
int armas_ac_test_tag(const void *args, enum armas_ac_types arg_type)
{
    size_t *tag = (size_t *)args;
    if ((*tag & arg_type) != arg_type)
        return 0;

    switch (arg_type) {
    case ARMAS_AC_BLAS1:
        return (*tag & ARMAS_AC_TYPEMASK) == sizeof(struct armas_ac_blas1);
    case ARMAS_AC_BLAS2:
        return (*tag & ARMAS_AC_TYPEMASK) == sizeof(struct armas_ac_blas2);
    case ARMAS_AC_BLAS3:
        return (*tag & ARMAS_AC_TYPEMASK) == sizeof(struct armas_ac_blas3);
    default:
        break;
    }
    return 0;
}

static inline
void armas_ac_set_blas2_args(struct armas_ac_blas2 *args, DTYPE beta, armas_x_dense_t *y,
                             DTYPE alpha, const armas_x_dense_t *A, const armas_x_dense_t *x, int flags)
{
    args->tag = armas_ac_blas2_tag();
    args->alpha = alpha;
    args->beta = beta;
    args->x = x;
    args->y = y;
    args->A = A;
    args->flags = flags;
}

static inline
void armas_ac_set_blas3_args(struct armas_ac_blas3 *args, DTYPE beta, armas_x_dense_t *C,
                             DTYPE alpha, const armas_x_dense_t *A, const armas_x_dense_t *B, int flags)
{
    args->tag = armas_ac_blas3_tag();
    args->alpha = alpha;
    args->beta = beta;
    args->C = C;
    args->B = B;
    args->A = A;
    args->flags = flags;
}


#ifdef __cplusplus
}
#endif

#endif  // ARMAS_ACCEL_H
