
#ifndef ARMAS_ACCEL_H
#define ARMAS_ACCEL_H 1

#include <stdint.h>
#define _GNU_SOURCE
#include <sched.h>

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

/**
 * @brief Accelerator virtual API functions.
 */
struct armas_ac_vtable {
    int (*dispatch)(int opcode, void *args, struct armas_conf *cf, void *private);
    int (*release)(void *private);
};

/**
 * @brief Accelerator configuration parameters
 */
struct armas_ac_env {
    size_t max_cores;
    size_t num_items;
    double weight;
    int options;
    cpu_set_t cpus;
};

struct armas_ac_env *armas_ac_getenv();

/**
 * @brief Accelerator object
 */
typedef struct armas_accel {
    /* Accelerator API vtable */
    struct armas_ac_vtable *vptr;
    /* Implementation private object */
    void *private;
    /* Dynamic link library handle for external accelerators. */
    void *handle;
} armas_accel_t;

struct armas_ac_blas1;
struct armas_ac_blas2;
struct armas_ac_blas3;

union armas_ac_args {
    struct armas_ac_blas1 *blas1;
    struct armas_ac_blas2 *blas2;
    struct armas_ac_blas3 *blas3;
};

struct armas_ac_block {
    int row;
    int column;
    int nrows;
    int ncolumns;
    int block_index;
    int is_last;
    int error;
    union armas_ac_args u;
};

/**
 *  Compute the start index if n'th block in ntotal items divided to nblocks.
 */
static inline
int armas_ac_block_index(int n, int nblocks, int ntotal)
{
    if (n == nblocks)
        return ntotal;
    return (n*ntotal/nblocks) - ((n*ntotal/nblocks) & 0x3);
}

#ifdef ARMAS_DTYPE_H
/**
 */
struct armas_ac_blas1 {
    size_t tag;
    DTYPE result;
    DTYPE beta;
    DTYPE alpha;
    struct armas_x_dense *y;
    const struct armas_x_dense *x;
    int flags;
};

/**
 */
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

/**
 */
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
#endif /* ARMAS_DTYPE_DEFINED */

#ifdef __cplusplus
}
#endif
#endif  /* ARMAS_ACCEL_H */
