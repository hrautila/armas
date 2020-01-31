

#ifndef __ARMAS_DEVICE_H
#define __ARMAS_DEVICE_H 1

#include <stddef.h>
#include "dtype.h"
#include "matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

    typedef struct armas_device {
        void *vptr;     ///< Virtual function table
        int  flags;     ///< Device type flags
    } armas_device_t;

    typedef struct armas_x_vptr {
        int (*gemm)(armas_device_t *dev, DTYPE beta, armas_x_dense_t *C, DTYPE alpha,
                    const armas_x_dense_t *A, const armas_x_dense_t *B, int K, int flags,
                    armas_conf_t *cf);
    } armas_x_vptr_t;

    int armas_x_mult_tlocal(armas_device_t *dev, DTYPE beta, armas_x_dense_t *C, DTYPE alpha,
                            const armas_x_dense_t *A, const armas_x_dense_t *B, int K, int flags,
                            armas_conf_t *cf);

#define ARMAS_DISPATCH(dev, func)   (*(((armas_x_vptr_t *)dev->vptr)->func))

#define ARMAS_FUNC(dev, name) (((armas_x_vptr_t *)dev->vptr)->name)

#ifdef __cplusplus
}
#endif

#endif  // __ARMAS_DEVICE_H
