
#include <stdio.h>
#include "armas.h"

#include "dtype.h"
#include "matrix.h"
#include "internal.h"

#ifdef __ARMAS_INLINE
#undef __ARMAS_INLINE
#endif

// define empty inline and include source

#define __ARMAS_INLINE

#include "pivot.h"

void armas_pivot_printf(FILE *out, const char *fmt, armas_pivot_t *P)
{
    int k;
    if (!P)
        return;
    fprintf(out, "[");
    for (k = 0; k < P->npivots; k++) {
        if (k > 0)
            fprintf(out, ", ");
        fprintf(out, fmt, P->indexes[k]);
    }
    fprintf(out, "]\n");
}
