
#include <stdio.h>
#include <armas/armas.h>
#include "dtype.h"
#include "internal.h"

#ifdef __INLINE
#undef __INLINE
#endif

// define empty inline and include source

#define __INLINE

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
