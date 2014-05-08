
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.


#include <stdlib.h>
#include <unistd.h>

#include <armas/armas.h>


static uint64_t __nproc_table[MAX_CPU+1] = {200*200, 300*300, 400*400, 500*500, 0};
static long __nproc = 1;

static armas_conf_t __default_conf = {
  64,   // MB (row count)
  128,  // NB (col count)
  160,  // KB (row/col count)
  36,   // LB (lapack blocking size)
  1,    // max processors
  0,    // last error
  0     // opt flags
};

// Set cpu allocation schedule
void armas_nproc_schedule(uint64_t *nproc_schedule, int count)
{
  int k;
  for (k = 0; k < count && k < MAX_CPU; k++) {
    __nproc_table[k] = nproc_schedule[k];
  }
  __nproc_table[k] = 0;
}

long armas_use_nproc(uint64_t nelems, armas_conf_t *conf)
{
  int k;
  long nproc = sysconf(_SC_NPROCESSORS_ONLN);

  if (conf->maxproc > 0 && conf->maxproc < nproc) {
    nproc = conf->maxproc;
  }

  for (k = 0; k < MAX_CPU; k++) {
    if (nelems < __nproc_table[k])
      break;
    // last entry
    if (__nproc_table[k] == 0)
      break;
    
  }
  return nproc < k+1 ? nproc : k+1;
}

/*! \brief Get default configuration.
 *
 * Returns pointer to default configuration block.
 */
armas_conf_t *armas_conf_default() {
  return &__default_conf;
}

int armas_last_error() {
  return __default_conf.error;
}


// Local Variables:
// indent-tabs-mode: nil
// End:
