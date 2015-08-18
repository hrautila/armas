
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.


#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

#include <armas/armas.h>
#include "scheduler.h"

static int __armas_init_flag = 0;

static armas_conf_t __default_conf = {
  .mb = 64,     // MB (row count)
  .nb = 128,    // NB (col count)
  .kb = 160,    // KB (row/col count)
  .lb = 48,     // LB (lapack blocking size)
  .maxproc = 1, // max processors
  .wb = 480,    // WB (scheduler blocking size)
  .error = 0,   // last error
  .optflags = 0, // opt flags
  .tolmult = 10   // error tolerance multiplier (tolerance = tolmult*EPSILON)
};

static armas_scheduler_t __default_sched = {
 .workers = NULL,  // no workers;
 .cpus    = {}, //ARMAS_EMPTY_CPUSET,     // 
 .opts    = 0,
 .nworker = 0,     // no workers
 .rrindex = 0,
 .status  = 0,     // not running
 .nsched  = 0      // zero schedules
};

long armas_use_nproc(uint64_t nelems, armas_conf_t *conf)
{
  int k;
  uint64_t bsize = conf->wb * conf->wb;
  long nproc = sysconf(_SC_NPROCESSORS_ONLN);

  if (conf->maxproc > 0 && conf->maxproc < nproc) {
    nproc = conf->maxproc;
  }

  if (conf->optflags & ARMAS_BLAS_TILED) {
    // work divided to tiles of wb*wb
    return nproc <= 1 || nelems < bsize ? 1 : nproc;
  }

  // divide to atmost nproc blocks
  k = (long)ceil((double)nelems/(double)bsize);
  return nproc < k ? nproc : k;
}

int armas_nblocks(uint64_t nelems, int wb, int maxproc, int flags)
{
  int k;
  uint64_t bsize = wb * wb;
  long nproc = sysconf(_SC_NPROCESSORS_ONLN);

  if (maxproc > 0 && maxproc < nproc) {
    nproc = maxproc;
  }

  if (flags & ARMAS_BLAS_TILED) {
    // work divided to tiles of wb*wb
    return nproc <= 1 || nelems < bsize ? 1 : nproc;
  }

  // divide to atmost nproc blocks
  k = (long)ceil((double)nelems/(double)bsize);
  return nproc < k ? nproc : k;
}

/*! \brief Get default configuration.
 *
 * Returns pointer to default configuration block.
 */
armas_conf_t *armas_conf_default()
{
  armas_init();
  return &__default_conf;
}

armas_scheduler_t *armas_sched_default()
{
  armas_init();
  return &__default_sched;
}

int armas_last_error() {
  return __default_conf.error;
}

void armas_parse_scheduling(char *str, cpu_set_t *cpus, armas_conf_t *conf)
{
  char *cstr, *tok, *s0, *s1;
  int i, n, val, start, end, step, hasp = 0;
  long nproc = sysconf(_SC_NPROCESSORS_ONLN);;
  unsigned long bits;

  if (!str)
    return;

  // starts with alp
  if (isalpha(*str)) {
    switch (toupper(*str)) {
    case 'B':
      __default_conf.optflags |= ARMAS_BLAS_BLOCKED;
      break;
    case 'T':
      __default_conf.optflags |= ARMAS_BLAS_TILED;
      break;
    case 'R':
      __default_conf.optflags |= ARMAS_BLAS_RECURSIVE;
      break;
    }
    str++;
    hasp = 1;
  }

  // second policy character
  if (isalpha(*str)) {
    switch (toupper(*str)) {
    case 'R':
      __default_sched.opts |= ARMAS_SCHED_ROUNDROBIN;
      break;
    case 'Z':
      __default_sched.opts |= ARMAS_SCHED_RANDOM;
      break;
    }
    str++;
    hasp = 1;
  }

  if (isdigit(*str)) {
    if (*str == '2')
      __default_sched.opts |= ARMAS_SCHED_TWO;
    ++str;
  }

  if (hasp) {
    // expect punctiation character to separate cpu spec from policy
    if (ispunct(*str))
      ++str;
  }

  cstr = str;
  CPU_ZERO(&__default_sched.cpus);
  // parse cpu reservations
  for (n = 0, tok = strsep(&cstr, ","); tok; tok = strsep(&cstr, ","), n++) {
    // empty token
    if (! *tok)
      continue;

    // KK>0xNNN = bitmask, N = number, K-L/N = from K to L with steps of N
    if ((s0 = strchr(tok, ':')) || tolower(tok[1]) == 'x') {
      // hex mask (unsigned long) with optional starting offset
      start = 0;
      if (s0) {
	*s0 = '\0';
	start = strtol(tok, NULL, 10);
	bits = strtoul(s0+1, NULL, 0);
      } else {
	bits = strtoul(tok, NULL, 16);
      }

      // only processer number less than nproc
      for (i = 0; i < 8*sizeof(val) && start+i < nproc; i++) {
	if ((bits & (1 << i)) != 0) {
	  CPU_SET(start+i, cpus);
	}
      }
    }
    else if ((s0 = strchr(tok, '-')) != 0) {
      // parse range with optional step
      step = 1;
      if ((s1 = strchr(tok, '/')) != 0) {;
	*s1 = '\0';
	step = strtol(s1+1, NULL, 10);
      }
      *s0 = '\0';
      start = strtol(tok, NULL, 10);
      end = strtol(s0+1, NULL, 10);
      for (i = start; i <= end && i < nproc; i += step) {
	CPU_SET(i, cpus);
      }
    }
    else {
      val = strtol(tok, NULL, 10);
      if (val < nproc)
        CPU_SET(val, cpus);
    }
  }
}

#define ENV_ARMAS_CONFIG "ARMAS_CONFIG"
#define ENV_ARMAS_SCHED  "ARMAS_SCHED"

/*!
 * \brief Initialize library configuration variables
 *
 * Reads configurations from environment variables ARMAS_CONFIG and ARMAS_SCHED.
 *
 * ARMAS_CONFIG defines default blocking configuration and has format:
 * "MB,NB,KB,LB,WB,NPROC,TOLMULT".
 *
 * ARMAS_SCHED scheduling policy and list of available CPUs. It's format is
 * "[SCHEDSPEC,]CPUSPEC,CPUSPEC,..". SCHED is two character encoding of <TYPE> and <POLICY>.
 *
 * <TYPE> is either recursive (R), blocked (B) or tiled (T) scheduling. <POLICY> is either
 * round-robin (R) or random (Z) and applies only to blocked or tiled scheduling.
 *
 * CPUSPEC is one of following
 *   a) [<offset>'>']<64bit hexstring>
 *      - cpu bit mask optional <offset> is index to first cpu in mask
 *   b) <start>-<end>[/<step>]
 *      - inclusive cpu range with optional <step> 
 *   c) <index>
 *      - cpu number
 */
void armas_init()
{
  char *cstr, *tok;
  int n, val;
  long nproc;

  if (__armas_init_flag)
    return;

  // this returns number of processors, not number of cores
  nproc = sysconf(_SC_NPROCESSORS_ONLN);
  //armas_init_cpuset();

  cstr = getenv(ENV_ARMAS_CONFIG);
  // parse string: "MB,NB,KB,LB,WB,NPROC,TOLMULT"
  for (n = 0, tok = strsep(&cstr, ","); tok; tok = strsep(&cstr, ","), n++) {
    //printf("n: %d, tok: '%s', cstr: '%s'\n", n, tok, cstr);
    val = atoi(tok);
    if (val <= 0)
      continue;
    // set non-zero positive value
    switch (n) {
    case 0:
      __default_conf.mb = val;
      break;
    case 1:
      __default_conf.nb = val;
      break;
    case 2:
      __default_conf.kb = val;
      break;
    case 3:
      __default_conf.lb = val;
      break;
    case 4:
      // this should be bigger than mb or nb
      if (val < __default_conf.mb)
        val = __default_conf.mb;
      if (val < __default_conf.nb)
        val = __default_conf.nb;
      __default_conf.wb = val;
      break;
    case 5:
      __default_conf.maxproc = val > nproc ? nproc : val;
      break;
    case 6:
      __default_conf.tolmult = val;
      break;
    default:
      break;
    }
  }

  if (__default_conf.maxproc > 1) {
    // read scheduling config
    cstr = getenv(ENV_ARMAS_SCHED);
    armas_parse_scheduling(cstr, &__default_sched.cpus, &__default_conf);

    // init scheduler
    armas_sched_init(&__default_sched, __default_conf.maxproc, 64);
  }

  __armas_init_flag = 1;
}



// Local Variables:
// indent-tabs-mode: nil
// End:
