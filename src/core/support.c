
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

//! \file
//! Library support functions

//! \cond
#if HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

#include <armas/armas.h>
#include "scheduler.h"

#define CMEMSIZE 384*1024L
//! \endcond

static int armas_x_init_flag = 0;

static armas_conf_t __default_conf = {
  .mb = 64,             // MB (row count)
  .nb = 128,            // NB (col count)
  .kb = 160,            // KB (row/col count)
  .lb = 48,             // LB (lapack blocking size)
  .maxproc = 1,         // max processors
  .wb = 480,            // WB (scheduler blocking size)
  .error = 0,           // last error
  .optflags = 0,        // opt flags
  .tolmult = 10,        // error tolerance multiplier (tolerance = tolmult*EPSILON)
  .cmem = CMEMSIZE,     // default per-thread cache memory size
  .l1mem = CMEMSIZE/4,
  .cbuf = (armas_cbuf_t *)0
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

static armas_cbuf_t __cache_mem = (armas_cbuf_t){
  .data = (char *)0,
  .len = 0,
  .__unaligned = (char *)0,
  .__nbytes = 0
};


long armas_use_nproc(uint64_t nelems, armas_conf_t *conf)
{
#if defined(ENABLE_THREADS)
  int k;
  uint64_t bsize = conf->wb * conf->wb;
  long nproc = sysconf(_SC_NPROCESSORS_ONLN);

  if (conf->maxproc > 0 && conf->maxproc < nproc) {
    nproc = conf->maxproc;
  }

  if (conf->optflags & ARMAS_OBLAS_TILED) {
    // work divided to tiles of wb*wb
    return nproc <= 1 || nelems < bsize ? 1 : nproc;
  }

  // divide to atmost nproc blocks
  k = (long)ceil((double)nelems/(double)bsize);
  return nproc < k ? nproc : k;
#else
  return 1;
#endif  // ENABLE_THREADS
}

int armas_nblocks(uint64_t nelems, int wb, int maxproc, int flags)
{
#if defined(ENABLE_THREADS)
  int k;
  uint64_t bsize = wb * wb;
  long nproc = sysconf(_SC_NPROCESSORS_ONLN);

  if (maxproc > 0 && maxproc < nproc) {
    nproc = maxproc;
  }

  if (flags & ARMAS_OBLAS_TILED) {
    // work divided to tiles of wb*wb
    return nproc <= 1 || nelems < bsize ? 1 : nproc;
  }

  // divide to atmost nproc blocks
  k = (long)ceil((double)nelems/(double)bsize);
  return nproc < k ? nproc : k;
#else
  return 1;
#endif
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

armas_cbuf_t *armas_cbuf_get(armas_conf_t *conf)
{
  return conf && conf->cbuf ? conf->cbuf : armas_cbuf_default();
}

armas_cbuf_t *armas_cbuf_default(void)
{
  armas_init();
  if (__cache_mem.__nbytes == 0) {
    armas_cbuf_init(&__cache_mem, __default_conf.cmem, __default_conf.l1mem);
  }
  return &__cache_mem;
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
      __default_conf.optflags |= ARMAS_OBLAS_BLOCKED;
      break;
    case 'T':
      __default_conf.optflags |= ARMAS_OBLAS_TILED;
      break;
    case 'R':
      __default_conf.optflags |= ARMAS_OBLAS_RECURSIVE;
      break;
    }
    str++;
    hasp = 1;
  }

  // second policy character
  if (isalpha(*str)) {
    switch (toupper(*str)) {
    case 'R':
      __default_sched.opts |= ARMAS_OSCHED_ROUNDROBIN;
      break;
    case 'Z':
      __default_sched.opts |= ARMAS_OSCHED_RANDOM;
      break;
    }
    str++;
    hasp = 1;
  }

  if (isdigit(*str)) {
    if (*str == '2')
      __default_sched.opts |= ARMAS_OSCHED_TWO;
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

    // KK:0xNNN = bitmask, N = number, K-L/N = from K to L with steps of N
    if ((s0 = strchr(tok, ':')) || tolower(tok[1]) == 'x') {
      // hex mask (unsigned long) with optional starting offset
      start = 0;
      if (s0) {
        // here is: kk:mask
	*s0 = '\0';
	start = strtol(tok, NULL, 10);
	bits = strtoul(s0+1, NULL, 0);
      } else {
        // here 0xmask
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
#define ENV_ARMAS_CACHE  "ARMAS_CACHE"
#define ENV_ARMAS_SCHED  "ARMAS_SCHED"

/*!
 * \brief Initialize library configuration variables
 *
 * Reads configurations from environment variables _ARMAS_CONFIG_ and _ARMAS_SCHED_.
 *
 * *ARMAS_CONFIG* defines the internal blocking parameters and has format:
 * "MB,NB,KB,LB,WB,NPROC,TOLMULT". For matrix-matrix multiplication (gemm) _MB_ is
 * the number of row of A, C matrices, _NB_ is number of columns of B, C matrices
 * and _KB_ is the number of A columns and B rows. The parameter _LB_ defines the
 * blocking factor for _LAPACK_ functions. _WB_ is to compute the number of threads
 * needed or in _tiled_ scheduling the operation is divided to tasks of _WB_,_WB_ blocks.
 *
 * *ARMAS_SCHED* defines scheduling policy and list of available CPUs. It's format is
 * "[SCHEDSPEC,]CPUSPEC,CPUSPEC,..". _SCHED_ is two character encoding of 
 * \<TYPE> and \<POLICY>.
 * \<TYPE> is either recursive (R), blocked (B) or tiled (T) scheduling. \<POLICY> is
 *  either round-robin (R) or random (Z) and applies only to blocked or tiled scheduling.
 *
 * _CPUSPEC_ is one of following
 *   - [\<offset>':']\<64bit hexstring>
 *      - cpu bit mask, optional \<offset> is index to first cpu in mask
 *   - \<start>-\<end>[/\<step>]
 *      - inclusive cpu range with optional \<step> 
 *   - \<index>
 *      - cpu number
 *
 * _ARMAS_CACHE_ defines size of internal per thread cache memory sizes, _Cmem_ and _l1mem_.
 * The _Cmem_ is the maximum memory available for blas3 function for internal caching.
 * Functions compute the effective blocking factors by dividing this cache memory to internal
 * buffers as needed and holding relative sizes of blocking parameter _MB_, _NB_ and _KB_ 
 * fixed. The _L1mem_ defines a limit for accessing the inner most matrix in kernel 
 * functions. Typical value for _Cmem_ is 512k and for _L1mem_ is 96k.
 *
 */
void armas_init()
{
  char *cstr, *tok;
  int n, val;
  long nproc;

  if (armas_x_init_flag)
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
#if defined(ENABLE_THREADS)
      __default_conf.maxproc = val > nproc ? nproc : val;
#else
      // use nproc to kill 'unused-variable' warning
      __default_conf.maxproc = nproc != 1 ? 1 : 1;
#endif
      break;
    case 6:
      __default_conf.tolmult = val;
      break;
    default:
      break;
    }
  }

  // get cache memory size; L2MEM,L1MEM
  cstr = getenv(ENV_ARMAS_CACHE);
  if (cstr) {
    char *endptr;
    size_t val;
    for (n = 0, tok = strsep(&cstr, ","); tok; tok = strsep(&cstr, ","), n++) {
      endptr = (char *)0;
      val = strtoul(tok, &endptr, 0);
      if (endptr) {
        switch (toupper(*endptr)) {
        case 'K':
          val *= 1024;
          break;
        case 'M':
          val *= 1024*1024;
          break;
        default:
          break;
        }
      }
      switch (n) {
      case 0:
        __default_conf.cmem = val;
        break;
      case 1:
        __default_conf.l1mem = val;
        break;
      }
    }
    //printf(".. cmem=%ld, l1mem=%ld\n", __default_conf.cmem, __default_conf.l1mem);
  }

#if defined(ENABLE_THREADS)
  if (__default_conf.maxproc > 1) {
    // read scheduling config
    cstr = getenv(ENV_ARMAS_SCHED);
    armas_parse_scheduling(cstr, &__default_sched.cpus, &__default_conf);

    // init scheduler
    armas_sched_conf(&__default_sched, &__default_conf, 64);
  }
#endif
  armas_x_init_flag = 1;
}



// Local Variables:
// indent-tabs-mode: nil
// End:
