
// Copyright (c) Harri Rautila, 2018-2020

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#define _GNU_SOURCE
#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include <ctype.h>
#include <sched.h>

#include "armas.h"
#include "accel.h"

#ifndef ENV_ARMAS_AC_CONFIG
#define ENV_ARMAS_AC_CONFIG "ARMAS_AC_CONFIG"
#endif

const char *ARMAS_AC_SIMPLE = "SIMPLE";
const char *ARMAS_AC_WORKERS = "WORKERS";
const char *ARMAS_AC_TRANSIENT = "TRANSIENT";

extern int armas_ac_threaded_init(struct armas_ac_vtable **vptr, void **private);
extern int armas_ac_workers_init(struct armas_ac_vtable **vptr, void **private);
extern int armas_ac_transient_init(struct armas_ac_vtable **vptr, void **private);

/**
 * @brief Initialize defined accelerator.
 *
 * @param handle
 *     On exit handle to initilized accelerator object.
 * @param name
 *     Accelerator name. Prefined name or path to dynamic library.
 *
 */
int armas_ac_init(armas_ac_handle_t *handle, const char *name)
{
    if (!name)
        return -1;

    struct armas_accel *ac = calloc(1, sizeof(struct armas_accel));
    if (!ac)
        return -ARMAS_EMEMORY;

    ac->handle = (void *)0;
    if (name == ARMAS_AC_SIMPLE || toupper(*name) == 'S') {
        if (armas_ac_threaded_init(&ac->vptr, &ac->private) < 0)
            goto error_out;
    } else  if (name == ARMAS_AC_WORKERS || toupper(*name) == 'W') {
        if (armas_ac_workers_init(&ac->vptr, &ac->private) < 0)
            goto error_out;
    } else  if (name == ARMAS_AC_TRANSIENT || toupper(*name) == 'T') {
        if (armas_ac_transient_init(&ac->vptr, &ac->private) < 0)
            goto error_out;
    }
    *handle = ac;
    return 0;
    // open dynamic library and call init
error_out:
    free(ac);
    *handle = (struct armas_accel *)0;
    return -1;
}

/**
 * @brief Release all accelerator resource.
 *
 * On exit handle is invalid.
 */
void armas_ac_release(armas_ac_handle_t handle)
{
    struct armas_accel *ac = (struct armas_accel *)handle;
    if (!ac)
        return ;

    if (ac->vptr)
        (*ac->vptr->release)(ac->private);
    if (ac->handle) {
        // release dlopened resources
    }
    ac->handle = (void *)0;
    ac->private = (void *)0;
    ac->vptr = (struct armas_ac_vtable *)0;
    free(ac);
}

/**
 * @brief Dispatch operation to accelerator.
 */
int armas_ac_dispatch(
    armas_ac_handle_t handle, int opcode, void *args, struct armas_conf *cf)
{
    struct armas_accel *ac = (struct armas_accel *)handle;
    if (!ac)
        return -ARMAS_EIMP;

    int rc = (*ac->vptr->dispatch)(opcode, args, cf, ac->private);
    return rc;
}

static
void parse_policy(struct armas_ac_env *env, char *str)
{
    if (!str)
        return;

    // starts with alp
    if (isalpha(*str)) {
        switch (toupper(*str)) {
        case 'B':
            env->options |= ARMAS_OBLAS_BLOCKED;
            break;
        case 'T':
            env->options |= ARMAS_OBLAS_TILED;
            break;
        }
        str++;
    }

    // second policy character
    if (isalpha(*str)) {
        switch (toupper(*str)) {
        case 'R':
            env->options |= ARMAS_OSCHED_ROUNDROBIN;
            break;
        case 'Z':
            env->options |= ARMAS_OSCHED_RANDOM;
            break;
        }
        str++;
    }

    if (isdigit(*str)) {
        if (*str == '2')
            env->options |= ARMAS_OSCHED_TWO;
        ++str;
    }
}

static
void parse_cpu_spec(struct armas_ac_env *env, char *str)
{
    char *tok, *s0, *s1;
    char *cstr = str;
    int n, i, val, start, end, step;
    unsigned long bits;
    long nproc = sysconf(_SC_NPROCESSORS_ONLN);;

    // parse cpu reservations
    for (n = 0, tok = strsep(&cstr, ","); tok; tok = strsep(&cstr, ","), n++) {
        // empty token
        if (!*tok)
            continue;

        // KK:0xNNN = bitmask, N = number, K-L/N = from K to L with steps of N
        if ((s0 = strchr(tok, ':')) || tolower(tok[1]) == 'x') {
            // hex mask (unsigned long) with optional starting offset
            start = 0;
            if (s0) {
                // here is: kk:mask
                *s0 = '\0';
                start = strtol(tok, NULL, 10);
                bits = strtoul(s0 + 1, NULL, 0);
            } else {
                // here 0xmask
                bits = strtoul(tok, NULL, 16);
            }

            // only processer number less than nproc
            for (i = 0; i < 8 * sizeof(val) && start + i < nproc; i++) {
                if ((bits & (1 << i)) != 0) {
                    CPU_SET(start + i, &env->cpus);
                }
            }
        } else if ((s0 = strchr(tok, '-')) != 0) {
            // parse range with optional step
            step = 1;
            if ((s1 = strchr(tok, '/')) != 0) {
                ;
                *s1 = '\0';
                step = strtol(s1 + 1, NULL, 10);
            }
            *s0 = '\0';
            start = strtol(tok, NULL, 10);
            end = strtol(s0 + 1, NULL, 10);
            for (i = start; i <= end && i < nproc; i += step) {
                CPU_SET(i, &env->cpus);
            }
        } else {
            val = strtol(tok, NULL, 10);
            if (val < nproc)
                CPU_SET(val, &env->cpus);
        }
    }
}

/**
 * @brief Read accelerator configuration from ARMAS_AC_CONFIG.
 */
void armas_ac_read_env(struct armas_ac_env *env)
{
    int n, val;
    char *tok, *cstr;

    env->max_cores = sysconf(_SC_NPROCESSORS_ONLN);
    cstr = getenv(ENV_ARMAS_AC_CONFIG);
    CPU_ZERO(&env->cpus);

    /* parse string:
     *   ELEMS_PER_THREAD,MAXPROCESSORS,POLICY,CPUSPEC [,CPUSPEC]
     */
    for (n = 0, tok = strsep(&cstr, ","); tok; tok = strsep(&cstr, ","), n++) {
        switch (n) {
        case 0:
            val = atoi(tok);
            if (val > 0)
                env->num_items = val;
            break;
        case 1:
            val = atoi(tok);
            if (val > 0 && val < env->max_cores)
                env->max_cores = val;
            break;
        case 2:
            parse_policy(env, tok);
            break;
        default:
            parse_cpu_spec(env, tok);
            break;
        }
    }
}

static struct armas_ac_env env = {
    .max_cores = 1,
    .num_items = 400 * 400,
    .weight = 1.0,
    .options = 0,
};

/**
 * @brief Get accelerator environment.
 */
struct armas_ac_env *armas_ac_getenv()
{
    static int initialized = 0;
    if (!initialized) {
        armas_ac_read_env(&env);
        initialized = 1;
    }
    return &env;
}
