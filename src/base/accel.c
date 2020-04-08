
// Copyright (c) Harri Rautila, 2018-2020

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include <string.h>
#include <stdio.h>
#include "armas.h"
#include "accel.h"

const char *ARMAS_AC_THREADED = "THREADED";
const char *ARMAS_AC_WORKERS = "WORKERS";

extern int armas_ac_threaded_init(struct armas_ac_vtable **vptr, void **private);
extern int armas_ac_workers_init(struct armas_ac_vtable **vptr, void **private);

int armas_ac_init(struct armas_accel *ac, const char *name)
{
    if (!ac || !name)
        return -1;

    ac->handle = (void *)0;
    if (name == ARMAS_AC_THREADED || *name == 'T') {
        return armas_ac_threaded_init(&ac->vptr, &ac->private);
    }
    if (name == ARMAS_AC_WORKERS || *name == 'W') {
        return -1;
    }
    // open dynamic library add call init
    return -1;
}

void armas_ac_release(struct armas_accel *ac)
{
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
}
