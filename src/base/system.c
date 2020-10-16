
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.


#include <sys/types.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <stdlib.h>

static void __on_exit(int c, void *p)
{
    void **ptr = (void **)p;
    if (p && *ptr) {
        void *pp = *ptr;
        *ptr = (void *)0;
        free(pp);
    }
}

void armas_on_exit(void *p)
{
    pid_t tid;
    tid = syscall(SYS_gettid);
    if (tid != getpid()) {
        on_exit(__on_exit, p);
    }
}
