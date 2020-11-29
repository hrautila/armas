
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "armas.h"

static
int file_getc(void *ptr)
{
    FILE *fp = (FILE *)ptr;
    return fgetc(fp);
}

static
void file_ungetc(void *ptr, int c)
{
    FILE *fp = (FILE *)ptr;
    ungetc(c, fp);
}

static
int file_putc(void *ptr, int c)
{
    FILE *fp = (FILE *)ptr;
    return fputc(c, fp);
}

static
armas_iostream_vtable_t file_vtable = (armas_iostream_vtable_t){
    .get_char = file_getc,
    .unget_char = file_ungetc,
    .put_char = file_putc
};

void armas_ios_filestream(armas_iostream_t *ios, FILE *fp)
{
    armas_ios_init(ios, &file_vtable, fp);
}
