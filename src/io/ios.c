
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
    .put_char = file_putc,
    .close = 0
};

void armas_ios_file(armas_iostream_t *ios, FILE *fp)
{
    armas_ios_init(ios, &file_vtable, fp);
}

static
int wb_getc(void *ptr)
{
    struct armas_wbuf *wb = (struct armas_wbuf *)ptr;
    if (wb->offset < wb->bytes)
        return wb->buf[wb->offset++];
    return -1;
}

static
void wb_ungetc(void *ptr, int c)
{
    struct armas_wbuf *wb = (struct armas_wbuf *)ptr;
    if (wb->offset > 0)
        wb->buf[--wb->offset] = c;
}

static
int wb_putc(void *ptr, int c)
{
    struct armas_wbuf *wb = (struct armas_wbuf *)ptr;
    if (wb->offset < wb->bytes) {
        wb->buf[wb->offset++] = c;
        return c;
    }
    return -1;
}

static
void str_close(void *ptr)
{
    if (ptr)
        free(ptr);
}

static
armas_iostream_vtable_t str_vtable = (armas_iostream_vtable_t){
    .get_char = wb_getc,
    .unget_char = wb_ungetc,
    .put_char = 0,
    .close = str_close
};

void armas_ios_string(armas_iostream_t *ios, const char *s, int len)
{
    struct armas_wbuf *wb = malloc(sizeof(*wb));
    if (wb) {
        wb->buf = (char *)s;
        wb->bytes = len;
        wb->offset = 0;
    }
    armas_ios_init(ios, &str_vtable, wb);
}

static
armas_iostream_vtable_t buf_vtable = (armas_iostream_vtable_t){
    .get_char = wb_getc,
    .unget_char = wb_ungetc,
    .put_char = wb_putc,
    .close = 0
};

void armas_ios_wbuf(armas_iostream_t *ios, struct armas_wbuf *wb)
{
    armas_ios_init(ios, &buf_vtable, wb);
}
