
// Copyright (c) Harri Rautila, 2018

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.


#include <stdio.h>
#include <ctype.h>
#include <math.h>
#include <armas/armas.h>

#define JSON_ONERROR(func)                        \
    do { if ((func) < 0) { return -1; } } while (0);

/**
 * @brief Write token to JSON stream.
 */
int armas_json_write_token(int token, const void *ptr, size_t len, armas_iostream_t *writer)
{
    char buf[64], *cp, *numformat = "%.14g";
    int n;
    double v;
    
    switch (token) {
    case '{':
    case '}':
    case '[':
    case ']':
    case ':':
    case ',':
        JSON_ONERROR(armas_putchar(writer, token));
        break;
    case ARMAS_JSON_STRING:
        cp = (char *)ptr;
        armas_putchar(writer, '"');
        for (int k = 0; *cp && k < len; k++, cp++) {
            switch (*cp) {
            case '\t':
                JSON_ONERROR(armas_putchar(writer, '\\'));
                JSON_ONERROR(armas_putchar(writer, 't'));
                break;
            case '\r':
                JSON_ONERROR(armas_putchar(writer, '\\'));
                JSON_ONERROR(armas_putchar(writer, 'r'));
                break;
            case '\n':
                JSON_ONERROR(armas_putchar(writer, '\\'));
                JSON_ONERROR(armas_putchar(writer, 'n'));
                break;
            case '\b':
                JSON_ONERROR(armas_putchar(writer, '\\'));
                JSON_ONERROR(armas_putchar(writer, 'b'));
                break;
            case '\\':
                JSON_ONERROR(armas_putchar(writer, '\\'));
                JSON_ONERROR(armas_putchar(writer, '\\'));
                break;
            case '/':
                JSON_ONERROR(armas_putchar(writer, '\\'));
                JSON_ONERROR(armas_putchar(writer, '/'));
                break;
            case '"':
                JSON_ONERROR(armas_putchar(writer, '\\'));
                JSON_ONERROR(armas_putchar(writer, '"'));
                break;
            default:
                if (!iscntrl(*cp))
                    JSON_ONERROR(armas_putchar(writer, *cp));
                break;
            }
        }
        JSON_ONERROR(armas_putchar(writer, '"'));
        break;
    case ARMAS_JSON_INT:
        if (len == sizeof(int)) {
            n = snprintf(buf, sizeof(buf), "%d", *((int *)ptr));
        } else {
            n = snprintf(buf, sizeof(buf), "%ld", *((long *)ptr));
        } 
        for (int k = 0; k < n; k++)
            JSON_ONERROR(armas_putchar(writer, buf[k]));
        break;
    case ARMAS_JSON_NUMBER:
        if (len == sizeof(double)) {
            v = *((double *)ptr);
        } else if (len == sizeof(float)) {
            v = (double)(*((float *)ptr));
            numformat = "%.8g";
        }
        if (isfinite(v)) {
            n = snprintf(buf, sizeof(buf), numformat, v);
            for (int k = 0; k < n; k++)
                JSON_ONERROR(armas_putchar(writer, buf[k]));
        }
        break;
    case ARMAS_JSON_TRUE:
        JSON_ONERROR(armas_putchar(writer, 't'));
        JSON_ONERROR(armas_putchar(writer, 'r'));
        JSON_ONERROR(armas_putchar(writer, 'u'));
        JSON_ONERROR(armas_putchar(writer, 'e'));
        break;
    case ARMAS_JSON_FALSE:
        JSON_ONERROR(armas_putchar(writer, 'f'));
        JSON_ONERROR(armas_putchar(writer, 'a'));
        JSON_ONERROR(armas_putchar(writer, 'l'));
        JSON_ONERROR(armas_putchar(writer, 's'));
        JSON_ONERROR(armas_putchar(writer, 'e'));
        break;
    case ARMAS_JSON_NULL:
        JSON_ONERROR(armas_putchar(writer, 'n'));
        JSON_ONERROR(armas_putchar(writer, 'u'));
        JSON_ONERROR(armas_putchar(writer, 'l'));
        JSON_ONERROR(armas_putchar(writer, 'l'));
        break;
    default:
        break;
    }
    return 0;
}

/**
 * @brief Write a simple token to JSON stream.
 */
int armas_json_write_simple_token(int tok, armas_iostream_t *ios)
{
    return armas_json_write_token(tok, (const void *)0, 0, ios);
}

// Local Variables:
// indent-tabs-mode: nil
// c-basic-offset: 4
// End:
