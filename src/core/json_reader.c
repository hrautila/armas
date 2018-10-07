
// Copyright (c) Harri Rautila, 2018

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.


#include <stdio.h>
#include <ctype.h>
#include <armas/armas.h>

/*
 *  ECMA 404  (https://json.org)
 *
 *  json:
 *     element
 *  element:
 *     ws value ws
 *  value:
 *     object | array | string | number | TRUE | FALSE | NULL
 *  object:
 *     '{' ws '}' | '{' members '}'
 *  members:
 *     member | member ',' members
 *  member:
 *     ws string ws ':' element
 *  array:
 *     '[' ws ']' | '[' elements ']'
 *  elements:
 *     element | element ',' elements
 *  number:
 *     int frac exp
 *  int:
 *     digit | onenine digits | '-' digit | '-' onenine digits
 *  digits:
 *     digit | digit digits
 *  digit:
 *     '0' | onenine
 *  onenine:
 *     '1' .. '9'
 *  frac:
 *     "" | '.' digits
 *  exp:
 *    "" | 'E' sign digits | 'e' sign digits
 *  sign: 
 *    "" | '+' | '-'
 *  string:
 *    '"' characters '"'
 *  characters:
 *    "" | character characters
 *  character:
 *    '0020' . '10ffff' - '"' - '\' | '\' escape
 *  escape:
 *    '"' | '\' | '/' | 'b' | 'n' | 'r' | 't' | 'u' hex hex hex hex
 *  hex:
 *    digit | 'A' . 'F' | 'a' . 'f'
 *  ws:
 *    "" | ' ' | '\t' | '\r' | '\n'
 *  
 */


enum {
    JSON_STATE_VALUE = 0,
    JSON_STATE_STRING,
    JSON_STATE_NUMBER,
    JSON_STATE_EXP,
    JSON_STATE_EXP1,
    JSON_STATE_ESCAPE,
    JSON_STATE_HEX1,
    JSON_STATE_HEX2,
    JSON_STATE_HEX3,
    JSON_STATE_HEX4,
    JSON_STATE_TRUE,
    JSON_STATE_TRUE1,
    JSON_STATE_TRUE2,
    JSON_STATE_TRUE3,
    JSON_STATE_FALSE,
    JSON_STATE_FALSE1,
    JSON_STATE_FALSE2,
    JSON_STATE_FALSE3,
    JSON_STATE_FALSE4,
    JSON_STATE_NULL,
    JSON_STATE_NULL1,
    JSON_STATE_NULL2,
    JSON_STATE_NULL3
}; 


/**
 * @brief Read next token from JSON stream.
 */
int armas_json_read_token(char *buf, size_t len, armas_iostream_t *reader)
{
    int c;
    int state = JSON_STATE_VALUE;
    int is_frac = 0;
    char lit[8], *cp, *bp;
    
    bp = buf;
    for (;;) {
        /* */
        c = armas_getchar(reader);
        if (c < 0) {
            switch (state) {
            case JSON_STATE_NUMBER:
            case JSON_STATE_EXP1:
                // we had one element; a number
                *bp = '\0';
                return is_frac ? ARMAS_JSON_NUMBER : ARMAS_JSON_INT;
            default:
                return ARMAS_JSON_EOF;
            }
        }

        switch (state) {
        case JSON_STATE_STRING:
            if (c == '\"') {
                *bp = '\0';
                return ARMAS_JSON_STRING;
            }
            if (c == '\\') {
                state = JSON_STATE_ESCAPE;
            }
            else {
                *bp++ = c;
            }
            break;

        case JSON_STATE_ESCAPE:
            state = JSON_STATE_STRING;
            switch (c) {
            case '"':
            case '\\':
            case '/':
                *bp++ = c;
                break;
            case 'b':
                *bp++ = '\b';
                break;
            case 'n':
                *bp++ = '\n';
                break;              
            case 'r':
                *bp++ = '\r';
                break;
            case 't':
                *bp++ = '\t';
                break;
            case 'u':
                state = JSON_STATE_HEX1;
                break;
            default:
                return ARMAS_JSON_EINVAL;
            }
            break;

        case JSON_STATE_HEX1:
            cp = lit;
            if (!isxdigit(c))
                return ARMAS_JSON_EINVAL;
            *cp++ = c;
            state = JSON_STATE_HEX2;
            break;

        case JSON_STATE_HEX2:
            if (!isxdigit(c))
                return ARMAS_JSON_EINVAL;
            *cp++ = c;
            state = JSON_STATE_HEX3;
            break;
            
        case JSON_STATE_HEX3:
            if (!isxdigit(c))
                return ARMAS_JSON_EINVAL;
            *cp++ = c;
            state = JSON_STATE_HEX4;
            break;

        case JSON_STATE_HEX4:
            if (!isxdigit(c))
                return ARMAS_JSON_EINVAL;
            *cp++ = c;
            *cp = '\0';
            c = atoi(lit) & 0xff;
            *bp++ = c;
            state = JSON_STATE_STRING;
            break;

        case JSON_STATE_NUMBER:
            switch (c) {
            case ',':
            case ']':
            case '}':
            case ' ':
            case '\n':
            case '\r':
            case '\t':
                armas_ungetchar(reader, c);
                *bp = '\0';
                return is_frac ? ARMAS_JSON_NUMBER : ARMAS_JSON_INT;
            case 'e':
            case 'E':
                *bp++ = c;
                state = JSON_STATE_EXP;
                break;
            case '.':
                if (is_frac)
                    return ARMAS_JSON_EINVAL;
                *bp++ = c;
                is_frac = 1;
                break;
            default:
                if (isdigit(c)) {
                    *bp++ = c;
                }
                else {
                    return ARMAS_JSON_EINVAL;
                }
                break;
            }
            break;

        case JSON_STATE_EXP:
            // must get at least one digit after E|e
            switch (c) {
            case '-':
            case '+':
                *bp++ = c;
                break;
            default:
                if (isdigit(c)) {
                    *bp++ = c;
                    state = JSON_STATE_EXP1;
                }
                else {
                    return ARMAS_JSON_EINVAL;
                }
                break;
            }
            break;
            
        case JSON_STATE_EXP1:
            // now we have at least one digit after e|E
            switch (c) {
            case ',':
            case ']':
            case '}':
            case ' ':
            case '\n':
            case '\r':
            case '\t':
                armas_ungetchar(reader, c);
                *bp = '\0';
                return ARMAS_JSON_NUMBER;
            default:
                if (isdigit(c)) {
                    *bp++ = c;
                }
                else {
                    return ARMAS_JSON_EINVAL;
                }
                break;
            }
            break;

        case JSON_STATE_TRUE:
            if (c != 'r')
                return ARMAS_JSON_EINVAL;
            state = JSON_STATE_TRUE1;
            break;
        case JSON_STATE_TRUE1:
            if (c != 'u')
                return ARMAS_JSON_EINVAL;
            state = JSON_STATE_TRUE2;
            break;
        case JSON_STATE_TRUE2:
            if (c != 'e')
                return ARMAS_JSON_EINVAL;
            return ARMAS_JSON_TRUE;
            
        case JSON_STATE_FALSE:
            if (c != 'a')
                return ARMAS_JSON_EINVAL;
            state = JSON_STATE_FALSE1;
            break;
        case JSON_STATE_FALSE1:
            if (c != 'l')
                return ARMAS_JSON_EINVAL;
            state = JSON_STATE_FALSE2;
            break;
        case JSON_STATE_FALSE2:
            if (c != 's')
                return ARMAS_JSON_EINVAL;
            state = JSON_STATE_FALSE3;
            break;
        case JSON_STATE_FALSE3:
            if (c != 'e')
                return ARMAS_JSON_EINVAL;
            return ARMAS_JSON_FALSE;

        case JSON_STATE_NULL:
            if (c != 'u')
                return ARMAS_JSON_EINVAL;
            state = JSON_STATE_NULL1;
            break;
        case JSON_STATE_NULL1:
            if (c != 'l')
                return ARMAS_JSON_EINVAL;
            state = JSON_STATE_NULL2;
            break;
        case JSON_STATE_NULL2:
            if (c != 'l')
                return ARMAS_JSON_EINVAL;
            return ARMAS_JSON_NULL;

        default:
            switch (c) {
            case '{':
            case '}':
            case '[':
            case ']':
            case ':':
            case ',':
                return c;

            case '\"':
                state = JSON_STATE_STRING;
                break;
            
            case ' ':
            case '\n':
            case '\t':
            case '\r':
                // white space; consume
                break;

            case 't':
                state = JSON_STATE_TRUE;
                break;

            case 'f':
                state = JSON_STATE_FALSE;
                break;

            case 'n':
                state = JSON_STATE_NULL;
                break;
                
            default:
                if (c == '\0') {
                    return ARMAS_JSON_EOF;
                }
                else if (iscntrl(c)) {
                    return ARMAS_JSON_EINVAL;
                }
                if (isdigit(c) || c == '-') {
                    *bp++ = c;
                    state = JSON_STATE_NUMBER;
                    break;
                }
                break;
            }
            
        }
        if ((size_t)(bp - buf) >= len-1) {
            // end buffer space
            return ARMAS_JSON_E2BIG;
        }
    }
    // reach here when hitting EOF
    return ARMAS_JSON_EOF;
}

// Local Variables:
// indent-tabs-mode: nil
// c-basic-offset: 4
// End:
