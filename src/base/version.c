// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "armas.h"

#ifndef CONFIG_SETTINGS
#define CONFIG_SETTINGS "unknown"
#endif
#ifndef COMPILER
#define "cc-unknown"
#endif
#ifndef COMPILE_TIME
#define "compile time unknown"
#endif

static const char *info[] = {
    PACKAGE_NAME,
    PACKAGE_VERSION,
    "configure " CONFIG_SETTINGS,
    COMPILER,
    COMPILE_TIME,
    "This is free software, distributed under the terms of GNU Lesser General Public License\n"
    "Version 3, or any later version. See the source for copying conditions. There is NO\n"
    "warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.",
    (char *)0
};

/**
 * @brief Return library version string.
 */
const char *armas_version()
{
    return PACKAGE_VERSION;
}

/**
 * @brief Return library name.
 */
const char *armas_name()
{
    return PACKAGE_NAME;
}

/**
 * @brief Return library build settings. Last entry is always null pointer.
 */
const char **armas_info()
{
    return info;
}
