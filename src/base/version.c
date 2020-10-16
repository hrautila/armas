// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "armas.h"

static const char *config_options[] = {
#ifdef CONFIG_OPT_FLOAT32
    CONFIG_OPT_FLOAT32,
#endif
#ifdef CONFIG_OPT_FLOAT64
    CONFIG_OPT_FLOAT64,
#endif
#ifdef CONFIG_OPT_SPARSE
    CONFIG_OPT_SPARSE,
#endif
#ifdef CONFIG_OPT_COMPAT
    CONFIG_OPT_COMPAT,
#endif
#ifdef CONFIG_OPT_ACCELERATOR
    CONFIG_OPT_ACCELERATOR,
#endif
#ifdef CONFIG_OPT_PRECISION
    CONFIG_OPT_PRECISION,
#endif
    ""
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
 * @brief Return library build configuration options. Last entry is always zero length string.
 */
const char **armas_config_options()
{
    return config_options;
}
