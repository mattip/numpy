#ifndef _RANDOMDGEN__ENTROPY_H_
#define _RANDOMDGEN__ENTROPY_H_

#include <stddef.h>
#ifdef _WIN32
#if _MSC_VER == 1500
#include "../common/stdint.h"
typedef int bool;
#define false 0
#define true 1
#else
#include <stdbool.h>
#include <stdint.h>
#endif
#else
#include <stdbool.h>
#include <stdint.h>
#endif

extern void entropy_fill(void *dest, size_t size);

extern bool entropy_getbytes(void *dest, size_t size);

extern bool entropy_fallback_getbytes(void *dest, size_t size);

#endif
