#pragma once
#ifndef SFMT_DIST_H
#define SFMT_DIST_H

#include "config.h"
#include "debug.h"

#if HAVE_STRING_H
#include <string.h>
#endif
#if HAVE_MEMORY_H
#include <memory.h>
#endif

#if HAVE_ZMMINTRIN_H
#include <zmmintrin.h>
#endif

#if HAVE_IMMINTRIN_H
#include <immintrin.h>
#endif

#endif // SFMT_DIST_H
