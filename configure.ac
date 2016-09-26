#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([sfmt-dist], [0.2], [saito@manieth.com])
LT_INIT
AC_CONFIG_SRCDIR([src/Makefile.am])
AC_CONFIG_HEADERS([src/config.h])
AC_CONFIG_MACRO_DIRS([m4])
AM_INIT_AUTOMAKE

AC_CANONICAL_HOST

# Checks for programs.
: ${CFLAGS=""}
: ${CXXFLAGS=""}
AC_PROG_CXX
AC_PROG_CC
AC_PROG_INSTALL
AC_PROG_LIBTOOL

# Checks for header files.
AC_HEADER_STDC
AC_CHECK_HEADERS([inttypes.h stdint.h stdlib.h x86intrin.h immintrin.h \
emmintrin.h smmintrin.h tmmintrin.h nmmintrin.h immintrin.h zmmintrin.h \
cpuid.h])
#AX_EXT

# Checks for typedefs, structures, and compiler characteristics.
#AC_TYPE_INT64_T
#AC_TYPE_UINT64_T

AC_LANG_SAVE
AC_LANG_CPLUSPLUS
AX_CHECK_COMPILE_FLAG([-std=c++11], [ac_cv_cpp11=yes], [ac_cv_cpp11=no])
if test $ac_cv_cpp11 = yes; then
   AX_APPEND_FLAG([-std=c++11], [CXXFLAGS])
else
   AC_MSG_ERROR(Need c++11)
fi
AC_LANG_RESTORE

# Checks for library functions.
#AC_CHECK_FUNCS([strtoull _may_i_use_cpu_feature __builtin_cpu_supports])
AC_MSG_CHECKING(for ac_may_i_use_cpu_feature)
AC_CACHE_VAL(ac_cv_may_i_use_cpu_feature,
AC_TRY_COMPILE([],
 [_may_i_use_cpu_feature(0)],
 ac_cv_may_i_use_cpu_feature=yes,
 ac_cv_may_i_use_cpu_feature=no))
AC_MSG_RESULT($ac_cv_may_i_use_cpu_feature)
if test $ac_cv_may_i_use_cpu_feature = yes; then
  AC_DEFINE(HAVE_MAY_I_USE_CPU_FEATURE, [1], [Define to 1 if you have _may_i_use_cpu_feature function])
fi

AC_MSG_CHECKING(for ac_builtin_cpu_supports)
AC_CACHE_VAL(ac_cv_builtin_cpu_supports,
AC_TRY_COMPILE([],
 [__builtin_cpu_supports("mmx")],
 ac_cv_builtin_cpu_supports=yes,
 ac_cv_builtin_cpu_supports=no))
AC_MSG_RESULT($ac_cv_builtin_cpu_supports)
if test $ac_cv_builtin_cpu_supports = yes; then
  AC_DEFINE(HAVE_BUILTIN_CPU_SUPPORTS, [1], [Define to 1 if you have __builtin_cpu_supports function])
fi

# SIMD compile flag check
AX_CHECK_COMPILE_FLAG([-mavx512f], [AX_APPEND_FLAG([-mavx512f], [CXXFLAGS])],
 [], [],
 [AC_LANG_PROGRAM([#include "zmmintrin.h"], [__m512i a, b;_mm512_xor_si512(a,b)])])
AX_CHECK_COMPILE_FLAG([-mavx2], [AX_APPEND_FLAG([-mavx2], [CXXFLAGS])],
 [], [],
 [AC_LANG_PROGRAM([#include "immintrin.h"], [__m256i a,b;_mm256_zeroupper();_mm256_xor_si256(a,b)])])
AX_CHECK_COMPILE_FLAG([-mavx], [AX_APPEND_FLAG([-mavx], [CXXFLAGS])],
 [], [],
 [AC_LANG_PROGRAM([#include "immintrin.h"], [__m256d a,b;_mm256_zeroupper();_mm256_add_pd(a,b)])])
AX_CHECK_COMPILE_FLAG([-msse4.2], [AX_APPEND_FLAG([-msse4.2], [CXXFLAGS])],
 [], [],
 [AC_LANG_PROGRAM([#include "nmmintrin.h"], [_mm_crc32_u16(0,0)])])
AX_CHECK_COMPILE_FLAG([-msse4.1], [AX_APPEND_FLAG([-msse4.1], [CXXFLAGS])],
 [], [],
 [AC_LANG_PROGRAM([#include "smmintrin.h"], [__m128i a,b;_mm_mul_epi32(a,b)])])
AX_CHECK_COMPILE_FLAG([-mssse3], [AX_APPEND_FLAG([-mssse3], [CXXFLAGS])],
 [], [],
 [AC_LANG_PROGRAM([#include "tmmintrin.h"], [__m128i a;_mm_abs_epi16(a)])])
AX_CHECK_COMPILE_FLAG([-msse2], [AX_APPEND_FLAG([-msse2], [CXXFLAGS])],
 [], [],
 [AC_LANG_PROGRAM([#include "emmintrin.h"], [__m128i a,b;_mm_and_si128(a,b)])])

AC_DEFINE(HAVE_AVX512F, [0], [Define to 1 if compiler can use -mavx512f])
AC_DEFINE(HAVE_AVX2, [0], [Define to 1 if compiler can use -mavx2])
AC_DEFINE(HAVE_AVX, [0], [Define to 1 if compiler can use -mavx])
AC_DEFINE(HAVE_SSE4_2, [0], [Define to 1 if compiler can use -msse4.2])
AC_DEFINE(HAVE_SSE4_1, [0], [Define to 1 if compiler can use -msse4.1])
AC_DEFINE(HAVE_SSSE3, [0], [Define to 1 if compiler can use -mssse3])
AC_DEFINE(HAVE_SSE2, [0], [Define to 1 if compiler can use -msse2])

for flag in $CXXFLAGS; do
  if test $flag = "-mavx512f"; then
    AC_DEFINE(HAVE_AVX512F, [1], [Define to 1 if compiler can use -mavx512f])
  fi
  if test $flag = "-mavx"; then
    AC_DEFINE(HAVE_AVX, [1], [Define to 1 if compiler can use -mavx])
  fi
  if test $flag = "-mavx2"; then
    AC_DEFINE(HAVE_AVX2, [1], [Define to 1 if compiler can use -mavx2])
  fi
  if test $flag = "-msse4.2"; then
    AC_DEFINE(HAVE_SSE4_2, [1], [Define to 1 if compiler can use -msse4.2])
  fi
  if test $flag = "-msse4.2"; then
    AC_DEFINE(HAVE_SSE4_1, [1], [Define to 1 if compiler can use -msse4.1])
  fi
  if test $flag = "-mssse3"; then
    AC_DEFINE(HAVE_SSSE3, [1], [Define to 1 if compiler can use -mssse3])
  fi
  if test $flag = "-msse2"; then
    AC_DEFINE(HAVE_SSE2, [1], [Define to 1 if compiler can use -msse2])
  fi
done

AC_CONFIG_FILES([Makefile src/Makefile include/Makefile])
AC_OUTPUT
