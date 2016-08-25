#include "sfmt-dist.h"
#include <stdint.h>
#include <cstdio>
#include <sfmt-dist/cpu_feature.h>

#if HAVE_IMMINTRIN_H
#include "immintrin.h"
#endif

namespace {
    using namespace MersenneTwister;
#define cpuid(feature, ax, bx, cx, dx)                                  \
    __asm__ __volatile__ ("xor %%ecx, %%ecx;"                           \
                          "cpuid"                                       \
                          : "=a"(ax), "=b"(bx), "=c"(cx), "=d"(dx)      \
                          : "a"(feature), "b"(bx), "c"(cx), "d"(dx))

    bool get_SIMD_feature(cpu_feature_t * simd)
    {
        const uint32_t SSE2_FLAG = 1 << 26;
        const uint32_t SSSE3_FLAG = 1 << 9;
        const uint32_t SSE41_FLAG = 1 << 19;
        const uint32_t SSE42_FLAG = 1 << 20;
        const uint32_t AVX_FLAG = 1 << 28;
        const uint32_t AVX2_FLAG = 1 << 5;
        const uint32_t AVX512F_FLAG = 1 << 16;
        uint32_t eax, ebx, ecx, edx;
        uint32_t max;
        bool any = false;
        ebx = 0;
        ecx = 0;
        edx = 0;
        cpuid(0, max, ebx, ecx, edx);
        if (max == 0) {
            return any;
        }
        if (max >= 1) {
            cpuid(1, eax, ebx, ecx, edx);
            //printf("ecx = %x SSSE3 = %x sse4 = %x\n", ecx, SSSE3, SSE42);
            if ((edx & SSE2_FLAG) != 0) {
                simd->sse2 = 1;
                any = true;
            }
            if ((ecx & SSSE3_FLAG) != 0) {
                simd->ssse3 = 1;
                any = true;
            }
            if ((ecx & SSE41_FLAG) != 0) {
                simd->sse4_1 = 1;
                any = true;
            }
            if ((ecx & SSE42_FLAG) != 0) {
                simd->sse4_2 = 1;
                any = true;
            }
            if ((ecx & AVX_FLAG) != 0) {
                simd->avx = 1;
                any = true;
            }
        }
        if (max >= 7) {
            cpuid(7, eax, ebx, ecx, edx);
            //printf("ebx = %x\n", ebx);
            if ((ebx & AVX2_FLAG) != 0) {
                simd->avx2 = 1;
                any = true;
            }
            if ((ebx & AVX512F_FLAG) != 0) {
                simd->avx512f = 1;
                any = true;
            }
        }
        return any;
    }
}

namespace MersenneTwister {
    cpu_feature_t cpu_feature()
    {
        cpu_feature_t simd = {0, 0, 0, 0, 0, 0, 0};
        cpu_feature_t result = {0, 0, 0, 0, 0, 0, 0};
        bool success = get_SIMD_feature(&simd);
        if (success) {
            if (simd.avx512f && HAVE_AVX512F) {
                result.avx512f = 1;
            }
            if (simd.avx2 && HAVE_AVX2) {
                result.avx2 = 1;
            }
            if (simd.avx && HAVE_AVX) {
                result.avx = 1;
            }
            if (simd.sse4_2 && HAVE_SSE4_2) {
                result.sse4_2 = 1;
            }
            if (simd.sse4_1 && HAVE_SSE4_1) {
                result.sse4_1 = 1;
            }
            if (simd.ssse3 && HAVE_SSSE3) {
                result.ssse3 = 1;
            }
            if (simd.sse2 && HAVE_SSE2) {
                result.sse2 = 1;
            }
            return result;
        }

#if HAVE_MAY_I_USE_CPU_FEATURE
        bool any = false;
#if HAVE_AVX512 && defined(_FEATURE_AVX512F)
        if (_may_i_use_cpu_feature(_FEATURE_AVX512F)) {
            result.avx512f = 1;
            any = true;
        }
#endif
#if HAVE_AVX2 && defined(_FEATURE_AVX2)
        if (_may_i_use_cpu_feature(_FEATURE_AVX2)) {
            result.avx2 = 1;
            any = true;
        }
#endif
#if HAVE_AVX && defined(_FEATURE_AVX)
        if (_may_i_use_cpu_feature(_FEATURE_AVX)) {
            result.avx = 1;
            any = true;
        }
#endif
#if HAVE_SSE4_2 && defined(_FEATURE_SSE4_2)
        if (_may_i_use_cpu_feature(_FEATURE_SSE4_2)) {
            result.sse4_2 = 1;
            any = true;
        }
#endif
#if HAVE_SSE4_1 && defined(_FEATURE_SSE4_1)
        if (_may_i_use_cpu_feature(_FEATURE_SSE4_1)) {
            result.sse4_1 = 1;
            any = true;
        }
#endif
#if HAVE_SSSE3 && defined(_FEATURE_SSSE3)
        if (_may_i_use_cpu_feature(_FEATURE_SSSE3)) {
            result.ssse3 = 1;
            any = true;
        }
#endif
#if HAVE_SSE2 && defined(_FEATURE_SSE2)
        if (_may_i_use_cpu_feature(_FEATURE_SSE2)) {
            result.sse2 = 1;
            any = true;
        }
#endif
        if (any) {
            return result;
        }
#endif // HAVE_MAY_I_USE_CPU_FEATURE


#if HAVE_BUILTIN_CPU_SUPPORTS && defined(_GNUC_)
        if (__builtin_cpu_supports("avx512f") && HAVE_AVX512F) {
            result.avx512f = 1;
        }
        if (__builtin_cpu_supports("avx2") && HAVE_AVX2) {
            result.avx2 = 1;
        }
        if (__builtin_cpu_supports("avx") && HAVE_AVX) {
            result.avx = 1;
        }
        if (__builtin_cpu_supports("sse4.2") && HAVE_SSE4_2) {
            result.sse4_2 = 1;
        }
        if (__builtin_cpu_supports("sse4.1") && HAVE_SSE4_1) {
            result.sse4_1 = 1;
        }
        if (__builtin_cpu_supports("ssse3") && HAVE_SSSE3) {
            result.ssse3 = 1;
        }
        if (__builtin_cpu_supports("sse2") && HAVE_SSE2) {
            result.sse2 = 1;
        }
#endif // HAVE_BUILTIN_CPU_SUPPORTS
        return result;
    }

    void print_cpu_feature(cpu_feature_t cf)
    {
        printf("AVX512F:%d\n", cf.avx512f);
        printf("AVX2   :%d\n", cf.avx2);
        printf("AVX    :%d\n", cf.avx);
        printf("SSE4.2 :%d\n", cf.sse4_2);
        printf("SSE4.1 :%d\n", cf.sse4_1);
        printf("SSSE3  :%d\n", cf.ssse3);
        printf("SSE2   :%d\n", cf.sse2);
    }
}
