#define DEBUG 1

#include <cmath>
#include <cfloat>
#include <sfmt-dist/aligned_alloc.h>
#define DSFMT_MEXP 19937
#include "sfmt_common.h"


#include "config.h"
#include <stdexcept>
#include "dsfmt_avx.h"
#include <sfmt-dist/cpu_feature.h>
#include "w256.h"
#include "w128.h"
#include "debug.h"

#if HAVE_ZMMINTRIN_H
#include <zmmintrin.h>
#endif

#if HAVE_IMMINTRIN_H
#include <immintrin.h>
#endif

#define DSFMT_MEXP 19937

#include "dsfmt_avx_params.h"
#include "dsfmt_avx_common.h"

namespace MersenneTwister {
    namespace dSFMT_AVX {
#if HAVE_AVX2 && HAVE_IMMINTRIN_H
        /**
         * This function fills the internal state array with pseudorandom
         * integers.
         * @param p parameter
         * @param state SFMT internal state
         */
        void fillState256_19937(double * state) {
            w256_t * pstate = reinterpret_cast<w256_t *>(state);
            __m256i lung = pstate[DSFMTAVX_SIZE].si256;
            __m256i mask = mask1.si256;
            int i = 0;
            for (;i < DSFMTAVX_SIZE - DSFMTAVX_POS1; i++) {
                pstate[i].si256 = recursion256(mask,
                                               pstate[i].si256,
                                               pstate[i + DSFMTAVX_POS1].si256,
                                               &lung);
            }
            for (;i < DSFMTAVX_SIZE; i++) {
                pstate[i].si256
                    = recursion256(mask,
                                   pstate[i].si256,
                                   pstate[i+DSFMTAVX_POS1-DSFMTAVX_SIZE].si256,
                                   &lung);
            }
            pstate[DSFMTAVX_SIZE].si256 = lung;
            _mm256_zeroall();
        }

        /**
         * This function fills the user-specified array with pseudorandom
         * integers.
         *
         * @param p parameter
         * @param state SFMT internal state.
         * @param array64 an 256-bit array to be filled by pseudorandom numbers.
         * @param length number of 256-bit pseudorandom numbers to be generated.
         */
        void fillArray256_c1o2_19937(double * state,
                                     double * array64, int length)
        {
            w256_t * pstate = reinterpret_cast<w256_t *>(state);
            w256_t * array = reinterpret_cast<w256_t *>(array64);
            const __m256i mask = mask1.si256;
            __m256i lung = pstate[2].si256;
//            __m256i a = pstate[0].si256;
//            __m256i b = pstate[1].si256;
            int i = 0;
            for (; i < DSFMTAVX_SIZE - DSFMTAVX_POS1; i++) {
                array[i].si256 = recursion256(mask,
                                              pstate[i].si256,
                                              pstate[i + DSFMTAVX_POS1].si256,
                                              &lung);
            }
            for (; i < DSFMTAVX_SIZE - DSFMTAVX_POS1; i++) {
                array[i].si256
                    = recursion256(mask,
                                   pstate[i].si256,
                                   array[i + DSFMTAVX_POS1-DSFMTAVX_SIZE].si256,
                                   &lung);
            }
            for (; i < length; i++) {
                array[i].si256
                    = recursion256(mask,
                                   array[i - DSFMTAVX_SIZE].si256,
                                   array[i + DSFMTAVX_POS1-DSFMTAVX_SIZE].si256,
                                   &lung);
            }
            int j = 0;
            for (int i = length - DSFMTAVX_SIZE; i < length ;i++) {
                pstate[j++].si256 = array[i].si256;
            }
            pstate[DSFMTAVX_SIZE].si256 = lung;
            _mm256_zeroall();
        }

        /**
         * This function fills the user-specified array with pseudorandom
         * integers.
         * @param p parameter
         * @param state SFMT internal state.
         * @param array64 an 256-bit array to be filled by pseudorandom numbers.
         * @param length number of 256-bit pseudorandom numbers to be generated.
         */
        void fillArray256_c0o1_19937(double * state, double * array64,
                                     int length)
        {
            w256_t * pstate = reinterpret_cast<w256_t *>(state);
            w256_t * array = reinterpret_cast<w256_t *>(array64);
            const __m256i mask = mask1.si256;
            __m256i lung = pstate[2].si256;
//            __m256i a = pstate[0].si256;
//            __m256i b = pstate[1].si256;
            int i = 0;
            for (; i < DSFMTAVX_SIZE - DSFMTAVX_POS1; i++) {
                array[i].si256 = recursion256(mask,
                                              pstate[i].si256,
                                              pstate[i + DSFMTAVX_POS1].si256,
                                              &lung);
            }
            for (; i < DSFMTAVX_SIZE - DSFMTAVX_POS1; i++) {
                array[i].si256
                    = recursion256(mask,
                                   pstate[i].si256,
                                   array[i+DSFMTAVX_POS1-DSFMTAVX_SIZE].si256,
                                   &lung);
            }
            for (; i < length; i++) {
                array[i].si256
                    = recursion256(mask,
                                   array[i - DSFMTAVX_SIZE].si256,
                                   array[i+DSFMTAVX_POS1-DSFMTAVX_SIZE].si256,
                                   &lung);
                convert256_c0o1(array[i - DSFMTAVX_SIZE].sd256);
            }
            int j = 0;
            for (int i = length - DSFMTAVX_SIZE; i < length ;i++) {
                pstate[j++].si256 = array[i].si256;
                convert256_c0o1(array[i].sd256);
            }
            pstate[DSFMTAVX_SIZE].si256 = lung;
            _mm256_zeroall();
        }

#else // don't HAVE_AVX
        void fillState256_19937(double *)
        {
            throw new std::logic_error("should not be called");
        }

        void fillArray256_c1o2_19937(double *, double *, int)
        {
            throw new std::logic_error("should not be called");
        }
        void fillArray256_c0o1_19937(double *, double *, int)
        {
            throw new std::logic_error("should not be called");
        }
        void fillArray256_o0c1_19937(double *, double *, int)
        {
            throw new std::logic_error("should not be called");
        }
        void fillArray256_o0o1_19937(double *, double *, int)
        {
            throw new std::logic_error("should not be called");
        }
#endif // HAVE_AVX2

        /**
         * This function fills the internal state array with pseudorandom
         * integers.
         * @param p parameter
         * @param sfmt SFMT internal state
         */
        void fillState64_19937(double * state64)
        {
            w256_t * state = reinterpret_cast<w256_t *>(state64);
            w256_t lung = state[DSFMTAVX_SIZE];
            int i = 0;
            for (; i < DSFMTAVX_SIZE - DSFMTAVX_POS1; i++) {
                recursion64(&state[i], &state[i],
                            &state[i + DSFMTAVX_POS1], &lung);
            }
            for (; i < DSFMTAVX_SIZE; i++) {
                recursion64(&state[i], &state[i],
                            &state[i + DSFMTAVX_POS1 - DSFMTAVX_SIZE], &lung);
            }
            state[DSFMTAVX_SIZE] = lung;
        }

        void fillArray64_c1o2_19937(double * state64,
                                    double * array64, int length)
        {
            DMSG("start fillArray64_c1o2");
            w256_t * state = reinterpret_cast<w256_t *>(state64);
            w256_t * array = reinterpret_cast<w256_t *>(array64);
            w256_t *lung = &state[2];
            int i = 0;
            for (; i < DSFMTAVX_SIZE - DSFMTAVX_POS1; i++) {
                recursion64(&array[i], &state[i],
                            &state[i + DSFMTAVX_POS1], lung);
            }
            for (; i < DSFMTAVX_SIZE; i++) {
                recursion64(&array[i], &state[i],
                            &array[i + DSFMTAVX_POS1 - DSFMTAVX_SIZE], lung);
            }
            for (; i < length; i++) {
                recursion64(&array[i], &array[i - DSFMTAVX_SIZE],
                            &array[i + DSFMTAVX_POS1 - DSFMTAVX_SIZE], lung);
            }
            int j = 0;
            for (i = length - DSFMTAVX_SIZE; i < length; i++) {
                state[j++] = array[i];
            }
            state[DSFMTAVX_SIZE] = *lung;
        }

        void fillArray64_c0o1_19937(double * state64,
                                    double * array64, int length)
        {
            w256_t * state = reinterpret_cast<w256_t *>(state64);
            w256_t * array = reinterpret_cast<w256_t *>(array64);
            w256_t *lung = &state[2];
            int i = 0;
            for (; i < DSFMTAVX_SIZE - DSFMTAVX_POS1; i++) {
                recursion64(&array[i], &state[i],
                            &state[i + DSFMTAVX_POS1], lung);
            }
            for (; i < DSFMTAVX_SIZE; i++) {
                recursion64(&array[i], &state[i],
                            &array[i+DSFMTAVX_POS1-DSFMTAVX_SIZE], lung);
            }
            for (; i < length; i++) {
                recursion64(&array[i], &array[i - DSFMTAVX_SIZE],
                            &array[i+DSFMTAVX_POS1-DSFMTAVX_SIZE], lung);
                convert64_c0o1(&array[i - DSFMTAVX_SIZE]);
            }
            int j = 0;
            for (i = length - DSFMTAVX_SIZE; i < length; i++) {
                state[j++] = array[i];
                convert64_c0o1(&array[i]);
            }
            state[DSFMTAVX_SIZE] = *lung;
        }

#if HAVE_AVX2
        inline __m128i do_uniform256(__m256d a, __m256d max256, __m128i min128)
        {
            a = _mm256_add_pd(a, m_one256.sd256);
            a = _mm256_mul_pd(a, max256);
            //__m128i y = _mm256_cvtpd_epi32(a);
            __m128i y = _mm256_cvttpd_epi32(a); // 常に切り捨て
            return _mm_add_epi32(y, min128);
        }

        void fillarray256_maxint_19937(double * state, int32_t *array32,
                                       uint64_t rmax, int32_t min)
        {
            DMSG("fillArray256_maxint step 1");
            //uint32_t mxcsr = _mm_getcsr();
            //_mm_setcsr((mxcsr & 0x9fff) | 0x6000);
            //_mm_setcsr((mxcsr & 0xdfff) | 0x2000);
            w256_t * pstate = reinterpret_cast<w256_t *>(state);
            w128_t * array = reinterpret_cast<w128_t *>(array32);
            __m128i min128 = _mm_set1_epi32(min);
            __m256d max256 = _mm256_set1_pd(static_cast<double>(rmax));
            __m256d a;
            fillState256_19937(state);
            DMSG("fillArray256_maxint step 2");
            int j = 0;
            for (int i = 0; i < DSFMTAVX_SIZE; i++) {
                a = pstate[i].sd256;
                array[j++].si128 = do_uniform256(a, max256, min128);
            }
            fillState256_19937(state);
            for (int i = 0; i < DSFMTAVX_SIZE; i++) {
                a = pstate[i].sd256;
                array[j++].si128 = do_uniform256(a, max256, min128);
            }
            //_mm_setcsr(mxcsr);
            _mm256_zeroall();
        }

        int fillarray256_boxmuller_19937(double * state, double * array,
                                         double mu, double sigma)
        {
            w256_t * state256 = reinterpret_cast<w256_t *>(state);
            w256_t * array256 = reinterpret_cast<w256_t *>(array);
            w256_t w;
            __m128d axy[DSFMTAVX_SIZE * 2];
            double ar[DSFMTAVX_SIZE * 2];
            fillState256_19937(state);
            int p = 0;
            for (int i = 0; i < DSFMTAVX_SIZE; i++) {
                w.sd256 = state256[i].sd256;
                w.sd256 = _mm256_mul_pd(two256.sd256, w.sd256);
                w.sd256 = _mm256_add_pd(w.sd256, m_three256.sd256);
                axy[i] = w.sd128[0];
                axy[p + 1] = w.sd128[1];
                w.sd256 = _mm256_mul_pd(w.sd256, w.sd256);
                ar[p] = w.d[0] + w.d[1];
                ar[p + 1] = w.d[2] + w.d[3];
                p += 2;
            }
            _mm256_zeroall();
            p = 0;
            for (int i = 0; i < DSFMTAVX_SIZE * 2; i++) {
                if (ar[i] > 1.0 || ar[i] == 0.0) {
                    continue;
                }
                axy[p] = axy[i];
                ar[p] = sqrt(-2.0 * log(ar[i]) / ar[i]);
                p++;
            }
            w256_t cmu;
            w256_t csigma;
            cmu.sd256 = _mm256_set1_pd(mu);
            csigma.sd256 = _mm256_set1_pd(sigma);
            int j = 0;
            __m256d * paxy = reinterpret_cast<__m256d *>(axy);
            for (int i = 0; i < p / 2; i++) {
                __m256d md = _mm256_setr_pd(ar[j], ar[j], ar[j+1], ar[j+1]);
                md = _mm256_mul_pd(md, paxy[i]);
                md = _mm256_mul_pd(md, csigma.sd256);
                w.sd256 = _mm256_add_pd(md, cmu.sd256);
                array256[i].sd256 = w.sd256;
                j += 2;
            }
            if (p % 2 == 1) {
                w128_t w2;
                __m128d md = _mm_set1_pd(ar[p - 1]);
                md = _mm_mul_pd(md, axy[p - 1]);
                md = _mm_mul_pd(md, csigma.sd128[0]);
                w2.sd128 = _mm_add_pd(md, cmu.sd128[0]);
                array[p * 2 - 2] = w2.d[0];
                array[p * 2 - 1] = w2.d[1];
            }
            _mm256_zeroall();
            return p * 2;
        }
#else
        void fillarray256_maxint_19937(double *, int32_t *, uint64_t, int32_t)
        {
            throw new std::logic_error("should not be called");
        }
        int fillarray256_boxmuller_19937(double *, double *, double, double)
        {
            throw new std::logic_error("should not be called");
        }
#endif

        void fillarray64_maxint_19937(double * state, int32_t * array,
                                      uint64_t rmax, int32_t min)
        {
            fillState64_19937(state);
            double dmax = static_cast<double>(rmax);
            int j = 0;
            for (int i = 0; i < DSFMTAVX_SIZE * 4; i++) {
                double x = (state[i] - 1.0) * dmax;
                int32_t y = static_cast<int32_t>(x);
                array[j++] = y + min;
            }
            fillState64_19937(state);
            for (int i = 0; i < DSFMTAVX_SIZE * 4; i++) {
                double x = (state[i] - 1.0) * dmax;
                int32_t y = static_cast<int32_t>(x);
                array[j++] = y + min;
            }
        }

        int fillarray64_boxmuller_19937(double * state, double * array,
                                        double mu, double sigma)
        {
            w128_t * state128 = reinterpret_cast<w128_t *>(state);
            //w128_t * array128 = reinterpret_cast<w128_t *>(array);
            w128_t w;
            double x;
            double y;
            double r;
            fillState64_19937(state);
            int p = 0;
            for (int i = 0; i < DSFMTAVX_SIZE * 2; i++) {
                w = state128[i];
                x = 2.0 * w.d[0] - 3.0;
                y = 2.0 * w.d[1] - 3.0;
                r = x * x + y * y;
                if (r > 1.0 || r == 0.0) {
                    continue;
                }
                double m = sqrt(-2.0 * log(r) / r);
                array[p] = mu + sigma * x * m;
                array[p + 1] = mu + sigma * y * m;
                p += 2;
            }
            return p;
        }

        bool self_test_19937(const cpu_feature_t& cf,
                             const params&,
                             double * state)
        {
#if defined(DEBUG)
            using std::cout;
            using std::dec;
            using std::endl;
#endif
            if (cf.avx2) {
                int asize = DSFMTAVX_SIZE * 4 * 2;
                int32_t * array1
                    = alignedAlloc<int32_t *>(asize * sizeof(int32_t));
                int32_t * array2
                    = alignedAlloc<int32_t *>(asize * sizeof(int32_t));
                sfmt_seed(reinterpret_cast<w256_t *>(state),
                          DSFMTAVX_SIZE, 0);
                initial_mask(reinterpret_cast<w256_t *>(state), DSFMTAVX_SIZE);
                fillarray64_maxint_19937(state, array1, 200, 1);
                sfmt_seed(reinterpret_cast<w256_t *>(state),
                          DSFMTAVX_SIZE, 0);
                initial_mask(reinterpret_cast<w256_t *>(state), DSFMTAVX_SIZE);
                fillarray256_maxint_19937(state, array2, 200, 1);
                for (int i = 0; i < asize; i++) {
                    if (array1[i] != array2[i] ||
                        array1[i] > 200 || array1[i] < 1) {
                        DMSG("fillArray_maxInt AVX test NG array mismatch");
#if defined(DEBUG)
                        cout << "i = " << dec << i << endl;
                        cout << "64:" << dec << array1[i] << endl;
                        cout << "128:" << dec << array2[i] << endl;
#endif
                        alignedFree(array1);
                        alignedFree(array2);
                        return false;
                    }
                }
                alignedFree(array1);
                alignedFree(array2);
            }
            return true;
        }
    }
}
