#include "sfmt-dist.h"
#include <sfmt-dist/dSFMTAVX19937.h>
#include "w256.h"
#include "w128.h"

#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdint.h>
#include <cmath>
#include <cfloat>

#define DSFMTAVX_MEXP 19937

#include "dsfmt_avx_params.h"
#include "dsfmt_avx_common.h"

namespace MersenneTwister {
    using namespace std;

    DSFMTAVX19937::DSFMTAVX19937(int, uint32_t seedValue)
    {
        cf = cpu_feature();
        stateSize = DSFMTAVX_SIZE;
        stateSize64 = stateSize * 4;
        size_t alloc_size = (stateSize + 1) * 4 * sizeof(uint64_t);
        state = alignedAlloc<double *>(alloc_size);
        if (state == NULL) {
            throw new runtime_error("can't get aligned memory");
        }
        seed(seedValue);
    }

    DSFMTAVX19937::DSFMTAVX19937(uint32_t seedValue)
    {
        cf = cpu_feature();
        stateSize = DSFMTAVX_SIZE;
        stateSize64 = stateSize * 4;
        size_t alloc_size = (stateSize + 1) * 4 * sizeof(uint64_t);
        state = alignedAlloc<double *>(alloc_size);
        if (state == NULL) {
            throw new runtime_error("can't get aligned memory");
        }
        seed(seedValue);
    }

    DSFMTAVX19937::DSFMTAVX19937(uint32_t seedValue[], int length)
    {
        cf = cpu_feature();
        stateSize = DSFMTAVX_SIZE;
        stateSize64 = stateSize * 4;
        size_t alloc_size = (stateSize + 1) * 4 * sizeof(uint64_t);
        state = alignedAlloc<double *>(alloc_size);
        if (state == NULL) {
            throw new runtime_error("can't get aligned memory");
        }
        seed(seedValue, length);
    }

    DSFMTAVX19937::~DSFMTAVX19937()
    {
        alignedFree(state);
    }

#if defined(DEBUG)
    void DSFMTAVX19937::d_p()
    {
        cout << "debug_print" << endl;
        cout << "array_size = " << dec << array_size << endl;
        cout << "mexp = " << dec << mexp << endl;
        cout << "pos1 = " << dec << pos1 << endl;
        cout << "mask[0] = " << hex << mask.u64[0] << endl;
        cout << "mask[1] = " << hex << mask.u64[1] << endl;
        cout << "mask[2] = " << hex << mask.u64[2] << endl;
        cout << "mask[3] = " << hex << mask.u64[3] << endl;
        w256_t * state256 = reinterpret_cast<w256_t *>(state);
        for (int i = 0; i < stateSize + 1; i++) {
            for (int j = 0; j < 4; j++) {
                cout << hex << setfill('0') << setw(16) << state256[i].u64[j]
                     << " ";
            }
            cout << endl;
        }
    }
#endif

    const string DSFMTAVX19937::getIDString()
    {
        return ::getIDString();
    }

    /**
     * This function initializes the internal state array with a 32-bit
     * integer seed.
     *
     * @param sfmt SFMT internal state
     * @param seed a 32-bit integer used as the seed.
     */
    void DSFMTAVX19937::seed(uint32_t seedValue)
    {
        w256_t * state256 = reinterpret_cast<w256_t *>(state);
        init(state256, seedValue);
        index = stateSize64;
    }

    /**
     * This function initializes the internal state array,
     * with an array of 32-bit integers used as the seeds
     * @param sfmt SFMT internal state
     * @param init_key the array of 32-bit integers, used as a seed.
     * @param key_length the length of init_key.
     */
    void DSFMTAVX19937::seed(uint32_t *init_key, int key_length)
    {
        w256_t * state256 = reinterpret_cast<w256_t *>(state);
        init(state256, init_key, key_length);
        index = stateSize64;
    }

    void DSFMTAVX19937::fillState()
    {
        if (cf.avx2) {
            fillState256(state);
        } else {
            fillState64(state);
        }
        index = 0;
    }

    void DSFMTAVX19937::fillArray(double * array, int length)
    {
        DMSG("start fillArray");
        if (length < stateSize64 || index != stateSize64 || (length % 4 != 0)) {
            for (int i = 0; i < length; i++) {
                array[i] = generate();
            }
            return;
        }
        DMSG("fillArray step 2");
        int align = reinterpret_cast<uintptr_t>(array) % 32;
        if ((align == 0) && cf.avx2) {
            DMSG("fillArray step 3");
            fillArray256_c0o1(state, array, length / 4);
        } else {
            DMSG("fillArray step 4");
            fillArray64_c0o1(state, array, length / 4);
        }
    }

    void DSFMTAVX19937::fillArrayClose1Open2(double * array, int length)
    {
        DMSG("start fillArray");
        if (length < stateSize64 || index != stateSize64 || (length % 4 != 0)) {
            for (int i = 0; i < length; i++) {
                array[i] = generateClose1Open2();
            }
            return;
        }
        DMSG("fillArray step 2");
        int align = reinterpret_cast<uintptr_t>(array) % 32;
        if ((align == 0) && cf.avx2) {
            DMSG("fillArray step 3");
            fillArray256_c1o2(state, array, length / 4);
        } else {
            DMSG("fillArray step 4");
            fillArray64_c1o2(state, array, length / 4);
        }
    }

    void DSFMTAVX19937::fillArrayOpen0Close1(double * array, int length)
    {
        DMSG("start fillArray");
        if (length < stateSize64 || index != stateSize64 || (length % 4 != 0)) {
            for (int i = 0; i < length; i++) {
                array[i] = generateOpen0Close1();
            }
            return;
        }
        DMSG("fillArray step 2");
        int align = reinterpret_cast<uintptr_t>(array) % 32;
        if ((align == 0) && cf.avx2) {
            DMSG("fillArray step 3");
            fillArray256_o0c1(state, array, length / 4);
        } else {
            DMSG("fillArray step 4");
            fillArray64_o0c1(state, array, length / 4);
        }
    }

    void DSFMTAVX19937::fillArrayOpen0Open1(double * array, int length)
    {
        DMSG("start fillArray");
        if (length < stateSize64 || index != stateSize64 || (length % 4 != 0)) {
            for (int i = 0; i < length; i++) {
                array[i] = generateOpen0Open1();
            }
            return;
        }
        DMSG("fillArray step 2");
        int align = reinterpret_cast<uintptr_t>(array) % 32;
        if ((align == 0) && cf.avx2) {
            DMSG("fillArray step 3");
            fillArray256_o0o1(state, array, length / 4);
        } else {
            DMSG("fillArray step 4");
            fillArray64_o0o1(state, array, length / 4);
        }
    }

    int DSFMTAVX19937::getMersenneExponent()
    {
        return DSFMTAVX_MEXP; //19937
    }

    /*
     * array の要素数はstateの要素数の2倍
     */
    void DSFMTAVX19937::fillArrayMaxInt(int32_t * array,
                                        uint64_t rmax, int32_t min)
    {
        int align = reinterpret_cast<uintptr_t>(array) % 32;
        if ((align == 0) && cf.avx2) {
            fillArray256_maxint(state, array, rmax, min);
        } else if ((align % 16 == 0) && cf.sse2) {
            //fillArray128_maxint(state, array, rmax, min);
        } else {
            fillArray64_maxint(state, array, rmax, min);
        }
    }

    /*
     * array の要素数はstateの要素数と同じ
     */
    int DSFMTAVX19937::fillArrayNormalDist(double * array,
                                           double mu, double sigma)
    {

        int align = reinterpret_cast<uintptr_t>(array) % 32;
        if ((align == 0) && cf.avx2) {
            return fillArray256_boxmuller(state, array, mu, sigma);
        } else if ((align % 16 == 0) && cf.sse2) {
            //return fillArray128_boxmuller(state, array, mu, sigma);
            return fillArray64_boxmuller(state, array, mu, sigma);
        } else {
            return fillArray64_boxmuller(state, array, mu, sigma);
        }
    }

    // static
    bool DSFMTAVX19937::selfTest()
    {
        DMSG("selfTest Start");
        DMSG("fillArray_maxInt test Start");
        cpu_feature_t cf = cpu_feature();
        DSFMTAVX19937 mt1(0);
        DSFMTAVX19937 mt2(0);
        if (cf.avx2) {
            DMSG("fillArray_maxInt AVX test Start");
            int asize = mt1.blockSize() * 2;
            int32_t * array1 = alignedAlloc<int32_t *>(asize * sizeof(double));
            int32_t * array2 = alignedAlloc<int32_t *>(asize * sizeof(double));
            mt2.cf.avx2 = 0;
            mt2.cf.sse2 = 0;
            mt1.fillArrayMaxInt(array1, 200, 1);
            mt2.fillArrayMaxInt(array2, 200, 1);
            for (int i = 0; i < asize; i++) {
                if (fabs(array1[i] - array2[i]) >= DBL_EPSILON) {
                    DMSG("fillArray_maxInt AVX test NG array mismatch");
#if defined(DEBUG)
                    std::cout << "i = " << std::dec << i << std::endl;
#endif
                    return false;
                }
            }
        }
        DMSG("fillArray_maxInt test OK");
        DMSG("fillArray_boxmuller test Start");
        if (cf.avx2) {
            int asize = mt1.blockSize();
            double * array1 = alignedAlloc<double *>(asize * sizeof(double));
            double * array2 = alignedAlloc<double *>(asize * sizeof(double));
            mt1.seed(0);
            mt2.seed(0);
            int c1 = mt1.fillArrayNormalDist(array1, 0, 1.0);
            int c2 = mt2.fillArrayNormalDist(array2, 0, 1.0);
            if (c1 != c2) {
                DMSG("fillArray_boxmuller test NG c1 != c2");
                return false;
            }
            for (int i = 0; i < asize; i++) {
                if (fabs(array1[i] - array2[i]) >= DBL_EPSILON) {
                    DMSG("fillArray_boxmuller test NG array mismatch");
#if defined(DEBUG)
                    std::cout << "i = " << std::dec << i << std::endl;
#endif
                    return false;
                }
            }
            DMSG("fillArray_boxmuller test OK");
        }
        return true;
    }

}
