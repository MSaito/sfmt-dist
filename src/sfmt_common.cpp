#include <stdint.h>
#include "config.h"
#include "sfmt_common.h"

namespace {
    /**
     * This function represents a function used in the initialization
     * by init_by_array
     * @param x 32-bit integer
     * @return 32-bit integer
     */
    inline uint32_t ini_func1(uint32_t x)
    {
        return (x ^ (x >> 27)) * UINT32_C(1664525);
    }

    /**
     * This function represents a function used in the initialization
     * by init_by_array
     * @param x 32-bit integer
     * @return 32-bit integer
     */
    inline uint32_t ini_func2(uint32_t x)
    {
        return (x ^ (x >> 27)) * UINT32_C(1566083941);
    }

    inline uint64_t parity(uint64_t x)
    {
#if HAVE_BUILTIN_PARITYLL
        return __builtin_parityll(x);
#else
        x ^= x >> 32;
        x ^= x >> 16;
        x ^= x >> 8;
        x ^= x >> 4;
        x ^= x >> 2;
        x ^= x >> 1;
        return x & 1;
#endif
    }
}

namespace MersenneTwister {
    using namespace std;

    /**
     * This function certificate the period of 2^{SFMT_MEXP}-1.
     * @param dsfmt dsfmt state vector.
     */
    void sfmt_period_certification(w256_t * vec, const w256_t *pcv)
    {
        uint64_t inner = 0;
        for (int i = 0; i < 4; i++) {
            inner ^= vec->u64[i] & pcv->u64[i];
        }
        inner = parity(inner);
        /* check OK */
        if (inner == 1) {
            return;
        }
        /* check NG, and modification */
        if ((pcv->u64[0] & 1) == 1) {
            vec->u64[0] ^= 1;
            return;
        }
        int i;
        int j;
        uint64_t work;
        for (i = 0; i < 4; i++) {
            work = 1;
            for (j = 0; j < 64; j++) {
                if ((work & pcv->u64[i]) != 0) {
                    vec->u64[i] ^= work;
                    return;
                }
                work = work << 1;
            }
        }
    }

    /**
     * This function initializes the internal state array with a 32-bit
     * integer seed.
     * @param dsfmt dsfmt state vector.
     * @param seed a 32-bit integer used as the seed.
     * @param mexp caller's mersenne expornent
     */
    void sfmt_seed(w256_t * state, int state_size, uint32_t seed)
    {
        int i;
        uint32_t *psfmt;

        int size = state_size;
        psfmt = &state[0].u32[0];
        psfmt[0] = seed;
        for (i = 1; i < size * 8; i++) {
            psfmt[i] = 1812433253UL
                * (psfmt[i - 1] ^ (psfmt[i - 1] >> 30)) + i;
        }
    }

    /**
     * This function initializes the internal state array,
     * with an array of 32-bit integers used as the seeds
     * @param dsfmt dsfmt state vector.
     * @param init_key the array of 32-bit integers, used as a seed.
     * @param key_length the length of init_key.
     * @param mexp caller's mersenne expornent
     */
    void sfmt_seed(w256_t * state, int state_size,
                   uint32_t init_key[], int key_length)
    {
        int i, j, count;
        uint32_t r;
        uint32_t *psfmt32;
        int lag;
        int mid;
        int size = state_size * 2 * 4;

        if (size >= 623) {
            lag = 11;
        } else if (size >= 68) {
            lag = 7;
        } else if (size >= 39) {
            lag = 5;
        } else {
            lag = 3;
        }
        mid = (size - lag) / 2;

        psfmt32 = &state[0].u32[0];
        // check here
        memset(state, 0x8b, sizeof(uint32_t) * size);
        if (key_length + 1 > size) {
            count = key_length + 1;
        } else {
            count = size;
        }
        r = ini_func1(psfmt32[0] ^ psfmt32[mid % size]
                      ^ psfmt32[(size - 1) % size]);
        psfmt32[mid % size] += r;
        r += key_length;
        psfmt32[(mid + lag) % size] += r;
        psfmt32[0] = r;
        count--;
        for (i = 1, j = 0; (j < count) && (j < key_length); j++) {
            r = ini_func1(psfmt32[i]
                          ^ psfmt32[(i + mid) % size]
                          ^ psfmt32[(i + size - 1) % size]);
            psfmt32[(i + mid) % size] += r;
            r += init_key[j] + i;
            psfmt32[(i + mid + lag) % size] += r;
            psfmt32[i] = r;
            i = (i + 1) % size;
        }
        for (; j < count; j++) {
            r = ini_func1(psfmt32[i]
                          ^ psfmt32[(i + mid) % size]
                          ^ psfmt32[(i + size - 1) % size]);
            psfmt32[(i + mid) % size] += r;
            r += i;
            psfmt32[(i + mid + lag) % size] += r;
            psfmt32[i] = r;
            i = (i + 1) % size;
        }
        for (j = 0; j < size; j++) {
            r = ini_func2(psfmt32[i]
                          + psfmt32[(i + mid) % size]
                          + psfmt32[(i + size - 1) % size]);
            psfmt32[(i + mid) % size] ^= r;
            r -= i;
            psfmt32[(i + mid + lag) % size] ^= r;
            psfmt32[i] = r;
            i = (i + 1) % size;
        }
    }
}
