#pragma once
#ifndef SFMT_DIST_UNIFORMINTFROMDOUBLE_H
#define SFMT_DIST_UNIFORMINTFROMDOUBLE_H

#include <cstddef>
#include <sfmt-dist/aligned_alloc.h>
#include <sfmt-dist/dSFMT19937.h>

namespace MersenneTwister {

    /*
     * use internal buffer
     */
    template<typename T, typename E = DSFMT19937>
    class UniformIntFromDouble {
    public:
        UniformIntFromDouble(T start, T end,
                             typename E::seed_type seed = 1234) {
            engine.seed(seed);
            range_start = start;
            max = end - start;
            buffer_size = engine.blockSize() * 2;
            index = buffer_size;
            buffer = alignedAlloc<int32_t *>(buffer_size * sizeof(int32_t));
        }

        ~UniformIntFromDouble() {
            alignedFree(buffer);
        }

        T generate() {
            if (index >= buffer_size) {
                engine.fillArrayMaxInt(buffer, max, range_start);
                index = 0;
            }
            return static_cast<T>(buffer[index++]);
        }

        T operator()() {
            return generate();
        }

    private:
        size_t index;
        size_t buffer_size;
        E engine;
        int32_t range_start;
        uint64_t max;
        int32_t * buffer;
    };


}
#endif // SFMT_DIST_UNIFORMINTFROMDOUBLE_H
