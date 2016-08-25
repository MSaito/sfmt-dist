#pragma once
#ifndef MT_NORMALFROMDOUBLE_HPP
#define MT_NORMALFROMDOUBLE_HPP

#include <cstddef>
#include <sfmt-dist/aligned_alloc.h>
#include <sfmt-dist/dSFMT19937.h>

namespace MersenneTwister {

    /*
     * use internal buffer
     */
    template<typename E = DSFMT19937>
    class NormalFromDouble {
    public:
        NormalFromDouble(double mu, double sigma,
                             typename E::seed_type seed = 1234) {
            engine.seed(seed);
            this->mu = mu;
            this->sigma = sigma;
            int asize = engine.blockSize();
            buffer = alignedAlloc<double *>(asize * sizeof(double));
            index = 0;
            buffer_size = 0;
        }

        ~NormalFromDouble() {
            alignedFree(buffer);
        }

        double generate() {
            if (index >= buffer_size) {
                buffer_size = engine.fillArrayNormalDist(buffer, mu, sigma);
                index = 0;
            }
            return buffer[index++];
        }

        double operator()() {
            return generate();
        }

    private:
        size_t index;
        size_t buffer_size;
        E engine;
        double mu;
        double sigma;
        double * buffer;
    };


}
#endif // MT_NORMALFROMDOUBLE_HPP
