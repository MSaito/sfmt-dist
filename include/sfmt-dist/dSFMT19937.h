#pragma once
#ifndef SFMT_DIST_DSFMT19937_H
#define SFMT_DIST_DSFMT19937_H
#include <stdint.h>
#include <inttypes.h>
#include <sfmt-dist/cpu_feature.h>

namespace MersenneTwister {
    class DSFMT19937 {
    private:
        double * state;
        int index;
        cpu_feature_t cf;
        void fillState();
        enum {stateSize64 = 382};
    public:
        // for uniform_distribution interface
#if __cplusplus >= 201103L
        using result_type = double;
        using seed_type = uint32_t;
#else
        typedef double result_type;
        typedef uint32_t seed_type;
#endif
        DSFMT19937(uint32_t seedValue = 1234);
        DSFMT19937(uint32_t seedValue[], int length);
        ~DSFMT19937();
        void seed(uint32_t seedValue);
        void seed(uint32_t seedValue[], int length);
        int blockSize() {
            return stateSize64;
        }
        double generate() {
            return generateClose1Open2() - 1.0;
        }
        double operator()() {
            return generate();
        }
        double generateClose1Open2() {
            if (index >= stateSize64) {
                fillState();
                index = 0;
            }
            return state[index++];
        }
        double generateClose0Open1() {
            return generateClose1Open2() - 1.0;
        }
        double generateOpen0Close1() {
            return 2.0 - generateClose1Open2();
        }
        double generateOpen0Open1() {
            union {
                double d;
                uint64_t u;
            } x;
            x.d = generateClose1Open2();
            x.u |= 1;
            return x.d - 1.0;
        }
        const char * getIDString();
        void fillArray(double * array, int length);
        void fillArrayClose0Open1(double * array, int length) {
            fillArray(array, length);
        }
        void fillArrayClose1Open2(double * array, int length);
        void fillArrayOpen0Close1(double * array, int length);
        void fillArrayOpen0Open1(double * array, int length);
        void fillArrayMaxInt(int32_t * array, uint64_t rmax, int32_t min);
        int fillArrayNormalDist(double * array, double mu, double sigma);
        bool selfTest();
    };
}
#endif // SFMT_DIST_DSFMT19937_H
