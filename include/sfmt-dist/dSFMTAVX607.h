#pragma once
#ifndef SFMT_DIST_DSFMTAVX607_H
#define SFMT_DIST_DSFMTAVX607_H
#include <stdint.h>
#include <string>
#include <inttypes.h>
#include <sfmt-dist/cpu_feature.h>

namespace MersenneTwister {
    class DSFMTAVX607 {
    private:
        double * state;
        int index;
        cpu_feature_t cf;
        void fillState();
        int stateSize64;
        int stateSize;
    public:
#if __cplusplus >= 201103L
        using result_type = double;
        using seed_type = uint32_t;
#else
        typedef double result_type;
        typedef uint32_t seed_type;
#endif
        DSFMTAVX607(int mexp, uint32_t seedValue);
        DSFMTAVX607(uint32_t seedValue = 1234);
        DSFMTAVX607(uint32_t seedValue[], int length);
        ~DSFMTAVX607();
#if defined(DEBUG)
        void d_p();
#endif
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
        const std::string getIDString();
        void fillArray(double * array, int length);
        void fillArrayClose0Open1(double * array, int length) {
            fillArray(array, length);
        }
        void fillArrayClose1Open2(double * array, int length);
        void fillArrayOpen0Close1(double * array, int length);
        void fillArrayOpen0Open1(double * array, int length);
        int getMersenneExponent();
        void fillArrayMaxInt(int32_t * array, uint64_t rmax, int32_t min);
        int fillArrayNormalDist(double * array, double mu, double sigma);
        bool selfTest();
    };
}
#endif // SFMT_DIST_DSFMTAVX607_H
