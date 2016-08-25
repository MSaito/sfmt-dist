#pragma once
#ifndef MT_SFMT19937_H
#define MT_SFMT19937_H
#include <stdint.h>
#include <inttypes.h>
#include <sfmt-dist/cpu_feature.h>

namespace MersenneTwister {
    class SFMT19937 {
    private:
        uint32_t * state;
        int index;
        cpu_feature_t cf;
        void fillState();
        enum {stateSize32 = 624};
    public:
        // for uniform_distribution interface
        using result_type = uint32_t;
        static uint32_t min() {return 0;}
        static uint32_t max() {return UINT32_MAX;}

        SFMT19937(uint32_t seedValue = 1234);
        SFMT19937(uint32_t seedValue[], int length);
        ~SFMT19937();
        void seed(uint32_t seedValue);
        void seed(uint32_t seedValue[], int length);
        int blockSize() {
            return stateSize32;
        }
        uint32_t generate() {
            if (index >= stateSize32) {
                fillState();
                index = 0;
            }
            return state[index++];
        }
        uint32_t operator()() {
            return generate();
        }
        void fillArray(uint32_t * array, int length, bool skip = false);
        const char * getIDString();
    };
}
#endif // MT_SFMT19937_H
