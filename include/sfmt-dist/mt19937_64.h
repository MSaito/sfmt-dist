#pragma once
#ifndef SFMT_DIST_MT19937_64_H
#define SFMT_DIST_MT19937_64_H
#include <stdint.h>
#include <inttypes.h>
#include <sfmt-dist/cpu_feature.h>

namespace MersenneTwister {
    class MT19937_64 {
    private:
        enum {stateSize = 312};
        //uint64_t mata;
        uint64_t * state;
        int index;
        cpu_feature_t cf;
        uint64_t tempering(uint64_t x) {
            x ^= (x >> 29) & UINT64_C(0x5555555555555555);
            x ^= (x << 17) & UINT64_C(0x71D67FFFEDA60000);
            x ^= (x << 37) & UINT64_C(0xFFF7EEE000000000);
            x ^= (x >> 43);
            return x;
        }
        void fillState();
    public:
        using result_type = uint64_t;
        static uint64_t min() {return 0;}
        static uint64_t max() {return UINT64_MAX;}
        MT19937_64(uint64_t seedValue = 5489);
        ~MT19937_64();
        void seed(uint64_t seedValue);
        void seed(uint64_t seedValue[], int length);
        int blockSize() {
            return stateSize;
        }
        uint64_t generate() {
            if (index >= stateSize) {
                fillState();
                index = 0;
            }
            return tempering(state[index++]);
        }
        uint64_t operator()() {
            return generate();
        }

        /* generates a random number on [0,1]-real-interval */
        double real1()
        {
            return (generate() >> 11) * (1.0/9007199254740991.0);
        }

        /* generates a random number on [0,1)-real-interval */
        double real2()
        {
            return (generate() >> 11) * (1.0/9007199254740992.0);
        }

        /* generates a random number on (0,1)-real-interval */
        double real3()
        {
            return ((generate() >> 12) + 0.5) * (1.0/4503599627370496.0);
        }

        void fillArray(uint64_t * array, int length);
        static bool selfTest();
    };
}
#endif // SFMT_DIST_MT19937_64_H
