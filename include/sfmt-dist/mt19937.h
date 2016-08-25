#pragma once
#ifndef SFMT_DIST_MT19937_H
#define SFMT_DIST_MT19937_H
#include <stdint.h>
#include <inttypes.h>
#include <sfmt-dist/cpu_feature.h>

namespace MersenneTwister {
    class MT19937 {
    private:
        enum {stateSize = 624};
        //uint32_t mata;
        uint32_t * state;
        int index;
        cpu_feature_t cf;
        uint32_t tempering(uint32_t y) {
            y ^= (y >> 11);
            y ^= (y << 7) & UINT32_C(0x9d2c5680);
            y ^= (y << 15) & UINT32_C(0xefc60000);
            y ^= (y >> 18);
            return y;
        }
        void fillState();
    public:
        using result_type = uint32_t;
        static uint32_t min() {return 0;}
        static uint32_t max() {return UINT32_MAX;}

        MT19937(uint32_t seedValue = 5489);
        ~MT19937();
        void seed(uint32_t seedValue);
        void seed(uint32_t seedValue[], int length);
        int blockSize() {
            return stateSize;
        }
        uint32_t generate() {
            if (index >= stateSize) {
                fillState();
                index = 0;
            }
            return tempering(state[index++]);
        }
        uint32_t operator()() {
            return generate();
        }
        /* generates a random number on [0,1]-real-interval */
        double genrand_real1()
        {
            return generate() * (1.0/4294967295.0);
            /* divided by 2^32-1 */
        }

        /* generates a random number on [0,1)-real-interval */
        double real2() {
            return generate() * (1.0/4294967296.0);
            /* divided by 2^32 */
        }

        /* generates a random number on (0,1)-real-interval */
        double real3()
        {
            return (((double)generate()) + 0.5)*(1.0/4294967296.0);
            /* divided by 2^32 */
        }

        /* generates a random number on [0,1) with 53-bit resolution*/
        double res53()
        {
            uint32_t a = generate() >> 5;
            uint32_t b = generate() >> 6;
            return (a * 67108864.0 + b) * (1.0/9007199254740992.0);
        }
        /* These real versions are due to Isaku Wada */

        void fillArray(uint32_t * array, int length);
        static bool selfTest();
    };
}
#endif // SFMT_DIST_MT19937_H
