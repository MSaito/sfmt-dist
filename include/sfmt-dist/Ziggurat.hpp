#pragma once
#ifndef SFMT_DIST_ZIGGURAT_HPP
#define SFMT_DIST_ZIGGURAT_HPP
/*
 * This file is copied from
 * http://people.sc.fsu.edu/~jburkardt/cpp_src/ziggurat/ziggurat.html.
 *
 * Modified by Mutsuo Saito (saito@manieth.com)
 *
 * License: the GNU LGPL license.
 *
 * Authors: George Marsaglia, Wai Wan Tsang.
 */
#include <cmath>
#include <cstdlib>
#include <inttypes.h>
#include <sfmt-dist/ziggurat1.h>

#if defined(DEBUG)
#include <iostream>
#include <iomanip>
using namespace std;
#endif

namespace Ziggurat {

    static inline float float01(int32_t x) {
        return 0.5 + x * 0.2328306e-09;
    }

    /*
     * Ziggurat algorithm for Normal Distribution
     */
    template<typename E>
    class ZigNormal {
    public:
        ZigNormal(float mu, float sigma, uint32_t seed_value = 1234) {
            this->mu = mu;
            this->sigma = sigma;
            engine.seed(seed_value);
        }
        float generate() {
            int32_t hz = rint32();
            uint32_t iz = hz & 127;
#if defined(DEBUG) && 0
            cout << setprecision(10);
            cout << "DEBUG:hz = " << dec << hz << endl;
            cout << "DEBUG:iz = " << dec << iz << endl;
            cout << "DEBUG:fabs(hz) = " << dec << fabs(hz) << endl;
            cout << "DEBUG:intabs(hz) = " << dec << intabs(hz) << endl;
            cout << "DEBUG:kn[iz] = " << dec << kn[iz] << endl;
#endif
            if (intabs(hz) < kn[iz]) {
                return hz * wn[iz] * sigma + mu;
            } else {
                return nfix(hz, iz) * sigma + mu;
            }
        }

        float operator()() {
            return generate();
        }

        void seed(uint32_t value) {
            engine.seed(value);
        }
        void seed(uint32_t value[], int length) {
            engine.seed(value, length);
        }
    private:
        float mu;
        float sigma;
        /*
         * copied from
         * http://graphics.stanford.edu/~seander/bithacks.html#IntegerAbs
         */
        static uint32_t intabs(int32_t x) {
            int32_t mask = x >> 31;
            return (x + mask) ^ mask;
        }

        E engine;

        int32_t rint32() {
            return engine();
        }

        float nfix(int32_t hz, uint32_t iz) {
            const float r = 3.442620;
            static float x;
            static float y;

            for ( ; ; ) {
#if defined(DEBUG)
                cout << "DEBUG:nfix" << endl;
                cout << "DEBUG:hz = " << dec << hz << endl;
                cout << "DEBUG:iz = " << dec << iz << endl;
#endif
                // IZ = 0 handles the base strip.
                x = static_cast<float>(hz * wn[iz]);
                if ( iz == 0 ) {
                    do {
                        x = - log(float01(rint32())) * 0.2904764;
                        y = - log(float01(rint32()));
                    } while ( y + y < x * x );
                    if (0 < hz) {
                        return r + x;
                    } else {
                        return -r - x;
                    }
                }
                // 0 < IZ, handle the wedges of other strips.
                bool result = fn[iz] + float01(rint32()) * (fn[iz-1] - fn[iz])
                    < exp(- 0.5 * x * x);
                if (result) {
                    return x;
                }
                // Initiate, try to exit the loop.
                hz = rint32();
                iz = hz & 127;
#if defined(DEBUG)
                cout << "DEBUG:hz = " << dec << hz << endl;
                cout << "DEBUG:iz = " << dec << iz << endl;
#endif
                if (intabs(hz) < kn[iz]) {
                    return static_cast<float>(hz * wn[iz]);
                }
            }
        }
    };

    /*
     * Ziggurat algorithm for Expornential Distribution
     */
    template<typename E>
    class ZigExp {
    public:
        float generate() {
            uint32_t jz = ruint32();
            uint32_t iz = jz & 255;
            if (jz < ke[iz]) {
                return jz * we[iz];
            } else {
                return efix(jz, iz);
            }
        }

        float operator()() {
            return generate();
        }

        void seed(uint32_t value) {
            engine.seed(value);
        }
        void seed(uint32_t value[], int length) {
            engine.seed(value, length);
        }
    private:
        E engine;

        uint32_t ruint32() {
            return engine();
        }

        float efix(uint32_t jz, uint32_t iz) {
            float x;
            for ( ; ; ) {
#if defined(DEBUG)
                cout << "DEBUG:efix" << endl;
                cout << "DEBUG:jz = " << dec << jz << endl;
                cout << "DEBUG:iz = " << dec << iz << endl;
#endif
                //IZ = 0.
                if (iz == 0) {
                    return 7.69711 - log(float01(ruint32()));
                }
                x = jz * we[iz];
                if (fe[iz] + float01(ruint32()) * (fe[iz-1] - fe[iz]) < exp(-x)) {
                    return x;
                }
                // Initiate, try to exit the loop.
                jz = ruint32();
                iz = jz & 255;
#if defined(DEBUG)
                cout << "DEBUG:jz = " << dec << jz << endl;
                cout << "DEBUG:iz = " << dec << iz << endl;
#endif
                if (jz < ke[iz]) {
                    return jz * we[iz];
                }
            }
        }
    };

}
#endif // SFMT_DIST_ZIGGURAT_HPP
