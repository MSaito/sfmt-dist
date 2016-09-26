#pragma once
#ifndef SFMT_DIST_KISS_HPP
#define SFMT_DIST_KISS_HPP
/*
 * This file is copied from
 * http://people.sc.fsu.edu/~jburkardt/cpp_src/ziggurat/ziggurat.html.
 * Modified by Mutsuo Saito (saito@manieth.com)
 *
 * License: the GNU LGPL license.
 *
 * Authors: George Marsaglia, Wai Wan Tsang.
 */
namespace Ziggurat {

    class KISS {
    public:
        KISS() {
            jsr = 123456789;
            jcong = 234567891;
            w = 345678912;
            z = 456789123;
            jz = 0;
        }
        void seed(uint32_t value) {
            uint32_t st[4] = {value, 0, 0, 0};
            for (int i = 1; i < 8; i++) {
                st[i & 3] ^= i + UINT32_C(1812433253)
                    * (st[(i - 1) & 3]
                       ^ (st[(i - 1) & 3] >> 30));
            }
            jsr = st[0];
            jcong = st[1];
            w = st[2];
            z = st[3];
        }
        void seed(uint32_t value[], int length) {
            if (length <= 0) {
                seed(0);
            } else if (length == 1) {
                seed(value[0]);
            } else if (length >= 4) {
                jsr = value[0];
                jcong = value[1];
                w = value[2];
                z = value[3];
            } else {
                seed(value[0]);
            }
        }
        // 0 < x < 1;
        //float float01() {
        //  return 0.5 + static_cast<int32_t>(generate_uint32()) * 0.2328306e-09;
        //}

        uint32_t generate_uint32() {
            return (mwc() ^ cong()) + shr3();
        }
        uint32_t operator()() {
            return generate_uint32();
        }
    private:
        uint32_t jcong;
        uint32_t jz;
        uint32_t jsr;
        uint32_t w;
        uint32_t z;
        uint32_t mwc() {
            return (znew() << 16) + wnew();
        }
        uint32_t cong() {
            jcong = 69069 * jcong + 1234567;
            return jcong;
        }
        uint32_t shr3() {
            jz = jsr;
            jsr ^= ( jsr << 13 );
            jsr ^= ( jsr >> 17 );
            jsr ^= ( jsr << 5 );
            return jz + jsr;
        }
        uint32_t wnew() {
            w = 18000 * ( w & 65535 ) + ( w >> 16 );
            return w;
        }
        uint32_t znew() {
            z = 36969 * ( z & 65535 ) + ( z >> 16 );
            return z;
        }
    };
}
#endif // SFMT_DIST_KISS_HPP
