#pragma once
#ifndef SFMT_DIST_CPU_FEATURE_H
#define SFMT_DIST_CPU_FEATURE_H

namespace MersenneTwister {
    struct cpu_feature_t {
        unsigned avx512f : 1;
        unsigned avx2 : 1;
        unsigned avx : 1;
        unsigned sse4_2 : 1;
        unsigned sse4_1 : 1;
        unsigned ssse3 : 1;
        unsigned sse2 : 1;
    };

    cpu_feature_t cpu_feature(void);
    void print_cpu_feature(cpu_feature_t cf);
}
#endif //SFMT_DIST_CPU_FEATURE_H
