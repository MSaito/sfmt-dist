#include "sfmt-dist.h"
#include <inttypes.h>
#include <climits>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <cerrno>
#include <random>
#include <sfmt-dist/sfmt19937.h>
#include <sfmt-dist/dSFMT19937.h>
#include <sfmt-dist/dSFMTAVX607.h>
#include <sfmt-dist/dSFMTAVX19937.h>
#include <sfmt-dist/mt19937.h>
#include <sfmt-dist/normalFromDouble.hpp>
#include <sfmt-dist/uniformIntFromDouble.hpp>
//#include "dSFMTAVX.hpp"
#include "getopt.hpp"

using std::cout;
using std::endl;
using std::setprecision;
using std::dec;

template<typename Engine>
int normal(uint64_t count, uint32_t seed)
{
    Engine engine(seed);
    std::normal_distribution<> dist(0.0, 1.0) ;
    cout.setf(std::ios::fixed);
    for (uint64_t i = 0; i < count; ++i) {
        cout << dec << setprecision(20) << dist(engine) << endl;
    }
    return 0;
}

template<typename Engine>
int uniform(uint64_t count, uint32_t seed, int32_t start, int32_t end)
{
    Engine engine(seed);
    std::uniform_int_distribution<uint32_t> dist(start, end) ;
    for (uint64_t i = 0; i < count; ++i) {
        cout << dec << dist(engine) << endl;
    }
    return 0;
}

template<typename Engine>
int d_normal(uint64_t count, uint32_t seed)
{
    using MersenneTwister::NormalFromDouble;

    NormalFromDouble<Engine> dist(0.0, 1.0, seed) ;
    cout.setf(std::ios::fixed);
    for (uint64_t i = 0; i < count; ++i) {
        cout << dec << setprecision(20) << dist() << endl;
    }
    return 0;
}

template<typename Engine>
int d_uniform(uint64_t count, uint32_t seed, int32_t start, int32_t end)
{
    using MersenneTwister::UniformIntFromDouble;

    UniformIntFromDouble<Engine> dist(start, end, seed) ;
    cout.setf(std::ios::fixed);
    for (uint64_t i = 0; i < count; ++i) {
        cout << dec << dist() << endl;
    }
    return 0;
}

int main(int argc, char * argv[])
{
    options opt;
    opt.count = 1000;
    opt.seed = 1234;
    opt.start = 0;
    opt.end = 200;
    if (!opt.parse(argc, argv)) {
        return -1;
    }
    uint64_t count = opt.count;
    uint32_t seed = opt.seed;
    int32_t start = opt.start;
    int32_t end = opt.end;
    if (opt.uniform_dist) {
        cout << "#uniform distribution[" << start << "," << end << "] :";
        switch (opt.generator_kind) {
        case 'm':
            cout << "std::mt19937" << endl;
            uniform<std::mt19937>(count, seed, start, end);
            break;
        case 'M':
            cout << "MersenneTwister::MT19937" << endl;
            uniform<MersenneTwister::MT19937>(count, seed, start, end);
            break;
        case 'S':
            cout << "MersenneTwister::SFMT19937" << endl;
            uniform<MersenneTwister::SFMT19937>(count, seed, start, end);
            break;
        case 'd':
            cout << "MersenneTwister::dSFMT19937" << endl;
            d_uniform<MersenneTwister::DSFMT19937>(count, seed, start, end);
            break;
        case 'a':
            cout << "MersenneTwister::dSFMTAVX607" << endl;
            d_uniform<MersenneTwister::DSFMTAVX607>(count, seed, start, end);
            break;
        case 'A':
            cout << "MersenneTwister::dSFMTAVX19937" << endl;
            d_uniform<MersenneTwister::DSFMTAVX19937>(count,
                                                      seed, start, end);
            break;
        default:
            break;
        }
    } else if (opt.normal_dist) {
        cout << "#normal distribution:";
        switch (opt.generator_kind) {
        case 'm':
            cout << "std::mt19937" << endl;
            normal<std::mt19937>(count, seed);
            break;
        case 'M':
            cout << "MersenneTwister::mt19937" << endl;
            normal<MersenneTwister::MT19937>(count, seed);
            break;
        case 'S':
            cout << "MersenneTwister::SFMT19937" << endl;
            normal<MersenneTwister::SFMT19937>(count, seed);
            break;
        case 'd':
            cout << "MersenneTwister::dSFMT19937" << endl;
            d_normal<MersenneTwister::DSFMT19937>(count, seed);
            break;
        case 'a':
            cout << "MersenneTwister::dSFMTAVX607" << endl;
            d_normal<MersenneTwister::DSFMTAVX607>(count, seed);
            break;
        case 'A':
            cout << "MersenneTwister::dSFMTAVX19937" << endl;
            d_normal<MersenneTwister::DSFMTAVX19937>(count, seed);
            break;
        default:
            break;
        }
    }
    return 0;
}
