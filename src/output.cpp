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
#include <sfmt-dist/mt19937.h>
#include <sfmt-dist/normalFromDouble.hpp>
#include <sfmt-dist/uniformIntFromDouble.hpp>

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
int uniform(uint64_t count, uint32_t seed)
{
    Engine engine(seed);
    std::uniform_int_distribution<uint32_t> dist(0, 200) ;
    for (uint64_t i = 0; i < count; ++i) {
        cout << dec << dist(engine) << endl;
    }
    return 0;
}

int d_normal(uint64_t count, uint32_t seed)
{
    using MersenneTwister::NormalFromDouble;
    using MersenneTwister::DSFMT19937;

    NormalFromDouble<DSFMT19937> dist(0.0, 1.0, seed) ;
    cout.setf(std::ios::fixed);
    for (uint64_t i = 0; i < count; ++i) {
        cout << dec << setprecision(20) << dist() << endl;
    }
    return 0;
}

int d_uniform(uint64_t count, uint32_t seed)
{
    using MersenneTwister::UniformIntFromDouble;
    using MersenneTwister::DSFMT19937;

    UniformIntFromDouble<uint32_t, DSFMT19937> dist(0, 200, seed) ;
    cout.setf(std::ios::fixed);
    for (uint64_t i = 0; i < count; ++i) {
        cout << dec << dist() << endl;
    }
    return 0;
}

int main(int argc, char * argv[])
{
    uint64_t count = 1000;
    uint32_t seed = 1234;
    if (argc <= 2) {
        cout << argv[0] << " [-n|-u] [-m|-M|-S|-d] [seed]" << endl;
        cout << "-n     normal distribution" << endl;
        cout << "-u     uniform distribution" << endl;
        cout << "-m     std::mersennetwister19937" << endl;
        cout << "-M     MersenneTwister19937" << endl;
        cout << "-S     SFMT19937" << endl;
        cout << "-d     dSFMT19937" << endl;
        cout << "seed   seed" << endl;
        return -1;
    }
    bool is_normal = false;
    bool is_uniform = true;
    if (argv[1][1] == 'n') {
        is_normal = true;
        is_uniform = false;
    } else if (argv[1][1] == 'u') {
        is_normal = false;
        is_uniform = true;
    }
    if (argc >= 4) {
        errno = 0;
        seed = strtoul(argv[3], NULL, 10);
        if (errno) {
            cout << "seed must be a number" << endl;
            return -1;
        }
    }
    if (is_uniform) {
        cout << "#uniform distribution:";
        switch (argv[2][1]) {
        case 'm':
            cout << "std::mt19937" << endl;
            uniform<std::mt19937>(count, seed);
            break;
        case 'M':
            cout << "MersenneTwister::MT19937" << endl;
            uniform<MersenneTwister::MT19937>(count, seed);
            break;
        case 'S':
            cout << "MersenneTwister::SFMT19937" << endl;
            uniform<MersenneTwister::SFMT19937>(count, seed);
            break;
        case 'd':
            cout << "MersenneTwister::dSFMT19937" << endl;
            d_uniform(count, seed);
            break;
        default:
            break;
        }
    } else if (is_normal) {
        cout << "#normal distribution:";
        switch (argv[2][1]) {
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
            d_normal(count, seed);
            break;
        default:
            break;
        }
    }
    return 0;
}
