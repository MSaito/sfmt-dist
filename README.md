# sfmt-dist

sfmt distribution

STANDARD INTERFACE
==================
std::uniform_int_distribution and std::normal_distribution adaptable
SFMT19937

sample code
std::uniform_int_distribution<uint32_t> dist(0, 200);
MersenneTwiseter::SFMT19937 sfmt(1234);
cout << dec << dist(sfmt) << endl;

ORIGINAL INTERFACE FOR dSFMT
============================
original distribution MersenneTwister::UniformIntFromDouble and
MersenneTwister::NormalFromDouble for DSFMT19937

sample code
using MersenneTwister::NormalFromDouble;
using MersenneTwister::DSFMT19937;
NormalFromDouble<DSFMT19937> dist(0.0, 1.0, 1234);
cout.setf(std::ios::fixed);
for (uint64_t i = 0; i < 100; ++i) {
    cout << dec << setprecision(20) << dist() << endl;
}

using MersenneTwister::UniformIntFromDouble;
UniformIntFromDouble<uint32_t, DSFMT19937> dist(0, 200, 1234) ;
cout.setf(std::ios::fixed);
for (uint64_t i = 0; i < count; ++i) {
    cout << dec << dist() << endl;
}
