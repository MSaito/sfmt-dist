#include "sfmt-dist.h"
#include <sfmt-dist/cpu_feature.h>
int main() {
    using namespace MersenneTwister;
    cpu_feature_t cf = cpu_feature();
    print_cpu_feature(cf);
    return 0;
}
