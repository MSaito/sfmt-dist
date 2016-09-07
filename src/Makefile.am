common_files = debug.h config.h sfmt-dist.h w256.h w128.h getopt.hpp \
sfmt_common.h dsfmt_avx.h dsfmt_avx_common.h dsfmt_avx_params.h

# LIB
lib_LTLIBRARIES = libsfmt_dist.la
libsfmt_dist_la_SOURCES = version.cpp cpu_feature.cpp aligned_alloc.cpp \
mt19937.cpp mt19937_64.cpp sfmt19937.cpp dSFMT19937.cpp dSFMTAVX607.cpp \
dSFMTAVX19937.cpp sfmt_common.cpp dsfmt_avx.cpp
#libsfmt_dist_la_LDFLAGS = -no-undefined -version-info 0:2:0
libsfmt_dist_la_CXXFLAGS = -I../include -O3 -Wall -Wextra \
-D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS

# make only when make check is invoked
check_PROGRAMS = test_mt19937 test_mt19937_64 test_sfmt19937 test_dsfmt19937 \
test_version test_dsfmtavx607 test_dsfmtavx19937
# execute when make check is invoked
TESTS = test_mt19937 test_mt19937_64 test_sfmt19937 test_dsfmt19937 \
test_version test_dsfmtavx607 test_dsfmtavx19937
test_mt19937_SOURCES = test_mt19937.cpp $(common_files)
test_mt19937_64_SOURCES = test_mt19937_64.cpp $(common_files)
test_sfmt19937_SOURCES = test_sfmt19937.cpp $(common_files)
test_dsfmt19937_SOURCES = test_dsfmt19937.cpp $(common_files)
test_dsfmtavx607_SOURCES = test_dsfmtavx607.cpp $(common_files)
test_dsfmtavx19937_SOURCES = test_dsfmtavx19937.cpp $(common_files)
test_version_SOURCES = test_version.cpp $(common_files)
test_mt19937_DEPENDENCIES = libsfmt_dist.la
test_mt19937_64_DEPENDENCIES = libsfmt_dist.la
test_sfmt19937_DEPENDENCIES = libsfmt_dist.la
test_dsfmt19937_DEPENDENCIES = libsfmt_dist.la
test_dsfmtavx607_DEPENDENCIES = libsfmt_dist.la
test_dsfmtavx19937_DEPENDENCIES = libsfmt_dist.la
test_version_DEPENDENCIES = libsfmt_dist.la
test_mt19937_LDADD = -lsfmt_dist
test_mt19937_64_LDADD = -lsfmt_dist
test_sfmt19937_LDADD = -lsfmt_dist
test_dsfmt19937_LDADD = -lsfmt_dist
test_dsfmtavx607_LDADD = -lsfmt_dist
test_dsfmtavx19937_LDADD = -lsfmt_dist
test_version_LDADD = -lsfmt_dist

#mt1_LDFLAGS = -L../../lib
#mt2_LDFLAGS = -L../../lib

#
noinst_PROGRAMS = cpu_feature output speed
cpu_feature_SOURCES = cpu_feature_main.cpp $(common_files)
output_SOURCES = output.cpp $(common_files)
speed_SOURCES = speed.cpp $(common_files)
cpu_feature_DEPENDENCIES = libsfmt_dist.la
output_DEPENDENCIES = libsfmt_dist.la
speed_DEPENDENCIES = libsfmt_dist.la
cpu_feature_LDADD = -lsfmt_dist
output_LDADD = -lsfmt_dist
speed_LDADD = -lsfmt_dist

AM_CXXFLAGS = -I../include -O3 -Wall -Wextra \
-D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS
