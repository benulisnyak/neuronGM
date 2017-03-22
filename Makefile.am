AUTOMAKE_OPTIONS = subdir-objects
ACLOCAL_AMFLAGS = ${ACLOCAL_FLAGS}
#SIM_PATH = "../SimKernel"
#AM_CPPFLAGS = "-I$(SIM_PATH)"
LIBS = $(DEPS_LIBS)
AM_CPPFLAGS = $(DEPS_CFLAGS)
AM_LDFLAGS = -Wl,--no-as-needed
DEBUG = false
bin_PROGRAMS = neuronGM
neuronGM_SOURCES = src/main.cc src/kernel.cc src/kernel.h

#neuronGM_LDADD = -lgsl -lgslcblas -lgcov -lsim
neuronGM_LDADD = -lsim -lrt -lgsl -lgslcblas
if DEBUG
neuronGM_LDFLAGS = -fprofile-arcs -pg -ggdb
neuronGM_CPPFLAGS = -I /usr/local/include/eigen3 -fprofile-arcs -ftest-coverage -pg -ggdb
else
neuronGM_CPPFLAGS = -I /usr/local/include/eigen3 
endif
# dist_noinst_SCRIPTS = autogen.sh

