
if BUILDCUDA
 CUDA_SOURCES=memory/CudaMemory.cu
else
 CUDA_SOURCES=
endif

LINKLIBS = $(PTHREAD_LIBS) $(OPT_LIBS) $(BOOST_LDFLAGS) $(BOOST_PROGRAM_OPTIONS_LIB) $(CUDA_LIBS) $(BOOST_THREAD_LIB)

LIBTOOL_DEPS = @LIBTOOL_DEPS@
libtool: $(LIBTOOL_DEPS)
	$(SHELL) ./config.status libtool

noinst_PROGRAMS = mem
mem_SOURCES = main.cpp memory/HeapMemory.cpp $(CUDA_SOURCES) Memleak_check.cpp
mem_CXXFLAGS = $(INCLUDES_PATH) $(BOOST_CPPFLAGS) -I/usr/local/include
mem_CFLAGS = 
mem_LDADD = $(LINKLIBS) -L/usr/lib64/nvidia-bumblebee/

lib_LIBRARIES = libofpmmemory.a libofpmmemory_se2.a
libofpmmemory_a_SOURCES = memory/HeapMemory.cpp $(CUDA_SOURCES) memory/PtrMemory.cpp Memleak_check.cpp
libofpmmemory_a_CXXFLAGS = $(INCLUDES_PATH) $(BOOST_CPPFLAGS) -I/usr/local/include
libofpmmemory_a_CFLAGS =

libofpmmemory_se2_a_SOURCES = memory/HeapMemory.cpp $(CUDA_SOURCES) memory/PtrMemory.cpp Memleak_check.cpp
libofpmmemory_se2_a_CXXFLAGS = $(INCLUDES_PATH) $(BOOST_CPPFLAGS) -I/usr/local/include -DSE_CLASS2
libofpmmemory_se2_a_CFLAGS = 

lib_LTLIBRARIES = libofpmmemory.la libofpmmemory_se2.la
libofpmmemory_la_SOURCES = memory/HeapMemory.cpp $(CUDA_SOURCES) memory/PtrMemory.cpp Memleak_check.cpp
libofpmmemory_la_CXXFLAGS = $(INCLUDES_PATH) $(BOOST_CPPFLAGS) -I/usr/local/include
libofpmmemory_la_CFLAGS =

libofpmmemory_se2_la_SOURCES = memory/HeapMemory.cpp $(CUDA_SOURCES) memory/PtrMemory.cpp Memleak_check.cpp
libofpmmemory_se2_la_CXXFLAGS = $(INCLUDES_PATH) $(BOOST_CPPFLAGS) -I/usr/local/include -DSE_CLASS2
libofpmmemory_se2_la_CFLAGS =


nobase_include_HEADERS = memory/ExtPreAlloc.hpp memory/BHeapMemory.hpp  memory/HeapMemory.hpp memory/memory.hpp memory/PtrMemory.hpp \
Memleak_check.hpp util/print_stack.hpp ptr_info.hpp \
util/se_util.hpp

.cu.o :
	$(NVCC) $(NVCCFLAGS) -std=c++11  -I. $(INCLUDES_PATH) -o $@ -c $<

test: mem
	source $(HOME)/openfpm_vars && cd .. && ./src/mem

