
LINKLIBS = $(PTHREAD_LIBS) $(OPT_LIBS) $(BOOST_LDFLAGS) $(BOOST_PROGRAM_OPTIONS_LIB) $(CUDA_LIBS) $(BOOST_THREAD_LIB)

if BUILDCUDA
 CUDA_SOURCES=memory/CudaMemory.cu
else
 CUDA_SOURCES=
endif

noinst_PROGRAMS = mem
mem_SOURCES = main.cpp memory/HeapMemory.cpp $(CUDA_SOURCES)
mem_CXXFLAGS = $(INCLUDES_PATH) -I/usr/local/include
mem_CFLAGS = 
mem_LDADD = $(LINKLIBS) -L/usr/lib64/nvidia-bumblebee/

lib_LIBRARIES = libofpmmemory.a
libofpmmemory_a_SOURCES = memory/HeapMemory.cpp $(CUDA_SOURCES) memory/PtrMemory.cpp
libofpmmemory_a_CXXFLAGS = $(INCLUDES_PATH) -I/usr/local/include
libofpmmemory_a_CFLAGS =

nobase_include_HEADERS = memory/ExtPreAlloc.hpp memory/HeapMemory.hpp memory/memory.hpp memory/PreAllocHeapMemory.hpp memory/PtrMemory.hpp

.cu.o :
	$(NVCC) $(NVCCFLAGS) -I. $(INCLUDES_PATH) -o $@ -c $<
