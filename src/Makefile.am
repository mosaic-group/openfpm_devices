
LINKLIBS = $(PTHREAD_LIBS) $(OPT_LIBS) $(BOOST_LDFLAGS) $(BOOST_PROGRAM_OPTIONS_LIB) $(CUDA_LIBS) $(BOOST_THREAD_LIB)

if BUILDCUDA
 CUDA_SOURCES=memory/CudaMemory.cu
else
 CUDA_SOURCES=
endif

bin_PROGRAMS = mem
mem_SOURCES = main.cpp memory/HeapMemory.cpp $(CUDA_SOURCES)
mem_CXXFLAGS = $(INCLUDES_PATH) -I/usr/local/include
mem_CFLAGS = 
mem_LDADD = $(LINKLIBS) -L/usr/lib64/nvidia-bumblebee/

.cu.o :
	$(NVCC) $(NVCCFLAGS) -I. $(INCLUDES_PATH) -o $@ -c $<
