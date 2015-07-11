
LINKLIBS = $(PTHREAD_LIBS) $(OPT_LIBS) $(BOOST_LDFLAGS) $(BOOST_PROGRAM_OPTIONS_LIB) $(CUDA_LIBS) $(BOOST_THREAD_LIB)

bin_PROGRAMS = mem
mem_SOURCES = main.cpp memory/HeapMemory.cpp memory/CudaMemory.cu
mem_CXXFLAGS = $(INCLUDES_PATH)
mem_CFLAGS = 
mem_LDADD = $(LINKLIBS) -L/usr/lib64/nvidia-bumblebee/

.cu.o :
	$(NVCC) $(NVCCFLAGS) -I. $(INCLUDES_PATH) -o $@ -c $<
