
LINKLIBS = $(PTHREAD_LIBS) $(OPT_LIBS) $(BOOST_LDFLAGS) $(BOOST_PROGRAM_OPTIONS_LIB) $(CUDA_LIBS) $(BOOST_THREAD_LIB)

bin_PROGRAMS = ppline
ppline_SOURCES = main.cpp
ppline_CXXFLAGS = $(CUDA_CFLAGS)
ppline_CFLAGS = $(CUDA_CFLAGS)
ppline_LDADD = $(LINKLIBS) -L/usr/lib64/nvidia-bumblebee/

.cu.o :
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<
