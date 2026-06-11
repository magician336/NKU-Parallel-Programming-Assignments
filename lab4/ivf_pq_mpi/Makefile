CXX = mpic++
CXXFLAGS = -O2 -std=c++11 -fopenmp -march=armv8-a
LDFLAGS = -lpthread

TARGET = main
SRC = main.cc

HEADERS = ivf_index.h adc_searcher.h sdc_searcher.h simd_searcher.h \
          searcher.h fast_scan_kernel.h pq_distance.h simd_l2.h \
          aligned_alloc.h kmeans.h profiler.h thread_pool.h search_buffers.h

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)
