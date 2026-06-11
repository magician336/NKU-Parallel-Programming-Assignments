CXX = mpic++
CXXFLAGS = -O2 -std=c++11 -fopenmp -march=armv8-a -I./hnswlib
LDFLAGS = -lpthread

TARGET = main
SRC = main.cc

HEADERS = hnsw_hnsw_index.h hnsw_hnsw_searcher.h kmeans.h simd_l2.h \
          aligned_alloc.h profiler.h thread_pool.h

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)
