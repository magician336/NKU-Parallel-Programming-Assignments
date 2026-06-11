CXX = g++
CXXFLAGS = -O3 -std=c++11 -fopenmp -march=armv8-a

TARGET = pq_openmp
SRC = main.cc

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC) search_openmp.h pq_quantizer.h simd_utils.h
	$(CXX) $(CXXFLAGS) -o $@ $(SRC)

clean:
	rm -f $(TARGET)
