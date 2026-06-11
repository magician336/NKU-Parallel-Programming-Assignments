CXX = g++
CXXFLAGS = -O3 -std=c++11 -pthread -march=armv8-a

TARGET = ivf_pthread
SRC = main.cc

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC) ivf_index.h pq_quantizer.h simd_utils.h
	$(CXX) $(CXXFLAGS) -o $@ $(SRC)

clean:
	rm -f $(TARGET)
