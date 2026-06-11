CXX = g++
CXXFLAGS = -O3 -std=c++11 -pthread -march=armv8-a

TARGET = flat_pthread
SRC = main.cc

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC) search_pthread.h simd_utils.h
	$(CXX) $(CXXFLAGS) -o $@ $(SRC)

clean:
	rm -f $(TARGET)
