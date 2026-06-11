CXX = g++
CXXFLAGS = -O3 -std=c++11 -pthread -march=armv8-a -I../hnswlib

TARGET = hnsw_pthread
SRC = main.cc

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC)

clean:
	rm -f $(TARGET)
