CXX = mpic++
CXXFLAGS = -O2 -std=c++11 -fopenmp -march=armv8-a -I./hnswlib
LDFLAGS = -lpthread

TARGET = main
SRC = main.cc

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)
