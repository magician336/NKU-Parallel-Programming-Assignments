# Flat-SIMD Makefile
# Supports Linux/Windows(MinGW)

CXX = g++
CXXFLAGS = -O2 -std=c++11 -fopenmp -lpthread

# Detect platform for architecture-specific flags
UNAME_S := $(shell uname -s 2>/dev/null)
UNAME_M := $(shell uname -m 2>/dev/null)

ifneq ($(UNAME_S),)
    ifeq ($(UNAME_M),aarch64)
        CXXFLAGS += -march=armv8-a
    else
        CXXFLAGS += -march=native
    endif
else
    # Windows (x86 with SSE)
    CXXFLAGS += -msse4.2
endif

TARGET = flat_simd

all: $(TARGET)

$(TARGET): main.cc search.h simd_utils.h
	$(CXX) $(CXXFLAGS) main.cc -o $(TARGET)

clean:
	$(RM) $(TARGET) $(TARGET).exe

.PHONY: all clean
