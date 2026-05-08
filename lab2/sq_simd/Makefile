# SQ-SIMD Makefile

CXX = g++
CXXFLAGS = -O2 -std=c++11 -fopenmp -lpthread

UNAME_S := $(shell uname -s 2>/dev/null)
UNAME_M := $(shell uname -m 2>/dev/null)

ifneq ($(UNAME_S),)
    ifeq ($(UNAME_M),aarch64)
        CXXFLAGS += -march=armv8-a
    else
        CXXFLAGS += -march=native
    endif
else
    CXXFLAGS += -msse4.2
endif

TARGET = sq_simd

all: $(TARGET)

$(TARGET): main.cc search.h sq_quantizer.h simd_utils.h
	$(CXX) $(CXXFLAGS) main.cc -o $(TARGET)

clean:
	$(RM) $(TARGET) $(TARGET).exe

.PHONY: all clean
