CUDA_PATH = /usr/local/cuda
NVCC = $(CUDA_PATH)/bin/nvcc
NVCC_VERSION = $(strip $(shell $(NVCC) --version | grep release | sed 's/.*release //' | sed 's/..*//'))

CXX = g++

ARCH_SUFFIX = x86_64
ARCH = -m64

GEN_SM80 = -gencode=arch=compute_80,code=\"sm_80,compute_80\"
GEN_SM75 = -gencode=arch=compute_75,code=\"sm_75,compute_75\"
GEN_SM70 = -gencode=arch=compute_70,code=\"sm_70,compute_70\"
GEN_SM60 = -gencode=arch=compute_60,code=\"sm_60,compute_60\"
GEN_SM37 = -gencode=arch=compute_37,code=\"sm_37,compute_37\"
GEN_SM35 = -gencode=arch=compute_35,code=\"sm_35,compute_35\"
GEN_SM30 = -gencode=arch=compute_30,code=\"sm_30,compute_30\"

SM_TARGETS = $(GEN_SM70)

EXT_INC= ../../externals
MGPU_INC= $(EXT_INC)/moderngpu/src
CUB_INC= $(EXT_INC)/cub
THRUST_INC= $(EXT_INC)/thrust
MTX_INC=  $(EXT_INC)/mtx
CUDA_INC= $(CUDA_PATH)/include

INC=-I.. -I../../src -I$(MGPU_INC) -I$(CUB_INC) -I$(THRUST_INC) -I$(CUDA_INC) -I$(MTX_INC) -lcudart -lcusparse

NVCCFLAGS += -std=c++14
NVCCFLAGS += $(SM_TARGETS)
NVCCFLAGS += --expt-extended-lambda --expt-relaxed-constexpr --generate-line-info --ptxas-options -v --relocatable-device-code true --use_fast_math

CXXFLAGS += -std=c++14
CXXFLAGS += -Wall
CXXFLAGS += -Wno-xxx -Wno-unused-local-typedefs -Wno-mismatched-tags -Wno-strict-aliasing -Wno-unused-function -Wno-format-security

ifeq ($(DEBUG), 1)
    CXXFLAGS += -O0 -g
    NVCCFLAGS += -O0 -G
else
    CXXFLAGS += -O3
    NVCCFLAGS += -O3
endif

SOURCE = ../../externals/mtx/mmio.c

DEPS= ./Makefile \
	  ../Makefile.inc \
	  $(SOURCE) \
	  $(wildcard ../../src/*.hxx) \
	  $(wildcard ../../src/*/*.hxx) \
	  $(wildcard ../../src/*.cuh) \
	  $(wildcard ../../src/*/*.cuh)

clean :
	rm -rf bin
	rm -f *.i* *.cubin *.cu.c *.cudafe* *.fatbin.c *.ptx *.hash *.cu.cpp *.o
