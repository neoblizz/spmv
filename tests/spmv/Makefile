include ../Makefile.inc

APP = spmv

test: bin/$(APP)

bin/$(APP) : test_$(APP).cu *.h $(DEPS)
	mkdir -p bin
	$(NVCC) -ccbin=${CXX} ${NVCCFLAGS} --compiler-options "${CXXFLAGS}" -o bin/$(APP) test_$(APP).cu $(SOURCE) $(ARCH) $(INC)

debug : test_$(APP).cu *.h $(DEPS)
	mkdir -p bin
	$(NVCC) -ccbin=${CXX} ${NVCCFLAGS} --compiler-options "${CXXFLAGS}" -o bin/$(APP) test_$(APP).cu $(SOURCE) $(ARCH) $(INC) -g -G

.DEFAULT_GOAL := test