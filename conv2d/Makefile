BIN = conv.cpu conv.gpu

NVCC = nvcc
CPPFLAGS = -DNDEBUG

all: $(BIN)

%.cpu: %.cc main.cc
	$(CXX) $(CPPFLAGS) -o $@ $^

%.gpu: %.cu main.cc
	$(NVCC) $(CPPFLAGS) -o $@ $^

clean:
	$(RM) $(BIN)
