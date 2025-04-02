CXX = nvcc

BIN = vecAdd.out

all: $(BIN)
	./$<

%.out: %.cu
	$(CXX) -o $@ $^

clean:
	$(RM) $(BIN)
