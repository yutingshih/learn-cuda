CXX = nvcc

BIN = hello.out

all: $(BIN)
	./$<

%.out: %.cu
	$(CXX) -o $@ $^

clean:
	$(RM) $(BIN)
