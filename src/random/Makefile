CXX = nvcc

BIN = random.out

all: $(BIN)
	./$<

%.out: %.cu
	$(CXX) -o $@ $^

clean:
	$(RM) $(BIN)
