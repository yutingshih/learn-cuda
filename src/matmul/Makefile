CXX = nvcc

BIN = matmul.cpu matmul.gpu

all: $(BIN)

%.cpu: %.c
	$(CC) -o $@ $^

%.gpu: %.cu
	$(CXX) -o $@ $^

clean:
	$(RM) $(BIN)
