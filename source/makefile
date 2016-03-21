CC=g++
CFLAGS=-Wall -ggdb -std=c++11
HEADERDIR=/usr/include/gecode
LIBDIR=/usr/lib
LINKFLAGS=-lgecodeflatzinc -lgecodedriver -lgecodegist -lgecodesearch\
		  -lgecodeminimodel -lgecodeset -lgecodefloat -lgecodeint -lgecodekernel\
		  -lgecodesupport -lpthread 

all: linalg.o neuralnet.o main.cpp
	$(CC) $(CFLAGS) -L $(LIBDIR) $(LINKFLAGS) -o NN \
        linalg.o neuralnet.o main.cpp

neuralnet.o: neuralnet.cpp neuralnet.h
	$(CC) $(CFLAGS) -c neuralnet.cpp

linalg.o: linalg.cpp linalg.h
	$(CC) $(CFLAGS) -c linalg.cpp

clean:
	rm *.o 
