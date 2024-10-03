# Makefile originally written by Wim R. Cardoen on 10/31/2011
#          modified on 09/30/2024
EXE=mul2
CC=nvcc
CFLAGS= -O2 -I./include
CUFLAGS= -O2 -arch=compute_70 -code=sm_70 -I./include
CPPFLAGS= -Dcheck
LDFLAGS= -arch=compute_70 -code=sm_70

mul: aux.o main.o mul.o 
	$(CC) $(LDFLAGS) -o $(EXE) aux.o main.o mul.o	

aux.o:aux.cc
	$(CC) -c $(CFLAGS) aux.cc		

main.o:main.cu
	$(CC) -c $(CUFLAGS) $(CPPFLAGS) main.cu	

mul.o: mul.cu
	$(CC) -c $(CUFLAGS) mul.cu	

clean:
	rm -rf $(EXE) *.o
