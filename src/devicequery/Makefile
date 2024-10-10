# Makefile written by Wim R.M. Cardoen on 10/10/2024
EXE=devinfo
CC=nvcc
CFLAGS= -O2
LDFLAGS=  --cudart=static

$(EXE): devinfo.o
	$(CC) $(LDFLAGS) -o $(EXE) devinfo.o
 
devinfo.o:devinfo.cu
	$(CC) -c $(CUFLAGS) devinfo.cu

clean:
	rm -rf $(EXE) *.o
