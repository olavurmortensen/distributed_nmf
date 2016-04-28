
dnmf_exec:
	rm -f -r *.o
	mpicc -std=gnu99 -c main.c
	mpicc main.o -o dnmf_exec
	rm -f -r *.o

