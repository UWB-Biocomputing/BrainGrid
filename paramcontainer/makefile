CC=g++
LD=g++
CCFLAGS=
LDFLAGS=

all:bin
bin:sample

sample:sample.o ParamContainer.o
	$(LD) $(LDFLAGS) sample.o ParamContainer.o -o sample

sample.o:sample.cpp ParamContainer.h
	$(CC) -c $(CCFLAGS) sample.cpp

ParamContainer.o:ParamContainer.cpp ParamContainer.h
	$(CC) -c $(CCFLAGS) ParamContainer.cpp

clean:
	rm -f sample.o ParamContainer.o sample
