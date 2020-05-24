CUDA_PATH=/usr/local/cuda
PGCXX=pgc++
TARGET=merging

PGCXX_FLAGS= -acc -ta=tesla:managed -Minfo=accel

all: ${TARGET}

merging: image_merging.cpp scrImagePgmPpmPackage.cpp
	$(EXEC) $(PGCXX) -o $@ $+ 

clean:
	rm -f ${TARGET} *.o
