.PHONY: clean

libdpuv1utils.so: libpostproc.o
	g++ -shared $^ -o $@
	echo "Done"

libpostproc.o: postproc/*.cpp
	g++ -std=c++11 -O3 -c -fPIC -static-libgcc -static-libstdc++ $^ -o $@

clean:
	rm *.o *.so
