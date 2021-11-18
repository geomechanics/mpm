make clean
rm CMakeCache.txt
export CXX_COMPILER=mpicxx
cmake  -DCMAKE_BUILD_TYPE=Release -DPARTIO_ROOT=~/application/partio/ -DKAHIP_ROOT=~/application/KaHIP/ -DCMAKE_CXX_FLAGS="-O3 -Wunused-variable"  ..
make -j16
