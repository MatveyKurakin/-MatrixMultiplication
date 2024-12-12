# Compile
icx /O2 /Qmkl /QxHost /Qopenmp matrix_multiplication.cpp -o matrix_multiplication.exe
icx /O2 /Qmkl /arch:AVX2 /Qopenmp matrix_multiplication.cpp -o matrix_multiplication.exe