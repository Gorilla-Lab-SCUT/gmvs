cmake -S . -B build \
-DNUMPY_INCLUDE_DIR=$(python -c "from numpy import get_include; print(get_include())")
cmake --build build -j32
