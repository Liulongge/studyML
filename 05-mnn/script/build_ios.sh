cd build

cmake \
-DMNN_OPENCL=ON \
-DMNN_OPENCL_PROFILE=ON \
-DMNN_BUILD_BENCHMARK=ON \
-DMNN_METAL=ON \
-DMNN_VULKAN=ON \
../../../../tmp/MNN

make -j8
