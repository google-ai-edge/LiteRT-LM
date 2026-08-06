# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Updated on 2026-03-25.
# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


set(TFLITE_TARGET_MAP
    "tensorflow-lite=${TFLITE_BUILD_DIR}/libtensorflow-lite.a"
    "xnnpack-delegate=${TFLITE_BUILD_DIR}/libxnnpack-delegate.a"
    "ruy::allocator=${TFLITE_LIB_DIR}/libruy_allocator.a"
    "ruy::apply_multiplier=${TFLITE_LIB_DIR}/libruy_apply_multiplier.a"
    "ruy::blocking_counter=${TFLITE_LIB_DIR}/libruy_blocking_counter.a"
    "ruy::block_map=${TFLITE_LIB_DIR}/libruy_block_map.a"
    "ruy::context=${TFLITE_LIB_DIR}/libruy_context.a"
    "ruy::context_get_ctx=${TFLITE_LIB_DIR}/libruy_context_get_ctx.a"
    "ruy::cpuinfo=${TFLITE_LIB_DIR}/libruy_cpuinfo.a"
    "ruy::ctx=${TFLITE_LIB_DIR}/libruy_ctx.a"
    "ruy::denormal=${TFLITE_LIB_DIR}/libruy_denormal.a"
    "ruy::frontend=${TFLITE_LIB_DIR}/libruy_frontend.a"
    "ruy::have_built_path_for_avx2_fma=${TFLITE_LIB_DIR}/libruy_have_built_path_for_avx2_fma.a"
    "ruy::have_built_path_for_avx512=${TFLITE_LIB_DIR}/libruy_have_built_path_for_avx512.a"
    "ruy::have_built_path_for_avx=${TFLITE_LIB_DIR}/libruy_have_built_path_for_avx.a"
    "ruy::kernel_arm=${TFLITE_LIB_DIR}/libruy_kernel_arm.a"
    "ruy::kernel_avx2_fma=${TFLITE_LIB_DIR}/libruy_kernel_avx2_fma.a"
    "ruy::kernel_avx512=${TFLITE_LIB_DIR}/libruy_kernel_avx512.a"
    "ruy::kernel_avx=${TFLITE_LIB_DIR}/libruy_kernel_avx.a"
    "ruy::pack_arm=${TFLITE_LIB_DIR}/libruy_pack_arm.a"
    "ruy::pack_avx2_fma=${TFLITE_LIB_DIR}/libruy_pack_avx2_fma.a"
    "ruy::pack_avx512=${TFLITE_LIB_DIR}/libruy_pack_avx512.a"
    "ruy::pack_avx=${TFLITE_LIB_DIR}/libruy_pack_avx.a"
    "ruy::prepacked_cache=${TFLITE_LIB_DIR}/libruy_prepacked_cache.a"
    "ruy::prepare_packed_matrices=${TFLITE_LIB_DIR}/libruy_prepare_packed_matrices.a"
    "ruy::profiler_instrumentation=${TFLITE_LIB_DIR}/libruy_profiler_instrumentation.a"
    "ruy::profiler_profiler=${TFLITE_LIB_DIR}/libruy_profiler_profiler.a"
    "ruy::system_aligned_alloc=${TFLITE_LIB_DIR}/libruy_system_aligned_alloc.a"
    "ruy::thread_pool=${TFLITE_LIB_DIR}/libruy_thread_pool.a"
    "ruy::trmul=${TFLITE_LIB_DIR}/libruy_trmul.a"
    "ruy::tune=${TFLITE_LIB_DIR}/libruy_tune.a"
    "ruy::wait=${TFLITE_LIB_DIR}/libruy_wait.a"
    "xnnpack=${TFLITE_LIB_DIR}/libXNNPACK.a"
    "xnnpack-microkernels-prod=${TFLITE_LIB_DIR}/libxnnpack-microkernels-prod.a"
    "cpuinfo=${TFLITE_LIB_DIR}/libcpuinfo.a"
    "pthreadpool=${TFLITE_LIB_DIR}/libpthreadpool.a"
    "gemmlowp=${TFLITE_LIB_DIR}/libeight_bit_int_gemm.a"
    "fft2d_fftsg2d=${TFLITE_LIB_DIR}/libfft2d_fftsg2d.a"
    "fft2d_fftsg=${TFLITE_LIB_DIR}/libfft2d_fftsg.a"
    "farmhash=${TFLITE_BUILD_DIR}/_deps/farmhash-build/libfarmhash.a"
    "fft2d_shrtdct=${TFLITE_BUILD_DIR}/_deps/fft2d-build/libfft2d_shrtdct.a"
    "fft2d_fft4f2d=${TFLITE_BUILD_DIR}/_deps/fft2d-build/libfft2d_fft4f2d.a"
    "fft2d_fftsg3d=${TFLITE_BUILD_DIR}/_deps/fft2d-build/libfft2d_fftsg3d.a"
    "fft2d_alloc=${TFLITE_BUILD_DIR}/_deps/fft2d-build/libfft2d_alloc.a"
    "xnnpack-microkernels-all=${TFLITE_BUILD_DIR}/_deps/xnnpack-build/libxnnpack-microkernels-all.a"
    "cpuinfo_internals=${TFLITE_BUILD_DIR}/_deps/cpuinfo-build/libcpuinfo_internals.a"
    "tflite_model_runtime_info_proto=${TFLITE_BUILD_DIR}/profiling/proto/libmodel_runtime_info_proto.a"
    "tflite_profiling_info_proto=${TFLITE_BUILD_DIR}/profiling/proto/libprofiling_info_proto.a"
    "tflite_benchmark_result_proto=${TFLITE_BUILD_DIR}/tools/benchmark/proto/libbenchmark_result_proto.a"
    "tflite_feature_proto=${TFLITE_BUILD_DIR}/example_proto_generated/libfeature_proto.a"
    "tflite_example_proto=${TFLITE_BUILD_DIR}/example_proto_generated/libexample_proto.a"
    "tflite_profiling=${TFLITE_LIB_DIR}/libtflite_profiling.a"
)

if(LITERTLM_ENABLE_KLEIDIAI)
    message(STATUS "[LiteRTLM] Cross-compilation detected: Appending ARM64 targets.")
    list(APPEND TFLITE_TARGET_MAP "kleidiai=${TFLITE_LIB_DIR}/libkleidiai.a")
endif()


# Exhaustive targets for Shim Redirection
set(_tflite_exhaustive_targets
    "tensorflow-lite"
    "xnnpack"
    "ruy"
    "cpuinfo"
    "pthreadpool"
    "farmhash"
    "fft2d"
)
