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

set(LITERTLM_TFLITE_TARGET_MAP
    "tensorflow-lite=${LITERTLM_TFLITE_BUILD_DIR}/libtensorflow-lite.a"
    "xnnpack-delegate=${LITERTLM_TFLITE_BUILD_DIR}/libxnnpack-delegate.a"
    "ruy::allocator=${LITERTLM_TFLITE_LIB_DIR}/libruy_allocator.a"
    "ruy::apply_multiplier=${LITERTLM_TFLITE_LIB_DIR}/libruy_apply_multiplier.a"
    "ruy::blocking_counter=${LITERTLM_TFLITE_LIB_DIR}/libruy_blocking_counter.a"
    "ruy::block_map=${LITERTLM_TFLITE_LIB_DIR}/libruy_block_map.a"
    "ruy::context=${LITERTLM_TFLITE_LIB_DIR}/libruy_context.a"
    "ruy::context_get_ctx=${LITERTLM_TFLITE_LIB_DIR}/libruy_context_get_ctx.a"
    "ruy::cpuinfo=${LITERTLM_TFLITE_LIB_DIR}/libruy_cpuinfo.a"
    "ruy::ctx=${LITERTLM_TFLITE_LIB_DIR}/libruy_ctx.a"
    "ruy::denormal=${LITERTLM_TFLITE_LIB_DIR}/libruy_denormal.a"
    "ruy::frontend=${LITERTLM_TFLITE_LIB_DIR}/libruy_frontend.a"
    "ruy::have_built_path_for_avx2_fma=${LITERTLM_TFLITE_LIB_DIR}/libruy_have_built_path_for_avx2_fma.a"
    "ruy::have_built_path_for_avx512=${LITERTLM_TFLITE_LIB_DIR}/libruy_have_built_path_for_avx512.a"
    "ruy::have_built_path_for_avx=${LITERTLM_TFLITE_LIB_DIR}/libruy_have_built_path_for_avx.a"
    "ruy::kernel_arm=${LITERTLM_TFLITE_LIB_DIR}/libruy_kernel_arm.a"
    "ruy::kernel_avx2_fma=${LITERTLM_TFLITE_LIB_DIR}/libruy_kernel_avx2_fma.a"
    "ruy::kernel_avx512=${LITERTLM_TFLITE_LIB_DIR}/libruy_kernel_avx512.a"
    "ruy::kernel_avx=${LITERTLM_TFLITE_LIB_DIR}/libruy_kernel_avx.a"
    "ruy::pack_arm=${LITERTLM_TFLITE_LIB_DIR}/libruy_pack_arm.a"
    "ruy::pack_avx2_fma=${LITERTLM_TFLITE_LIB_DIR}/libruy_pack_avx2_fma.a"
    "ruy::pack_avx512=${LITERTLM_TFLITE_LIB_DIR}/libruy_pack_avx512.a"
    "ruy::pack_avx=${LITERTLM_TFLITE_LIB_DIR}/libruy_pack_avx.a"
    "ruy::prepacked_cache=${LITERTLM_TFLITE_LIB_DIR}/libruy_prepacked_cache.a"
    "ruy::prepare_packed_matrices=${LITERTLM_TFLITE_LIB_DIR}/libruy_prepare_packed_matrices.a"
    "ruy::profiler_instrumentation=${LITERTLM_TFLITE_LIB_DIR}/libruy_profiler_instrumentation.a"
    "ruy::profiler_profiler=${LITERTLM_TFLITE_LIB_DIR}/libruy_profiler_profiler.a"
    "ruy::system_aligned_alloc=${LITERTLM_TFLITE_LIB_DIR}/libruy_system_aligned_alloc.a"
    "ruy::thread_pool=${LITERTLM_TFLITE_LIB_DIR}/libruy_thread_pool.a"
    "ruy::trmul=${LITERTLM_TFLITE_LIB_DIR}/libruy_trmul.a"
    "ruy::tune=${LITERTLM_TFLITE_LIB_DIR}/libruy_tune.a"
    "ruy::wait=${LITERTLM_TFLITE_LIB_DIR}/libruy_wait.a"
    "xnnpack=${LITERTLM_TFLITE_LIB_DIR}/libXNNPACK.a"
    "xnnpack-microkernels-prod=${LITERTLM_TFLITE_LIB_DIR}/libxnnpack-microkernels-prod.a"
    "cpuinfo=${LITERTLM_TFLITE_LIB_DIR}/libcpuinfo.a"
    "pthreadpool=${LITERTLM_TFLITE_LIB_DIR}/libpthreadpool.a"
    "gemmlowp=${LITERTLM_TFLITE_LIB_DIR}/libeight_bit_int_gemm.a"
    "fft2d_fftsg2d=${LITERTLM_TFLITE_LIB_DIR}/libfft2d_fftsg2d.a"
    "fft2d_fftsg=${LITERTLM_TFLITE_LIB_DIR}/libfft2d_fftsg.a"
    "farmhash=${LITERTLM_TFLITE_BUILD_DIR}/_deps/farmhash-build/libfarmhash.a"
    "fft2d_shrtdct=${LITERTLM_TFLITE_BUILD_DIR}/_deps/fft2d-build/libfft2d_shrtdct.a"
    "fft2d_fft4f2d=${LITERTLM_TFLITE_BUILD_DIR}/_deps/fft2d-build/libfft2d_fft4f2d.a"
    "fft2d_fftsg3d=${LITERTLM_TFLITE_BUILD_DIR}/_deps/fft2d-build/libfft2d_fftsg3d.a"
    "fft2d_alloc=${LITERTLM_TFLITE_BUILD_DIR}/_deps/fft2d-build/libfft2d_alloc.a"
    "xnnpack-microkernels-all=${LITERTLM_TFLITE_BUILD_DIR}/_deps/xnnpack-build/libxnnpack-microkernels-all.a"
    "cpuinfo_internals=${LITERTLM_TFLITE_BUILD_DIR}/_deps/cpuinfo-build/libcpuinfo_internals.a"
    "tflite_model_runtime_info_proto=${LITERTLM_TFLITE_BUILD_DIR}/profiling/proto/libmodel_runtime_info_proto.a"
    "tflite_profiling_info_proto=${LITERTLM_TFLITE_BUILD_DIR}/profiling/proto/libprofiling_info_proto.a"
    "tflite_benchmark_result_proto=${LITERTLM_TFLITE_BUILD_DIR}/tools/benchmark/proto/libbenchmark_result_proto.a"
    "tflite_feature_proto=${LITERTLM_TFLITE_BUILD_DIR}/example_proto_generated/libfeature_proto.a"
    "tflite_example_proto=${LITERTLM_TFLITE_BUILD_DIR}/example_proto_generated/libexample_proto.a"
    "tflite_profiling=${LITERTLM_TFLITE_LIB_DIR}/libtflite_profiling.a"
)

if(LITERTLM_TOOLCHAIN_ARGS)
    message(STATUS "[LiteRTLM] Cross-compilation detected: Appending ARM64 targets.")
    list(APPEND TFLITE_TARGET_MAP "kleidiai=${LITERTLM_TFLITE_LIB_DIR}/libkleidiai.a")
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
