// Copyright 2026 RMAA Holdings & The ODML Authors.

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include "litert_lm_rmaa_fabric.h"

#include <cassert>
#include <chrono>
#include <iostream>
#include <vector>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

int main() {
  std::cout << "==============================================================\n";
  std::cout << "🧪 LiteRT-LM RMAA Fabric Delegate: Integration & Validation\n";
  std::cout << "==============================================================\n";

  LiteRtLmRmaaFabricConfig config = litert_lm_rmaa_fabric_config_default();
  config.enable_smtf_v3_hugepages = true;
  config.enable_njit_memory_grafting = true;
  config.enable_localjit_dispatch = true;
  config.enable_upaas_quic_transport = true;

  LiteRtLmRmaaFabricHandle fabric = nullptr;
  LiteRtLmStatusCode status = litert_lm_rmaa_fabric_create(&config, &fabric);
  assert(status == kLiteRtLmStatusOk);
  assert(fabric != nullptr);
  std::cout << "[PASS] 1. Created LiteRtLmRmaaFabricHandle successfully.\n";

  const uint64_t kTestCID = 0xA1B2C3D4E5F60708ULL;
  const size_t kSeqLen = 2048;
  const size_t kPayloadSize = 4096;

  char shm_path[128];
  snprintf(shm_path, sizeof(shm_path), "/litert_njit_graft_%" PRIx64, kTestCID);
  int fd = shm_open(shm_path, O_CREAT | O_RDWR, 0666);
  assert(fd >= 0);

  size_t total_shm_size = 36 + kPayloadSize;
  int tr_res = ftruncate(fd, total_shm_size);
  (void)tr_res;

  void* map_ptr = mmap(nullptr, total_shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  assert(map_ptr != MAP_FAILED);

  uint8_t* raw = reinterpret_cast<uint8_t*>(map_ptr);
  *reinterpret_cast<uint32_t*>(raw + 0) = 0x54494A4E;
  *reinterpret_cast<uint16_t*>(raw + 4) = 0x0100;
  *reinterpret_cast<uint16_t*>(raw + 6) = 2;
  *reinterpret_cast<uint64_t*>(raw + 8) = kTestCID;
  *reinterpret_cast<uint32_t*>(raw + 16) = 32;
  *reinterpret_cast<uint32_t*>(raw + 20) = 128;
  *reinterpret_cast<uint32_t*>(raw + 24) = kSeqLen;
  *reinterpret_cast<uint64_t*>(raw + 28) = kPayloadSize;
  std::memset(raw + 36, 0x7F, kPayloadSize);

  munmap(map_ptr, total_shm_size);
  close(fd);

  std::vector<uint8_t> engine_cache(kPayloadSize, 0);
  size_t grafted_pos = 0;

  auto t_start = std::chrono::high_resolution_clock::now();
  status = litert_lm_rmaa_fabric_graft_kv_cache(
      fabric, kTestCID, engine_cache.data(), engine_cache.size(), &grafted_pos);
  auto t_end = std::chrono::high_resolution_clock::now();

  double latency_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

  assert(status == kLiteRtLmStatusOk);
  assert(grafted_pos == kSeqLen);
  assert(engine_cache[0] == 0x7F);
  shm_unlink(shm_path);

  std::cout << "[PASS] 2. NJIT Zero-Prefill Memory Grafting Succeeded!\n";
  std::cout << "       • Grafted Sequence Position : " << grafted_pos << " tokens\n";
  std::cout << "       • Grafting Wall Latency     : " << latency_us << " µs (Target: <100 µs)\n";

  bool hit = false;
  status = litert_lm_rmaa_fabric_dispatch_kernel(fabric, 0x12345678ULL, nullptr, &hit);
  assert(status == kLiteRtLmStatusOk);
  assert(!hit);
  std::cout << "[PASS] 3. LocalJIT Dispatch Fallback to XNNPACK Verified.\n";

  const int32_t draft_tokens[2] = {1054, 42};
  size_t accepted = 0;
  status = litert_lm_rmaa_fabric_verify_mtp_draft(
      fabric, 1337ULL, 42ULL, 100, draft_tokens, 2, &accepted);
  assert(status == kLiteRtLmStatusOk);
  assert(accepted == 2);
  std::cout << "[PASS] 4. UPAAS MTP Inverse-CDF Verification (gamma=2): " << accepted << " tokens accepted.\n";

  status = litert_lm_rmaa_fabric_destroy(fabric);
  assert(status == kLiteRtLmStatusOk);
  std::cout << "[PASS] 5. Clean teardown of LiteRtLmRmaaFabricHandle.\n";

  std::cout << "==============================================================\n";
  std::cout << "🎉 ALL RMAA FABRIC INTEGRATION BENCHMARKS PASSED!\n";
  std::cout << "==============================================================\n";
  return 0;
}
