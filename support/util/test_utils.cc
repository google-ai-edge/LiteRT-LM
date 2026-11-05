// Copyright 2026 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "support/util/test_utils.h"

#include <string>

#if defined(__linux__)
#include <cstdint>
#include <fstream>
#include <ios>
#include <sstream>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <mach/vm_map.h>
#include <mach/vm_region.h>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif  // NOMINMAX
#include <windows.h>
#endif

namespace litert::support {

std::string MappingProtection(const void* address) {
#if defined(__linux__)
  const uintptr_t target = reinterpret_cast<uintptr_t>(address);
  std::ifstream maps("/proc/self/maps");
  std::string line;
  while (std::getline(maps, line)) {
    // "begin-end perms offset dev inode path", addresses in hex.
    std::istringstream fields(line);
    uintptr_t begin = 0;
    uintptr_t end = 0;
    char dash = '\0';
    std::string perms;
    if (!(fields >> std::hex >> begin >> dash >> end >> perms)) continue;
    if (target >= begin && target < end) return perms.substr(0, 3);
  }
  return "";
#elif defined(__APPLE__)
  vm_address_t region_address = reinterpret_cast<vm_address_t>(address);
  vm_size_t region_size = 0;
  vm_region_basic_info_data_64_t info;
  mach_msg_type_number_t info_count = VM_REGION_BASIC_INFO_COUNT_64;
  mach_port_t object_name = MACH_PORT_NULL;
  const kern_return_t kr = vm_region_64(
      mach_task_self(), &region_address, &region_size, VM_REGION_BASIC_INFO_64,
      reinterpret_cast<vm_region_info_t>(&info), &info_count, &object_name);
  if (object_name != MACH_PORT_NULL) {
    mach_port_deallocate(mach_task_self(), object_name);
  }
  // vm_region_64 returns the first region at or above `address`; one that
  // starts past it means `address` itself is unmapped.
  if (kr != KERN_SUCCESS ||
      region_address > reinterpret_cast<vm_address_t>(address)) {
    return "";
  }
  std::string perms = (info.protection & VM_PROT_READ) ? "r" : "-";
  perms += (info.protection & VM_PROT_WRITE) ? "w" : "-";
  perms += (info.protection & VM_PROT_EXECUTE) ? "x" : "-";
  return perms;
#elif defined(_WIN32)
  MEMORY_BASIC_INFORMATION info;
  if (VirtualQuery(address, &info, sizeof(info)) == 0) return "";
  // The low byte holds the access protection; PAGE_GUARD and the caching
  // modifiers live above it.
  switch (info.Protect & 0xFF) {
    case PAGE_READONLY:
      return "r--";
    case PAGE_READWRITE:
    case PAGE_WRITECOPY:
      return "rw-";
    default:
      return "";
  }
#else
  (void)address;
  return "";
#endif
}

}  // namespace litert::support
