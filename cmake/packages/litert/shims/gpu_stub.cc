// stubs/gpu_stub.cc
#include "litert/core/environment.h"  // Include the header where GpuEnvironment is declared

namespace litert {
namespace internal {

class GpuEnvironment {
 public:
  // Default constructor
  GpuEnvironment() = default;

  // Default destructor - defined inline
  ~GpuEnvironment() = default;

  // Add declarations for any OTHER methods of GpuEnvironment
  // that the linker might complain about being undefined.
  // Define them as empty or default if needed.
  // Example:
  // static GpuEnvironment* Create() { return nullptr; }
  // void Initialize() {}
};

}  // namespace internal
}  // namespace litert
