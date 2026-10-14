/* Linker dynamic-list for the LiteRT-LM C API shared library.
 *
 * Used by the libLiteRtLm.so / libLiteRtLm.dylib cc_binary target on Linux to
 * extend the default set of exported symbols with the LiteRT-LM C API surface
 * while keeping internal helpers, absl, protobuf, etc. visible to plugins
 * that need to dlsym(RTLD_DEFAULT) against them (e.g. WebGPU accelerator
 * plugin auto-registration).
 *
 * Glob patterns mirror the LiteRtLm.dll Windows .def exports — anything
 * starting with `LiteRt` (LiteRT C API and types) or `litert_lm_` (LiteRT-LM
 * C API).
 */
{
  LiteRt*;
  litert_lm_*;
};
