# buildifier: disable=load-on-top

workspace(name = "litert_lm")

# UPDATED = 2026-10-22
LITERT_REF = "67055c6f0fc2a9b09b2963288ffee530b1069d73"

LITERT_SHA256 = "51a6ab2adb7e6a03b67356cc612d80aae2fbdab99071cac3b144f6b043f2a7e3"

TENSORFLOW_REF = "dca3c607680b991d2dcdc286d92b0e02b68d5dfb"

TENSORFLOW_SHA256 = "3a6e954614003e452bf3131078179d21cb09e0f11238e6b3d3d8e8e999fe9300"

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Declared before `rules_shell_dependencies()` and `apple_rules_dependencies()`
# because those macros also declare `bazel_skylib`, and the first declaration of
# a repository wins. The pinned TensorFlow expects 1.9.0: its `@rules_cc` uses
# the `scope` attribute on `bool_flag`, which older bazel_skylib releases do not
# define. Keep this in sync with the version in TensorFlow's
# `tensorflow/workspace3.bzl`.
http_archive(
    name = "bazel_skylib",
    sha256 = "3b5b49006181f5f8ff626ef8ddceaa95e9bb8ad294f7b5d7b11ea9f7ddaf8c59",
    urls = ["https://github.com/bazelbuild/bazel-skylib/releases/download/1.9.0/bazel-skylib-1.9.0.tar.gz"],
)

# Declared before `tf_workspace2()` so that it takes precedence over TensorFlow's
# `FP16`, whose BUILD file lacks the lowercase `fp16` target that ml_drift
# depends on (via the `@fp16` -> `@FP16` repo_mapping below). A separate
# lowercase `fp16` repository must not be declared: it collides with `FP16` in
# the `external/` directory on case-insensitive filesystems (e.g. Windows).
http_archive(
    name = "FP16",
    build_file_content = """
package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(["LICENSE"])

cc_library(
    name = "FP16",
    hdrs = glob(["include/**/*.h"]),
    includes = ["include"],
    strip_include_prefix = "include",
)

alias(
    name = "fp16",
    actual = ":FP16",
)
""",
    sha256 = "d973501a40c55126b31accc2d9f08d931ec3cc190c0430309a5e341d3c0ce32a",
    strip_prefix = "FP16-4dfe081cf6bcd15db339cf2680b9281b8451eeb3",
    url = "https://github.com/Maratyszcza/FP16/archive/4dfe081cf6bcd15db339cf2680b9281b8451eeb3.zip",
)

http_archive(
    name = "rules_shell",
    sha256 = "bc61ef94facc78e20a645726f64756e5e285a045037c7a61f65af2941f4c25e1",
    strip_prefix = "rules_shell-0.4.1",
    url = "https://github.com/bazelbuild/rules_shell/releases/download/v0.4.1/rules_shell-v0.4.1.tar.gz",
)

load("@rules_shell//shell:repositories.bzl", "rules_shell_dependencies", "rules_shell_toolchains")

rules_shell_dependencies()

rules_shell_toolchains()

http_archive(
    name = "rules_platform",
    sha256 = "0aadd1bd350091aa1f9b6f2fbcac8cd98201476289454e475b28801ecf85d3fd",
    url = "https://github.com/bazelbuild/rules_platform/releases/download/0.1.0/rules_platform-0.1.0.tar.gz",
)

# Use recent platoforms version to support uefi platform.
http_archive(
    name = "platforms",
    sha256 = "3384eb1c30762704fbe38e440204e114154086c8fc8a8c2e3e28441028c019a8",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/platforms/releases/download/1.0.0/platforms-1.0.0.tar.gz",
        "https://github.com/bazelbuild/platforms/releases/download/1.0.0/platforms-1.0.0.tar.gz",
    ],
)

# Use 3.22.0 (from 3.5.1 of tensorflow) to fix binary signing issue on MacOS Tahoe.
http_archive(
    name = "build_bazel_rules_apple",
    sha256 = "a78f26c22ac8d6e3f3fcaad50eace4d9c767688bd7254b75bdf4a6735b299f6a",
    url = "https://github.com/bazelbuild/rules_apple/releases/download/3.22.0/rules_apple.3.22.0.tar.gz",
)

load(
    "@build_bazel_rules_apple//apple:repositories.bzl",
    "apple_rules_dependencies",
)

apple_rules_dependencies()

http_archive(
    name = "build_bazel_rules_swift",
    sha256 = "f7a67197cd8a79debfe70b8cef4dc19d03039af02cc561e31e0718e98cad83ac",
    url = "https://github.com/bazelbuild/rules_swift/releases/download/2.9.0/rules_swift.2.9.0.tar.gz",
)

# Lower the version from 1.24.5 that tensorflow uses to 1.23.1, the highest version which don't have
# issues with missing LC_UUID, DEVELOPER_DIR or SDKROOT on MacOS Tahoe.
http_archive(
    name = "build_bazel_apple_support",
    sha256 = "ee20cc5c0bab47065473c8033d462374dd38d172406ecc8de5c8f08487943f2f",
    url = "https://github.com/bazelbuild/apple_support/releases/download/1.23.1/apple_support.1.23.1.tar.gz",
)

http_archive(
    name = "bazel_features",
    sha256 = "c26b4e69cf02fea24511a108d158188b9d8174426311aac59ce803a78d107648",
    strip_prefix = "bazel_features-1.43.0",
    url = "https://github.com/bazel-contrib/bazel_features/releases/download/v1.43.0/bazel_features-v1.43.0.tar.gz",
)

# Same version that tensorflow uses, but with patches to fix build errors.
http_archive(
    name = "com_google_absl",
    patch_cmds = [
        # Replace @googletest with @com_google_googletest.
        "sed -i -e 's|@googletest|@com_google_googletest|g' absl/*/BUILD* absl/*/*/BUILD* absl/*/*/*/BUILD*",
    ],
    patches = ["@//third_party/abseil:abseil.patch"],
    sha256 = "6e1aee535473414164bf83e4ebc40240dec71a4701f8a642d906e95bea1aea0c",
    strip_prefix = "abseil-cpp-20260526.0",
    url = "https://github.com/abseil/abseil-cpp/archive/20260526.0.tar.gz",
)

# Toolchains for ML projects
# Older version than one tensorflow uses as current tensorflow breaks workspace build.
# Details: https://github.com/google-ml-infra/rules_ml_toolchain
http_archive(
    name = "rules_ml_toolchain",
    sha256 = "3b05687842427041c65d1bc2f4aeda3a3079557120f5be8c34690087a88c5de5",
    strip_prefix = "rules_ml_toolchain-2eddbc595cc0bbe650c2640204f66b14f015f1a8",
    url = "https://github.com/google-ml-infra/rules_ml_toolchain/archive/2eddbc595cc0bbe650c2640204f66b14f015f1a8.tar.gz",
)

# Kotlin rules (must be declared before tf_workspace to override)
http_archive(
    name = "rules_kotlin",
    sha256 = "e40ccc013f874e063bb64fed0f816b881991cb300bce45085e1204644892c59c",
    url = "https://github.com/bazel-contrib/rules_kotlin/releases/download/v2.4.10/rules_kotlin-v2.4.10.tar.gz",
)

# Android rules (must be declared before kotlin_repositories and tf_workspace to override
# older versions whose com_android_dex and apksig archives lacked sha256 and mirror URLs;
# pinned to v0.7.2 because v0.7.3+ requires @protobuf_maven for @com_google_protobuf//java/util)
http_archive(
    name = "rules_android",
    sha256 = "0da7198c7c8bac7e11e08dca3c434617b8593075858716595672e9aeefbef2a7",
    strip_prefix = "rules_android-0.7.2",
    url = "https://github.com/bazelbuild/rules_android/releases/download/v0.7.2/rules_android-v0.7.2.tar.gz",
)

# Go rules and Gazelle for rules_android (must be declared before tf_workspace to override ancient versions)
http_archive(
    name = "io_bazel_rules_go",
    sha256 = "68af54cb97fbdee5e5e8fe8d210d15a518f9d62abfd71620c3eaff3b26a5ff86",
    urls = [
        "https://mirror.bazel.build/github.com/bazel-contrib/rules_go/releases/download/v0.59.0/rules_go-v0.59.0.zip",
        "https://github.com/bazel-contrib/rules_go/releases/download/v0.59.0/rules_go-v0.59.0.zip",
    ],
)

http_archive(
    name = "bazel_gazelle",
    sha256 = "675114d8b433d0a9f54d81171833be96ebc4113115664b791e6f204d58e93446",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-gazelle/releases/download/v0.47.0/bazel-gazelle-v0.47.0.tar.gz",
        "https://github.com/bazelbuild/bazel-gazelle/releases/download/v0.47.0/bazel-gazelle-v0.47.0.tar.gz",
    ],
)

# TensorFlow
http_archive(
    name = "org_tensorflow",
    patches = ["@//third_party/tensorflow:tensorflow.patch"],
    sha256 = TENSORFLOW_SHA256,
    strip_prefix = "tensorflow-" + TENSORFLOW_REF,
    url = "https://github.com/tensorflow/tensorflow/archive/" + TENSORFLOW_REF + ".tar.gz",
)

# Initialize the TensorFlow repository and all dependencies.
#
# The cascade of load() statements and tf_workspace?() calls works around the
# restriction that load() statements need to be at the top of .bzl files.
# E.g. we can not retrieve a new repository with http_archive and then load()
# a macro from that repository in the same file.
load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")

tf_workspace3()

load("@bazel_features//:deps.bzl", "bazel_features_deps")

bazel_features_deps()

load("@rules_cc//cc:extensions.bzl", "compatibility_proxy_repo")

compatibility_proxy_repo()

load("@io_bazel_rules_go//go:deps.bzl", "go_rules_dependencies")

go_rules_dependencies()

load(
    "@rules_ml_toolchain//cc/deps:cc_toolchain_deps.bzl",
    "cc_toolchain_deps",
)

cc_toolchain_deps()

# Initialize hermetic Python
load("@xla//third_party/py:python_init_rules.bzl", "python_init_rules")

python_init_rules()

load("@xla//third_party/py:python_init_repositories.bzl", "python_init_repositories")

python_init_repositories(
    default_python_version = "system",
    requirements = {
        "3.10": "@org_tensorflow//:requirements_lock_3_10.txt",
        "3.11": "@org_tensorflow//:requirements_lock_3_11.txt",
        "3.12": "@org_tensorflow//:requirements_lock_3_12.txt",
        "3.13": "@org_tensorflow//:requirements_lock_3_13.txt",
        "3.14": "@org_tensorflow//:requirements_lock_3_14.txt",
    },
)

load("@xla//third_party/py:python_init_toolchains.bzl", "python_init_toolchains")

python_init_toolchains()

load("@xla//third_party/py:python_init_pip.bzl", "python_init_pip")

python_init_pip()

load("@pypi//:requirements.bzl", "install_deps")

install_deps()
# End hermetic Python initialization

RULES_JVM_EXTERNAL_TAG = "6.8"

RULES_JVM_EXTERNAL_SHA = "704a0197e4e966f96993260418f2542568198490456c21814f647ae7091f56f2"

http_archive(
    name = "rules_jvm_external",
    sha256 = RULES_JVM_EXTERNAL_SHA,
    strip_prefix = "rules_jvm_external-%s" % RULES_JVM_EXTERNAL_TAG,
    url = "https://github.com/bazelbuild/rules_jvm_external/releases/download/%s/rules_jvm_external-%s.tar.gz" % (RULES_JVM_EXTERNAL_TAG, RULES_JVM_EXTERNAL_TAG),
)

load("@rules_jvm_external//:defs.bzl", "maven_install")

maven_install(
    name = "maven",
    artifacts = [
        "com.google.code.gson:gson:2.14.0",
        "org.jetbrains.kotlinx:kotlinx-coroutines-core-jvm:1.11.0",
        "org.jetbrains.kotlinx:kotlinx-coroutines-android:1.11.0",
    ],
    repositories = [
        "https://maven.google.com",
        "https://repo1.maven.org/maven2",
    ],
)

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")

tf_workspace2()

load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")

tf_workspace1()

load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")

tf_workspace0()

load(
    "@xla//third_party/py:python_wheel.bzl",
    "python_wheel_version_suffix_repository",
)

python_wheel_version_suffix_repository(name = "tf_wheel_version_suffix")

load(
    "@rules_ml_toolchain//gpu/cuda:cuda_json_init_repository.bzl",
    "cuda_json_init_repository",
)

cuda_json_init_repository()

load(
    "@cuda_redist_json//:distributions.bzl",
    "CUDA_REDISTRIBUTIONS",
    "CUDNN_REDISTRIBUTIONS",
)
load(
    "@rules_ml_toolchain//gpu/cuda:cuda_redist_init_repositories.bzl",
    "cuda_redist_init_repositories",
    "cudnn_redist_init_repository",
)

cuda_redist_init_repositories(
    cuda_redistributions = CUDA_REDISTRIBUTIONS,
)

cudnn_redist_init_repository(
    cudnn_redistributions = CUDNN_REDISTRIBUTIONS,
)

load(
    "@rules_ml_toolchain//gpu/cuda:cuda_configure.bzl",
    "cuda_configure",
)

cuda_configure(name = "local_config_cuda")

load(
    "@rules_ml_toolchain//gpu/nccl:nccl_redist_init_repository.bzl",
    "nccl_redist_init_repository",
)

nccl_redist_init_repository()

load(
    "@rules_ml_toolchain//gpu/nccl:nccl_configure.bzl",
    "nccl_configure",
)

nccl_configure(name = "local_config_nccl")

# Kotlin rules

load("@rules_kotlin//kotlin:repositories.bzl", "kotlin_repositories")

kotlin_repositories()  # if you want the default. Otherwise see custom kotlinc distribution below

load("@rules_kotlin//kotlin:core.bzl", "kt_register_toolchains")

kt_register_toolchains()  # to use the default toolchain, otherwise see toolchains below

# Rust (for HuggingFace Tokenizers)
http_archive(
    name = "rules_rust",
    patches = ["@//third_party/rules_rust:rules_rust.patch"],
    sha256 = "53c1bac7ec48f7ce48c4c1c6aa006f27515add2aeb05725937224e6e00ec7cea",
    url = "https://github.com/bazelbuild/rules_rust/releases/download/0.61.0/rules_rust-0.61.0.tar.gz",
)

load("@rules_rust//rust:repositories.bzl", "rules_rust_dependencies", "rust_register_toolchains")

rules_rust_dependencies()

rust_register_toolchains(
    edition = "2021",
    extra_target_triples = [
        # Explicitly add toolchains for mobile. Desktop platforms are supported by default.
        "aarch64-linux-android",
        "aarch64-apple-ios",
        "aarch64-apple-ios-sim",
        "x86_64-linux-android",
        "x86_64-apple-darwin",
    ],
)

load("@rules_rust//crate_universe:repositories.bzl", "crate_universe_dependencies")

crate_universe_dependencies()

load("@rules_rust//crate_universe:defs.bzl", "crate", "crates_repository")
load("@rules_rust//rust/platform:triple_mappings.bzl", "SUPPORTED_PLATFORM_TRIPLES")

# NOTE: llguidance's build file and the llguidance/toktrie patches below live at
# the repository root rather than under third_party/ with the other dependencies.
# rules_rust hashes these label strings into the checked-in cargo-bazel-lock.json,
# so renaming them requires re-pinning the lockfile with
# `CARGO_BAZEL_REPIN=1 bazel sync --only=crate_index`.
crates_repository(
    name = "crate_index",
    annotations = {
        "llguidance": [
            crate.annotation(
                additive_build_file = "@//:BUILD.llguidance",
                gen_build_script = False,
                patches = [
                    "@//:PATCH.llguidance_regexvec",
                    "@//:PATCH.llguidance_numeric",
                    "@//:PATCH.llguidance_grammar",
                    "@//:PATCH.llguidance_perf",
                ],
            ),
        ],
        "toktrie": [
            crate.annotation(
                patches = ["@//:PATCH.toktrie"],
            ),
        ],
    },
    cargo_lockfile = "//:Cargo.lock",
    lockfile = "//:cargo-bazel-lock.json",
    manifests = [
        "//:Cargo.toml",
    ],
    supported_platform_triples = SUPPORTED_PLATFORM_TRIPLES + [
        "x86_64-linux-android",
    ],
)

load("@crate_index//:defs.bzl", "crate_repositories")

crate_repositories()

# cxxbridge-cmd is a binary-only package so we follow the steps in
# https://bazelbuild.github.io/rules_rust/crate_universe_workspace.html#binary-dependencies.
http_archive(
    name = "cxxbridge_cmd",
    build_file = "//cxxbridge_cmd:BUILD.cxxbridge_cmd.bazel",
    integrity = "sha256-pf/3kWu94FwtuZRp8J3PryA78lsJbMv052GgR5JBLhA=",
    strip_prefix = "cxxbridge-cmd-1.0.149",
    type = "tar.gz",
    url = "https://static.crates.io/crates/cxxbridge-cmd/cxxbridge-cmd-1.0.149.crate",
)

crates_repository(
    name = "cxxbridge_cmd_deps",
    cargo_lockfile = "//cxxbridge_cmd:Cargo.lock",
    manifests = ["@cxxbridge_cmd//:Cargo.toml"],
)

load("@cxxbridge_cmd_deps//:defs.bzl", cxxbridge_cmd_deps = "crate_repositories")

cxxbridge_cmd_deps()

# Same one downloaded by tensorflow, but refer contrib/minizip.
http_archive(
    name = "minizip",
    add_prefix = "minizip",
    build_file = "@//third_party/minizip:minizip.BUILD",
    sha256 = "9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23",
    strip_prefix = "zlib-1.3.1/contrib/minizip",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/zlib.net/fossils/zlib-1.3.1.tar.gz",
        "https://mirror.bazel.build/zlib.net/fossils/zlib-1.3.1.tar.gz",
        "https://zlib.net/fossils/zlib-1.3.1.tar.gz",
    ],
)

http_archive(
    name = "sentencepiece",
    build_file = "@//third_party/sentencepiece:sentencepiece.BUILD",
    patch_cmds = [
        "printf '#ifndef CONFIG_H_\\n#define CONFIG_H_\\n#define VERSION \"0.2.2\"\\n#define PACKAGE \"sentencepiece\"\\n#define PACKAGE_STRING \"sentencepiece\"\\n#define INSTALL_DATADIR \"\"\\n#endif\\n' > config.h",
        "mv src/* .",
        # Replace third_party/absl/ with absl/ in *.h and *.cc files.
        "sed -i -e 's|#include \"third_party/absl/|#include \"absl/|g' *.h *.cc",
        # Replace third_party/protobuf-lite/ with google/protobuf/ in *.h and *.cc files.
        "sed -i -e 's|#include \"third_party/protobuf-lite/google/protobuf/|#include \"google/protobuf/|g' *.h *.cc",
        "sed -i -e 's|#include \"third_party/protobuf/|#include \"google/protobuf/|g' *.h *.cc",
        "rm -rf third_party/protobuf-lite third_party/protobuf third_party/absl third_party/abseil-cpp",
    ],
    # TODO: b/570513475 remove this patch once we guarantee that sentencepiece tokenizer is properly
    # prepared for all supported models.
    patches = ["@//third_party/sentencepiece:sentencepiece.patch"],
    sha256 = "92381f713e094a15a1ccff1ac4a5315a4c4b82a99ac1332d6ac53c9dc8e1bcf1",
    strip_prefix = "sentencepiece-0.2.2",
    url = "https://github.com/google/sentencepiece/archive/refs/tags/v0.2.2.tar.gz",
)

http_archive(
    name = "litert",
    patch_cmds = [
        # Replace @//third_party with @litert//third_party in files under third_party/.
        # Only regular files are passed to `sed -i`, which fails on directories such as
        # third_party/py/numpy. Uses a shell loop rather than `find`, which may resolve
        # to find.exe on Windows.
        "for f in third_party/*/*; do if [ -f \"$f\" ]; then sed -i -e 's|\"@//third_party/|\"@litert//third_party/|g' \"$f\" || exit 1; fi; done",
        "sed -i -e 's|build_file = \"//third_party/dawn:dawn.BUILD\"|build_file = \"@litert//third_party/dawn:dawn.BUILD\"|g' third_party/dawn/workspace.bzl",
        "sed -i -e 's|python3 -c|$(if python3 -V >/dev/null 2>/dev/null; then echo python3; else echo python; fi) -c|g' third_party/dawn/workspace.bzl",
        # TODO: b/465678331 - Remove patch_cmds for visibility and legacy_create_init once
        # cl/1000909231 is submitted and LITERT_REF is updated.
        # Allow external @litert_lm workspace to depend on ml_drift_delegate and litert_runtime_builtin_none targets.
        "sed -i -e '/default_visibility/,/\\]/d' ml_drift_delegate/delegate/BUILD",
        "sed -i -e 's|package(|package(default_visibility = [\"//visibility:public\"],|g' ml_drift_delegate/delegate/BUILD",
        "sed -i -e 's|visibility = litert_jni_friends() + \\[|visibility = litert_jni_friends() + [\"@//:__subpackages__\", |g' litert/c/internal/BUILD",
        # Remove unsupported aspect_hints from ml_drift_delegate BUILD files.
        "sed -i -e '/no_module/d' ml_drift_delegate/delegate/BUILD ml_drift_delegate/delegate/shared_memory_manager/BUILD",
        # Fix unscrubbed path in gpu_backend_metal_litert.mm.
        "sed -i -e 's|#import \".*ml_drift/|#import \"ml_drift_delegate/|g' ml_drift_delegate/delegate/gpu_backend_metal_litert.mm",
        # Disable implicit __init__.py creation in dawn_json_generator and generate_build_identifier_py so @jinja2 is not shadowed.
        "sed -i -e 's|name = \"dawn_json_generator\",|name = \"dawn_json_generator\", legacy_create_init = 0,|g' third_party/dawn/dawn.BUILD",
        "sed -i -e 's|\"\\:dawn_native\",|\":dawn_native\", \"@//runtime/components:webgpu_dawn_shared_lib\",|g' third_party/dawn/dawn.BUILD",
        "sed -i -e 's|name = \"generate_build_identifier_py\",|name = \"generate_build_identifier_py\", legacy_create_init = 0,|g' ml_drift_delegate/delegate/serialization_weight_cache/BUILD",
    ],
    sha256 = LITERT_SHA256,
    strip_prefix = "LiteRT-" + LITERT_REF,
    url = "https://github.com/google-ai-edge/LiteRT/archive/" + LITERT_REF + ".tar.gz",
)

# SafeInt. Same version that LiteRT uses.
http_archive(
    name = "SafeInt",
    build_file = "@litert//:BUILD.SafeInt",
    sha256 = "4bd737e5885112e796742b7b6ba7181ff346855dc22d29ed2b258de567a1de8d",
    strip_prefix = "SafeInt-72f0745e26091af6d1186fd91426377f16688ba7",
    url = "https://github.com/dcleblanc/SafeInt/archive/72f0745e26091af6d1186fd91426377f16688ba7.tar.gz",
)

ML_DRIFT_COMMIT = "1b18771f44c0be3d972847b31a80f6dc3d6964b5"

ML_DRIFT_SHA256 = "0a7384901754a4ee84220561bcf5447fde4363cbc8a42c94743f183d178feb30"

http_archive(
    name = "ml_drift",
    patch_cmds = [
        # TODO: b/465678331 - Remove patch_cmds once cl/1000935259 is submitted and ML_DRIFT_COMMIT is updated.
        # Add external/ml_drift to flatc include paths for gl_inference_context_cc_fbs.
        "sed -i -e 's|\"-I .\",|\"-I .\", \"-I external/ml_drift\",|g' ml_drift/gl/BUILD",
        # Point ml_drift_delegate references to @litert//ml_drift_delegate.
        "sed -i -e 's|@//ml_drift_delegate|@litert//ml_drift_delegate|g' ml_drift/common/default/BUILD ml_drift/common/task/BUILD ml_drift/common/kernels/BUILD",
        # Avoid passing -ldl to MSVC linker on Windows in opencl_wrapper.
        "sed -i -e 's|\"//conditions:default\": \\[\"-ldl\"\\],|\"@platforms//os:windows\": [], \"//conditions:default\": [\"-ldl\"],|g' ml_drift/cl/BUILD",
        # Use @FP16//:FP16 from tf_workspace2() to avoid case-insensitive @fp16/@FP16 collision on Windows.
        "sed -i -e 's|\"@fp16\"|\"@FP16//:FP16\"|g' ml_drift/common/BUILD",
        # Avoid int -> uint32_t narrowing conversion in VulkanInfo default member initializers on MSVC.
        "sed -i -e 's|uint32_t api_version\\(.*\\) = -1;|uint32_t api_version\\1 = static_cast<uint32_t>(-1);|g' ml_drift/common/gpu_info.h",
    ],
    repo_mapping = {
        "@fp16": "@FP16",
        "@com_github_google_re2": "@com_googlesource_code_re2",
    },
    sha256 = ML_DRIFT_SHA256,
    strip_prefix = "ml-drift-" + ML_DRIFT_COMMIT,
    urls = [
        "https://github.com/google-ai-edge/ml-drift/archive/" + ML_DRIFT_COMMIT + ".tar.gz",
    ],
)

EGL_HEADERS_COMMIT = "db3425b8246136faccb5e2782b5694960bd6edf1"

http_archive(
    name = "egl_headers",
    build_file = "@ml_drift//third_party/egl_headers:egl_headers.BUILD",
    sha256 = "3d6b42b23da8030f5b29ade07d7384c5f885e713f17f04650f6cd4b8329b88e8",
    strip_prefix = "EGL-Registry-" + EGL_HEADERS_COMMIT + "/api",
    urls = [
        "https://github.com/KhronosGroup/EGL-Registry/archive/" + EGL_HEADERS_COMMIT + ".tar.gz",
    ],
)

OPENGL_HEADERS_COMMIT = "1cdd228e34966dd6b95bd203e9f84faba0f371a1"

http_archive(
    name = "opengl_headers",
    build_file = "@ml_drift//third_party/opengl_headers:opengl_headers.BUILD",
    sha256 = "d8ac2e70b4a8173d4e30ceb726eb8b3137883d459e4666983eb1d2444483c62c",
    strip_prefix = "OpenGL-Registry-" + OPENGL_HEADERS_COMMIT + "/api",
    urls = [
        "https://github.com/KhronosGroup/OpenGL-Registry/archive/" + OPENGL_HEADERS_COMMIT + ".tar.gz",
    ],
)

load("@litert//third_party/markupsafe:workspace.bzl", markupsafe = "repo")

markupsafe()

load("@litert//third_party/jinja2:workspace.bzl", jinja2 = "repo")

jinja2()

load("@litert//third_party/dawn:workspace.bzl", dawn = "repo")

dawn()

http_archive(
    name = "tokenizers_cpp",
    build_file = "@//third_party/tokenizers_cpp:tokenizers_cpp.BUILD",
    sha256 = "3e0b9ec325a326b0a2cef5cf164ee94a74ac372c5881ae5af634036db0441823",
    strip_prefix = "tokenizers-cpp-0.1.1",
    url = "https://github.com/mlc-ai/tokenizers-cpp/archive/refs/tags/v0.1.1.tar.gz",
)

http_archive(
    name = "absl_py",
    sha256 = "8a3d0830e4eb4f66c4fa907c06edf6ce1c719ced811a12e26d9d3162f8471758",
    strip_prefix = "abseil-py-2.1.0",
    url = "https://github.com/abseil/abseil-py/archive/refs/tags/v2.1.0.tar.gz",
)

http_archive(
    name = "nlohmann_json",
    sha256 = "34660b5e9a407195d55e8da705ed26cc6d175ce5a6b1fb957e701fb4d5b04022",
    strip_prefix = "json-3.12.0",
    url = "https://github.com/nlohmann/json/archive/refs/tags/v3.12.0.zip",
)

http_archive(
    name = "miniaudio",
    build_file = "@//third_party/miniaudio:miniaudio.BUILD",
    sha256 = "bcb07bfb27e6fa94d34da73ba2d5642d4940b208ec2a660dbf4e52e6b7cd492f",
    strip_prefix = "miniaudio-0.11.22",
    url = "https://github.com/mackron/miniaudio/archive/refs/tags/0.11.22.tar.gz",
)

http_archive(
    name = "stb",
    build_file = "@//third_party/stb:stb.BUILD",
    sha256 = "119b9f3cca3e50225dc946ed1acd1b7a160943bc8bf549760109cea4e4e7c836",
    strip_prefix = "stb-f58f558c120e9b32c217290b80bad1a0729fbb2c",
    url = "https://github.com/nothings/stb/archive/f58f558c120e9b32c217290b80bad1a0729fbb2c.zip",
)

http_archive(
    name = "skia",
    patch_args = ["-p1"],
    patch_cmds = [
        # Replace <jpeglib.h> with "jpeglib.h".
        "sed -i -e 's|#include <jpeglib.h>|#include \"jpeglib.h\"|g' */*.cpp */*.h */*/*.cpp */*/*.h",
    ],
    patches = ["@//third_party/skia:skia.patch"],
    repo_mapping = {
        "@libpng": "@png",
    },
    sha256 = "2fe28173428f8eebf2aa8a665bad32136086cc065f50c7154678a96250d1cde1",
    strip_prefix = "skia-226ae9d866748a2e68b6dbf114b37129c380a298",
    urls = ["https://github.com/google/skia/archive/226ae9d866748a2e68b6dbf114b37129c380a298.zip"],
)

http_archive(
    name = "skia_user_config",
    patch_args = ["-p1"],
    patches = ["@//third_party/skia:skia_user_config.patch"],
    sha256 = "2fe28173428f8eebf2aa8a665bad32136086cc065f50c7154678a96250d1cde1",
    strip_prefix = "skia-226ae9d866748a2e68b6dbf114b37129c380a298/include/config",
    urls = ["https://github.com/google/skia/archive/226ae9d866748a2e68b6dbf114b37129c380a298.zip"],
)

http_archive(
    name = "cppjieba",
    build_file = "@//third_party/cppjieba:cppjieba.BUILD",
    patches = ["@//third_party/cppjieba:cppjieba.patch"],
    sha256 = "8b27931630abab1136b7c3ee920e4a95e78cb84a3cab4670198172bc82104844",
    strip_prefix = "OpenCC-025f371dc76b598d77384fbdab90c937471844d8/plugins/jieba/deps/cppjieba",
    urls = ["https://github.com/BYVoid/OpenCC/archive/025f371dc76b598d77384fbdab90c937471844d8.tar.gz"],
)

# Android rules ####################################################################################

# Android SDK
load("@rules_android//:prereqs.bzl", "rules_android_prereqs")

rules_android_prereqs()

load("@rules_android//:defs.bzl", "rules_android_workspace")

rules_android_workspace()

load("@rules_java//java:repositories.bzl", "java_tools_repos")

java_tools_repos()

load("@rules_android//rules:rules.bzl", "android_sdk_repository")

android_sdk_repository(name = "androidsdk")

register_toolchains(
    "@rules_android//toolchains/android:android_default_toolchain",
    "@rules_android//toolchains/android_sdk:android_sdk_tools",
)

# Android NDK. Need latest rules_android_ndk to use NDK 26+.
load("@rules_android_ndk//:rules.bzl", "android_ndk_repository")

android_ndk_repository(name = "androidndk")

# Configure Android NDK only when ANDROID_NDK_HOME is set.
# Creates current_android_ndk_env.bzl as a workaround since shell environment is available only
# through repository rule's context.
load("//:android_ndk_env.bzl", "check_android_ndk_env")

check_android_ndk_env(name = "android_ndk_env")

load("@android_ndk_env//:current_android_ndk_env.bzl", "ANDROID_NDK_HOME_IS_SET")

# Use "@android_ndk_env//:all" as a dummy toolchain target as register_toolchains() does not take
# an empty string.
register_toolchains("@androidndk//:all" if ANDROID_NDK_HOME_IS_SET else "@android_ndk_env//:all")

# VENDOR SDKS ######################################################################################

# QUALCOMM ---------------------------------------------------------------------------------------

# The actual macro call will be set during configure for now.
load("@litert//third_party/qairt:workspace.bzl", "qairt")

qairt()

# MEDIATEK ---------------------------------------------------------------------------------------

# Currently only works with local sdk
load("@litert//third_party/neuro_pilot:workspace.bzl", "neuro_pilot")

neuro_pilot()

# GOOGLE TENSOR ----------------------------------------------------------------------------------
load("@litert//third_party/google_tensor:workspace.bzl", "google_tensor")

google_tensor()

# INTEL OPENVINO ---------------------------------------------------------------------------------
load("@litert//third_party/intel_openvino:openvino.bzl", "openvino_configure")

openvino_configure()

# SAMSUNG EXYNOS_AI_LITECORE----------------------------------------------------------------------
load("@litert//third_party/exynos_ai_litecore:workspace.bzl", "exynos_ai_litecore")

exynos_ai_litecore()

load("@rules_python//python:pip.bzl", "pip_parse")

pip_parse(
    name = "custom_pip_deps",
    extra_pip_args = ["--index-url=https://pypi.org/simple"],
    requirements_lock = "//:requirements.txt",
)

load("@custom_pip_deps//:requirements.bzl", install_custom_deps = "install_deps")

install_custom_deps()

# DirectX Shader Compiler DLLs for Windows
http_archive(
    name = "directx_shader_compiler",
    build_file = "@//third_party/directx_shader_compiler:directx_shader_compiler.BUILD",
    sha256 = "a1e89031421cf3c1fca6627766ab3020ca4f962ac7e2caa7fab2b33a8436151e",
    url = "https://github.com/microsoft/DirectXShaderCompiler/releases/download/v1.9.2602/dxc_2026_02_20.zip",
)

http_archive(
    name = "patchelf_linux_x86_64",
    build_file_content = """
filegroup(
    name = "patchelf",
    srcs = ["bin/patchelf"],
    visibility = ["//visibility:public"],
)
""",
    sha256 = "ce84f2447fb7a8679e58bc54a20dc2b01b37b5802e12c57eece772a6f14bf3f0",
    url = "https://github.com/NixOS/patchelf/releases/download/0.18.0/patchelf-0.18.0-x86_64.tar.gz",
)

http_archive(
    name = "patchelf_linux_arm64",
    build_file_content = """
filegroup(
    name = "patchelf",
    srcs = ["bin/patchelf"],
    visibility = ["//visibility:public"],
)
""",
    sha256 = "ae13e2effe077e829be759182396b931d8f85cfb9cfe9d49385516ea367ef7b2",
    url = "https://github.com/NixOS/patchelf/releases/download/0.18.0/patchelf-0.18.0-aarch64.tar.gz",
)
