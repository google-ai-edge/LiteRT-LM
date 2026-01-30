# buildifier: disable=load-on-top

workspace(name = "litert_lm")

# UPDATED = 2026-01-30
LITERT_REF = "cc3278e8165a52496a5c3578cc1c162acd5ee4ac"

LITERT_SHA256 = "547a410381fc51a84744b0fb2421bf881123774ba382d8f156625f37b0462c06"

TENSORFLOW_REF = "44b586672cb376833121d0acc924aba1bb020aef"

TENSORFLOW_SHA256 = "f002f200b8af1aa8bfea14add3960fa97eb9f5a907c7160a6f831e4a5b4b6c1a"

# buildifier: disable=load-on-top

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_jar")

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

# Lower the version from 1.24.5 that tensorflow uses to 1.23.1, the highest version which don't have
# issues with missing LC_UUID, DEVELOPER_DIR or SDKROOT on MacOS Tahoe.
http_archive(
    name = "build_bazel_apple_support",
    sha256 = "ee20cc5c0bab47065473c8033d462374dd38d172406ecc8de5c8f08487943f2f",
    url = "https://github.com/bazelbuild/apple_support/releases/download/1.23.1/apple_support.1.23.1.tar.gz",
)

# TensorFlow
http_archive(
    name = "org_tensorflow",
    patches = ["@//:PATCH.tensorflow"],
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

# Toolchains for ML projects
# Details: https://github.com/google-ml-infra/rules_ml_toolchain
http_archive(
    name = "rules_ml_toolchain",
    sha256 = "9dbee8f24cc1b430bf9c2a6661ab70cbca89979322ddc7742305a05ff637ab6b",
    strip_prefix = "rules_ml_toolchain-545c80f1026d526ea9c7aaa410bf0b52c9a82e74",
    url = "https://github.com/google-ml-infra/rules_ml_toolchain/archive/545c80f1026d526ea9c7aaa410bf0b52c9a82e74.tar.gz",
)

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
    local_wheel_dist_folder = "dist",
    local_wheel_inclusion_list = [
        "tensorflow*",
        "tf_nightly*",
    ],
    local_wheel_workspaces = ["@org_tensorflow//:WORKSPACE"],
    requirements = {
        "3.10": "@org_tensorflow//:requirements_lock_3_10.txt",
        "3.11": "@org_tensorflow//:requirements_lock_3_11.txt",
        "3.12": "@org_tensorflow//:requirements_lock_3_12.txt",
        "3.13": "@org_tensorflow//:requirements_lock_3_13.txt",
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
        "com.google.code.gson:gson:2.13.2",
        "org.jetbrains.kotlinx:kotlinx-coroutines-core-jvm:1.9.0",
        "org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0",
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
http_archive(
    name = "rules_kotlin",
    sha256 = "e1448a56b2462407b2688dea86df5c375b36a0991bd478c2ddd94c97168125e2",
    url = "https://github.com/bazelbuild/rules_kotlin/releases/download/v2.1.3/rules_kotlin-v2.1.3.tar.gz",
)

load("@rules_kotlin//kotlin:repositories.bzl", "kotlin_repositories")

kotlin_repositories()  # if you want the default. Otherwise see custom kotlinc distribution below

load("@rules_kotlin//kotlin:core.bzl", "kt_register_toolchains")

kt_register_toolchains()  # to use the default toolchain, otherwise see toolchains below

# Rust (for HuggingFace Tokenizers)
http_archive(
    name = "rules_rust",
    patches = ["@//:PATCH.rules_rust"],
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
    ],
)

load("@rules_rust//crate_universe:repositories.bzl", "crate_universe_dependencies")

crate_universe_dependencies()

load("@rules_rust//crate_universe:defs.bzl", "crate", "crates_repository")

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
                    "@//:PATCH.llguidance_parser",
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
    build_file = "@//:BUILD.minizip",
    sha256 = "9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23",
    strip_prefix = "zlib-1.3.1/contrib/minizip",
    url = "https://zlib.net/fossils/zlib-1.3.1.tar.gz",
)

http_archive(
    name = "sentencepiece",
    build_file = "@//:BUILD.sentencepiece",
    patch_cmds = [
        # Empty config.h seems enough.
        "touch config.h",
        # Replace third_party/absl/ with absl/ in *.h and *.cc files.
        "sed -i -e 's|#include \"third_party/absl/|#include \"absl/|g' *.h *.cc",
        # Replace third_party/darts_clone/ with include/ in *.h and *.cc files.
        "sed -i -e 's|#include \"third_party/darts_clone/|#include \"include/|g' *.h *.cc",
    ],
    patches = ["@//:PATCH.sentencepiece"],
    sha256 = "9970f0a0afee1648890293321665e5b2efa04eaec9f1671fcf8048f456f5bb86",
    strip_prefix = "sentencepiece-0.2.0/src",
    url = "https://github.com/google/sentencepiece/archive/refs/tags/v0.2.0.tar.gz",
)

http_archive(
    name = "darts_clone",
    build_file = "@//:BUILD.darts_clone",
    sha256 = "4a562824ec2fbb0ef7bd0058d9f73300173d20757b33bb69baa7e50349f65820",
    strip_prefix = "darts-clone-e40ce4627526985a7767444b6ed6893ab6ff8983",
    url = "https://github.com/s-yata/darts-clone/archive/e40ce4627526985a7767444b6ed6893ab6ff8983.tar.gz",
)

http_archive(
    name = "litert",
    patch_cmds = [
        # Replace @//third_party with @litert//third_party in files under third_party/.
        "sed -i -e 's|\"@//third_party/|\"@litert//third_party/|g' third_party/*/*",
    ],
    sha256 = LITERT_SHA256,
    strip_prefix = "LiteRT-" + LITERT_REF,
    url = "https://github.com/google-ai-edge/LiteRT/archive/" + LITERT_REF + ".tar.gz",
)

http_archive(
    name = "tokenizers_cpp",
    build_file = "@//:BUILD.tokenizers_cpp",
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
    name = "minja",
    build_file = "@//:BUILD.minja",
    patches = ["@//:PATCH.minja"],
    sha256 = "752f47dd2a2f4920a66f497c952785073c1983f12f084b99e5c12bf89f96acfe",
    strip_prefix = "minja-58568621432715b0ed38efd16238b0e7ff36c3ba",
    url = "https://github.com/google/minja/archive/58568621432715b0ed38efd16238b0e7ff36c3ba.zip",
)

http_archive(
    name = "miniaudio",
    build_file = "@//:BUILD.miniaudio",
    sha256 = "bcb07bfb27e6fa94d34da73ba2d5642d4940b208ec2a660dbf4e52e6b7cd492f",
    strip_prefix = "miniaudio-0.11.22",
    url = "https://github.com/mackron/miniaudio/archive/refs/tags/0.11.22.tar.gz",
)

http_archive(
    name = "stb",
    build_file = "@//:BUILD.stb",
    sha256 = "119b9f3cca3e50225dc946ed1acd1b7a160943bc8bf549760109cea4e4e7c836",
    strip_prefix = "stb-f58f558c120e9b32c217290b80bad1a0729fbb2c",
    url = "https://github.com/nothings/stb/archive/f58f558c120e9b32c217290b80bad1a0729fbb2c.zip",
)

http_jar(
    name = "javax_json",
    sha256 = "0e1dec40a1ede965941251eda968aeee052cc4f50378bc316cc48e8159bdbeb4",
    url = "https://jcenter.bintray.com/org/glassfish/javax.json/1.0.4/javax.json-1.0.4.jar",
)

# Android rules. Need latest rules_android_ndk to use NDK 26+.
load("@rules_android_ndk//:rules.bzl", "android_ndk_repository")

android_ndk_repository(name = "androidndk")

android_sdk_repository(name = "androidsdk")

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
