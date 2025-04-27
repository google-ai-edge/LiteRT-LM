# buildifier: disable=load-on-top

workspace(name = "litert_lm")

# buildifier: disable=load-on-top

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Java rules
http_archive(
    name = "rules_java",
    sha256 = "c73336802d0b4882e40770666ad055212df4ea62cfa6edf9cb0f9d29828a0934",
    url = "https://github.com/bazelbuild/rules_java/releases/download/5.3.5/rules_java-5.3.5.tar.gz",
)

# Tensorflow
http_archive(
    name = "org_tensorflow",
    patch_cmds = [
        # Update XNNPACK latest for litert.
        "sed -i -e 's|24794834234a7926d2f553d34e84204c8ac99dfd|d10bb18be825fab45f43e3958238f0df61101426|g' tensorflow/workspace2.bzl",  # git SHA-1 tag
        "sed -i -e 's|04291b4c49693988f8c95d07968f6f3da3fd89d85bd9e4e26f73abbdfd7a8a45|5f55c355247e380e26d12758d74a22a2fc8e427b87813e998ae44e009cb48f5f|g' tensorflow/workspace2.bzl",  # SHA256
        # Update pthreadpool latest for XNNPACK.
        "sed -i -e 's|b1aee199d54003fb557076a201bcac3398af580b|bd09d5ca9155863fc47a9cbf0df1908ef9d8cad2|g' tensorflow/workspace2.bzl",  # git SHA-1 tag
        "sed -i -e 's|215724985c4845cdcadcb5f26a2a8777943927bb5a172a00e7716fe16a6f3c1b|034a78c0ef8cce1cc2ac4c90234bbb04c334000e68f7813db91ea42766870925|g' tensorflow/workspace2.bzl",  # SHA256
    ],
    sha256 = "c18d887db64ec045681c61923b74365dd7231d9731e5be3055e60bc809eb61f4",
    strip_prefix = "tensorflow-f61e140e1610f155e34908b36790bb1c8e2341c3",
    # 2025-03-04 when lite_headers_filegroup got public.
    url = "https://github.com/tensorflow/tensorflow/archive/f61e140e1610f155e34908b36790bb1c8e2341c3.tar.gz",
)

# Initialize the TensorFlow repository and all dependencies.
#
# The cascade of load() statements and tf_workspace?() calls works around the
# restriction that load() statements need to be at the top of .bzl files.
# E.g. we can not retrieve a new repository with http_archive and then load()
# a macro from that repository in the same file.
load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")

tf_workspace3()

# Initialize hermetic Python
load("@local_xla//third_party/py:python_init_rules.bzl", "python_init_rules")

python_init_rules()

load("@local_xla//third_party/py:python_init_repositories.bzl", "python_init_repositories")

python_init_repositories(
    default_python_version = "system",
    local_wheel_dist_folder = "dist",
    local_wheel_inclusion_list = [
        "tensorflow*",
        "tf_nightly*",
    ],
    local_wheel_workspaces = ["@org_tensorflow//:WORKSPACE"],
    requirements = {
        "3.9": "@org_tensorflow//:requirements_lock_3_9.txt",
        "3.10": "@org_tensorflow//:requirements_lock_3_10.txt",
        "3.11": "@org_tensorflow//:requirements_lock_3_11.txt",
        "3.12": "@org_tensorflow//:requirements_lock_3_12.txt",
    },
)

load("@local_xla//third_party/py:python_init_toolchains.bzl", "python_init_toolchains")

python_init_toolchains()

load("@local_xla//third_party/py:python_init_pip.bzl", "python_init_pip")

python_init_pip()

load("@pypi//:requirements.bzl", "install_deps")

install_deps()
# End hermetic Python initialization

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")

tf_workspace2()

load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")

tf_workspace1()

load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")

tf_workspace0()

load(
    "@local_tsl//third_party/gpus/cuda/hermetic:cuda_json_init_repository.bzl",
    "cuda_json_init_repository",
)

cuda_json_init_repository()

load(
    "@cuda_redist_json//:distributions.bzl",
    "CUDA_REDISTRIBUTIONS",
    "CUDNN_REDISTRIBUTIONS",
)
load(
    "@local_tsl//third_party/gpus/cuda/hermetic:cuda_redist_init_repositories.bzl",
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
    "@local_tsl//third_party/gpus/cuda/hermetic:cuda_configure.bzl",
    "cuda_configure",
)

cuda_configure(name = "local_config_cuda")

load(
    "@local_tsl//third_party/nccl/hermetic:nccl_redist_init_repository.bzl",
    "nccl_redist_init_repository",
)

nccl_redist_init_repository()

load(
    "@local_tsl//third_party/nccl/hermetic:nccl_configure.bzl",
    "nccl_configure",
)

nccl_configure(name = "local_config_nccl")

load("@rules_jvm_external//:defs.bzl", "maven_install")

maven_install(
    name = "maven",
    artifacts = [
        "androidx.lifecycle:lifecycle-common:2.8.7",
        "com.google.android.play:ai-delivery:0.1.1-alpha01",
        "com.google.guava:guava:33.4.6-android",
        "org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.1",
        "org.jetbrains.kotlinx:kotlinx-coroutines-guava:1.10.1",
        "org.jetbrains.kotlinx:kotlinx-coroutines-play-services:1.10.1",
    ],
    repositories = [
        "https://maven.google.com",
        "https://repo1.maven.org/maven2",
    ],
)

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
        # Update visibility to public for litert/c/BUILD and litert/cc/BUILD.
        "sed -i -e 's|//litert:litert_stable_abi_users|//visibility:public|g' litert/c/BUILD litert/cc/BUILD",
    ],
    sha256 = "1c5745dd56c968da23e48b8194914e5154a71d40d5aef6613e2d6ac9261f852a",
    strip_prefix = "LiteRT-5a816dca891292a96e4562f879cd74c1e3dd851e",
    # 2025-04-25. Tags don't include litert/.
    url = "https://github.com/google-ai-edge/LiteRT/archive/5a816dca891292a96e4562f879cd74c1e3dd851e.tar.gz",
)
