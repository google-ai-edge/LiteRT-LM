"""Python packaging file for a LiteRT LM library."""
import os
import pathlib
import shutil

import setuptools
import setuptools.command.build_py

Path = pathlib.Path
build_py = setuptools.command.build_py.build_py
setup = setuptools.setup


# Detect the true workspace root
# When running `uv build`, pep517 copies the package to a tmpdir.
# We need to find the REAL workspace rooted at `LiteRT-LM`.
# An easy proxy is checking if we are in a tmpdir and resolving
# the original path via environment variables (like PWD from the shell).
def _find_workspace_root():
  """Detects the true workspace root directory for the LiteRT-LM repository."""

  # If the user is running `uv build`, they run it from schema/py.
  # The original directory is exposed often via `PWD`
  original_pwd = Path(os.environ.get("PWD", ".")).resolve()
  # Try walking up from original_pwd until we see MODULE.bazel
  current = original_pwd
  for _ in range(5):
    if (current / "MODULE.bazel").exists() or (current / "WORKSPACE").exists():
      return current
    if current.parent == current:
      break
    current = current.parent
    # Fallback to the file location if not deep in an isolated build
  return Path(__file__).parent.parent.parent.resolve()

WORKSPACE_ROOT = _find_workspace_root()

PACKAGE_NAME = os.environ.get("PROJECT_NAME", "litertlm_builder")
PACKAGE_VERSION = os.environ.get("PACKAGE_VERSION", "0.0.1")


class BazelBuildPy(build_py):
  """Custom build steps to generate protobufs and flatbuffers via Bazel."""

  def run(self):
    """Runs the standard build and integrates Bazel artifacts.

    This method first runs the standard `build_py` process, then
    generates protobuf and flatbuffer artifacts using Bazel, and finally
    copies and patches these generated files into the correct package
    structure within the build directory (`build/lib/litert_lm`).
    """
    # 1. Run Bazel to build the generated files
    print("Running Bazel build for protos and flatbuffers...")
    super().run()

    # 3. Copy ALL files into the build/lib/litert_lm directory manually
    # This is because the source directory is just `schema/py`, but we
    # want the package to be `litert_lm`.
    build_lib = Path(self.build_lib).resolve()
    litert_lm_pkg_dir = build_lib / "litert_lm"
    # Ensure target directories exist
    litert_lm_pkg_dir.mkdir(parents=True, exist_ok=True)
    proto_target_dir = litert_lm_pkg_dir / "runtime" / "proto"
    proto_target_dir.mkdir(parents=True, exist_ok=True)
    schema_target_dir = litert_lm_pkg_dir / "schema" / "core"
    schema_target_dir.mkdir(parents=True, exist_ok=True)

    # A) Add __init__.py files so Python treats them as packages
    (litert_lm_pkg_dir / "__init__.py").touch(exist_ok=True)
    (litert_lm_pkg_dir / "runtime" / "__init__.py").touch(exist_ok=True)
    (litert_lm_pkg_dir / "runtime" / "proto" / "__init__.py").touch(
        exist_ok=True
    )
    (litert_lm_pkg_dir / "schema" / "__init__.py").touch(exist_ok=True)
    (litert_lm_pkg_dir / "schema" / "core" / "__init__.py").touch(exist_ok=True)
    # B) Copy current directory python files to build_lib/litert_lm
    current_dir = Path(__file__).parent.resolve()
    for filename in os.listdir(current_dir):
      if filename.endswith(".py") and filename not in [
          "setup.py",
      ]:
        src_file = current_dir / filename
        dst_file = litert_lm_pkg_dir / filename
        with open(src_file, "r", encoding="utf-8") as g:
          content = g.read()
        # Rewrite internal Bazel imports format to package structure
        content = content.replace(
            "from litert_lm.schema.py import ", "from litert_lm import "
        )
        content = content.replace(
            "import litert_lm.schema.py.", "import litert_lm."
        )
        with open(dst_file, "w", encoding="utf-8") as g:
          g.write(content)
        shutil.copystat(src_file, dst_file)

    # C) Copy Bazel Artifacts
    bazel_bin = WORKSPACE_ROOT / "bazel-bin"

    # Handle Protobufs (need patching)
    proto_bin_dir = bazel_bin / "runtime" / "proto"
    if proto_bin_dir.exists():
      for filename in os.listdir(proto_bin_dir):
        if filename.endswith("_pb2.py"):
          src_file = proto_bin_dir / filename
          dst_file = proto_target_dir / filename
          print(f"Patching and copying proto: {filename}")
          with open(src_file, "r", encoding="utf-8") as f:
            content = f.read()
          content = content.replace(
              "from runtime.proto", "from litert_lm.runtime.proto"
          )
          content = content.replace(
              "runtime.proto.", "litert_lm.runtime.proto."
          )
          with open(dst_file, "w", encoding="utf-8") as f:
            f.write(content)

    # Handle Flatbuffers (direct copy)
    schema_bin_dir = bazel_bin / "schema" / "core"
    if schema_bin_dir.exists():
      for filename in os.listdir(schema_bin_dir):
        if filename.endswith(".py") and not filename.startswith("__init__"):
          src_file = schema_bin_dir / filename
          dst_file = schema_target_dir / filename
          print(f"Copying flatbuffer module: {filename}")
          shutil.copy2(src_file, dst_file)
    print("Build py step completed: files copied and patched thoroughly.")


long_description = ""
readme_path = WORKSPACE_ROOT / "README.md"
if readme_path.exists():
  with open(readme_path, "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name=PACKAGE_NAME.replace("_", "-"),
    version=PACKAGE_VERSION,
    description="Python tools for building LiteRT-LM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/google-ai-edge/LiteRT-LM",
    author="Litert-lm Authors",
    license="Apache 2.0",
    include_package_data=True,
    # Only declare the top-level package for discovery.
    # The build_py step dynamically fills out the subpackages inside build/lib
    packages=["litert_lm"],
    package_dir={"litert_lm": "."},
    cmdclass={
        "build_py": BazelBuildPy,
    },
    install_requires=[
        "protobuf",
        "flatbuffers",
        "absl-py",
        "tomli",
    ],
    entry_points={
        "console_scripts": [
            "litertlm-builder=litert_lm.litertlm_builder_cli:main",
            "litertlm-peek=litert_lm.litertlm_peek_main:main",
        ],
    },
)

