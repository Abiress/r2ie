"""Build hook: compiles the C++ HDQ memory-manager extension via CMake.

If CMake or pybind11 are unavailable, the build is skipped gracefully and the
package falls back to the pure-Python MassState (see r2ie.hdq_bridge).
"""

import os
import shutil
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "src", "r2ie_cpp"))
        build_dir = os.path.abspath(os.path.join(self.build_temp, "r2ie_cpp"))
        out_dir = os.path.abspath(os.path.join(self.build_lib, "r2ie"))

        os.makedirs(build_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)

        # Use the same Python that runs setup (the one with pybind11).
        py_exe = sys.executable
        try:
            subprocess.check_call(
                ["cmake", "-S", src_dir, "-B", build_dir,
                 "-DCMAKE_BUILD_TYPE=Release", f"-DPython_EXECUTABLE={py_exe}"],
                cwd=build_dir,
            )
            subprocess.check_call(["cmake", "--build", build_dir, "--target", "_r2ie_cpp"], cwd=build_dir)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"[r2ie] C++ extension build skipped ({e}); using pure-Python fallback.")
            return

        # Copy the compiled module next to the Python package.
        for fname in os.listdir(build_dir):
            if fname.startswith("_r2ie_cpp") and fname.endswith((".so", ".pyd")):
                shutil.copy(os.path.join(build_dir, fname), os.path.join(out_dir, fname))
                print(f"[r2ie] C++ extension built: {fname}")


def _needs_cpp_ext():
    # Build the C++ extension unless explicitly disabled.
    return os.environ.get("R2IE_SKIP_CPP", "").lower() not in ("1", "true", "yes")


if _needs_cpp_ext():
    ext_modules = [Extension("_r2ie_cpp", sources=[])]
    cmdclass = {"build_ext": CMakeBuild}
else:
    ext_modules = []
    cmdclass = {}


setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
