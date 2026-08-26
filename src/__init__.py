"""Jumpy world model source package.

Forces the CPU backend on Apple Silicon before submodules run. Entry points call it themselves.
"""

from src.utils.platform_guard import force_cpu_backend_on_apple_silicon

force_cpu_backend_on_apple_silicon()
