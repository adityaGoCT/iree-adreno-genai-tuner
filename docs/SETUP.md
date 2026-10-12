# Host setup

Everything below was used on macOS (Apple silicon). Nothing is installed on the phone's system
partition; on-device files live in `/data/local/tmp/{iree,iree312,turnip}`.

| Tool | Version used | Install | Why |
|---|---|---|---|
| Homebrew | 7.x | https://brew.sh | package manager |
| cmake, ninja | 4.2 / 1.13 | `brew install cmake ninja` | IREE runtime cross-builds |
| Android NDK | r30 | `brew install --cask android-ndk` | arm64 toolchain (`ANDROID_NDK`) |
| adb | 37 | `brew install --cask android-platform-tools` | phone over USB or Wi-Fi |
| uv + Python 3.12 | 0.10 | https://docs.astral.sh/uv/ | venvs (IREE wheels: cp311/cp312) |
| meson, bison | 1.12 / 3.8 | `brew install meson bison` | Mesa Turnip build (bison is keg-only: `BISON_DIR`) |
| Python packages | see `scripts/setup_venvs.sh` | | torch 2.14.1, transformers 5.19.0, iree-turbine 3.9.0, IREE 3.11.0 / 3.12.0, mako |

Why two IREE versions: IREE 3.12 rewrote its Vulkan HAL and requires Vulkan 1.3 +
synchronization2. The stock Adreno 710 driver is Vulkan 1.1, so it needs 3.11; with Mesa Turnip
(Vulkan 1.4) 3.12 works. Compiler and runtime versions must match.

Undo: `rm -rf third_party build models .venv .venv-iree312`, `adb shell rm -rf /data/local/tmp/{iree,iree312,turnip}`,
and `brew uninstall` what you added.
