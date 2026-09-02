# Void Linux Migration Assessment for Local LLM Inference

## Verdict

**Void Linux can run this local-model setup on this machine, but it will not be a drop-in migration.** Use the **x86_64 glibc edition**, not musl.

The two main migration costs are:

1. Installing and maintaining CUDA on a distribution NVIDIA does not officially validate.
2. Replacing this repository's systemd integration with runit services.

## Machine findings

| Component | Detected | Assessment |
|---|---|---|
| CPU | AMD Ryzen 5 5600X, 6 cores/12 threads, AVX2 | Fully suitable |
| RAM | 46 GiB plus 4 GiB swap | Better than the repository's outdated 16 GiB assumption |
| Primary GPU | NVIDIA GeForce RTX 5060 Ti, 16 GiB, compute capability 12.0 | Supported through NVIDIA open kernel modules |
| Secondary GPU | NVIDIA GeForce RTX 3070, 8 GiB, compute capability 8.6 | Supported by the same modern driver |
| NVIDIA driver | Open kernel module 610.43.03 | Currently operates both cards correctly |
| CUDA | 13.3.1, with `nvcc` under `/opt/cuda` | Must be recreated on Void |
| Storage | 465 GiB root filesystem, 283 GiB free | Enough for the current 103 GiB model directory |
| Secure Boot | Disabled | Avoids NVIDIA module-signing complications |

All three existing `llama-server` builds detect both GPUs:

- `CUDA0`: RTX 3070
- `CUDA1`: RTX 5060 Ti

The presets intentionally put smaller models on `CUDA0` and most larger models on `CUDA1`. Consistent device ordering must therefore be verified after migration.

## Current software relevant to inference

The current Arch installation has:

- `cuda` 13.3.1
- `nvidia-open` 610.43.03
- `nvidia-utils` 610.43.03
- GCC 16 and the CUDA-compatible GCC 15 compatibility compiler
- CMake
- GNU Make
- Git
- wget and curl
- pkg-config
- Linux firmware and the NVIDIA firmware package

Ninja is currently missing, even though every repository build script uses CMake's `-G Ninja` generator. The build scripts would currently fail on Arch until Ninja is reinstalled.

The running service uses:

- `versions/turboquant/src/build/bin/llama-server`
- `model-presets.turboquant.ini`
- port `8080`
- router mode with one dynamically loaded model
- both expected devices: `CUDA0,CUDA1`
- `nvidia-persistenced`
- a bounded CUDA/device readiness check before startup

## Void package and support assessment

### Standard build environment

Void provides the ordinary tools required to build llama.cpp:

- GCC and `base-devel`
- CMake
- Ninja
- Git
- wget/curl
- pkg-config
- Linux headers
- coreutils, including `timeout`

These are not expected to be migration blockers.

### NVIDIA driver

Void's official documentation places modern NVIDIA support in its nonfree repository. Its current guidance assigns Turing and newer cards to the `nvidia` package using open kernel modules. That covers both the Ampere RTX 3070 and Blackwell RTX 5060 Ti.

The expected installation is approximately:

```sh
xbps-install -S void-repo-nonfree
xbps-install -S nvidia
```

Exact package names and versions must be checked immediately before migration because Void is rolling-release and RTX 50 support depends on a sufficiently recent driver.

### CUDA toolkit: principal caveat

NVIDIA's CUDA 13.3 support matrix does **not** list Void Linux as a validated distribution. It validates distributions such as Debian, Ubuntu, Fedora, RHEL, and SUSE.

Consequences:

- Void's packaged NVIDIA driver can be used.
- The CUDA toolkit may require a distribution-independent NVIDIA installation, a locally built restricted Void package, or another manually managed installation method.
- `/opt/cuda`, `nvcc`, cuBLAS, and the CUDA runtime must be available for the repository's current builds.
- If the NVIDIA runfile is used, it should install the **toolkit only** and must not replace Void's packaged GPU driver.
- CUDA 13.3 officially supports GCC only through major version 15. If Void ships GCC 16 as its default, a compatible GCC must be selected for `nvcc`.

This is workable, but it is less integrated than Arch's packaged `cuda` installation.

### glibc versus musl

Use **Void glibc** because the inference setup depends on:

- NVIDIA CUDA user-space libraries
- cuBLAS
- standard CUDA compiler tooling
- glibc-linked C++ software

Void musl would introduce substantial compatibility work and is not appropriate for this host.

### cuDNN

There is no evidence that this repository needs cuDNN. The existing binaries link against CUDA, cuBLAS, and cuBLASLt, but not cuDNN. It should not be included in the minimum migration requirements.

## Repository compatibility findings

### Build configuration

All three forks are built locally with:

```text
CMAKE_CUDA_ARCHITECTURES=86;120
GGML_CUDA=ON
GGML_NATIVE=ON
```

The CUDA architecture selection correctly targets both GPUs:

- `86`: RTX 3070
- `120`: RTX 5060 Ti

The upstream and TurboQuant+ scripts assume CUDA is installed under `/opt/cuda`. Preserving that path on Void minimizes changes.

The buun build script is explicitly Arch-specific because it runs:

```sh
sudo pacman -S --needed --noconfirm git cmake ninja base-devel cuda
```

That installation step must be replaced or separated from the build itself for Void.

### Service management

The repository is currently systemd-specific:

- `llama-server@.service`
- `manage-server.sh`
- `scripts/install-system-service.sh`
- `scripts/test-service-start.sh`
- calls to `systemctl`
- journal-based logging
- dependency on `nvidia-persistenced.service`

Void uses **runit**, so these facilities need runit equivalents, likely including:

```text
/etc/sv/nvidia-persistenced/run
/etc/sv/llama-server/run
/etc/sv/llama-server/log/run
```

The existing CUDA readiness loop is reusable conceptually and remains important because runit does not express systemd-style `After=` and `Wants=` dependency ordering.

The runit setup should:

1. Start `nvidia-persistenced`, unless Void's package already supplies and enables a suitable service.
2. Repeatedly check `nvidia-smi` during cold boot.
3. Check that `llama-server --list-devices` reports both `CUDA0` and `CUDA1`.
4. Execute the router with the current environment values.
5. Log through `svlogd`.
6. Run as `yuriteixeira`, not root.
7. Configure equivalents for the current file-descriptor and memory-lock limits.
8. Restart the server after unexpected exits.

### Presets and paths

The presets contain absolute model paths under:

```text
/home/yuriteixeira/Workspaces/local/services/llama/models
```

Keeping the same username and repository location will avoid rewriting them. Otherwise, all preset model paths and the service environment must be updated.

The current GPU ordering is recorded in the presets. It must not be assumed to remain stable across a new driver and operating-system installation.

## Rebuild requirement

Do not rely on carrying the existing compiled binaries into Void. Rebuild all three forks after migration.

The current binaries link against:

- Arch's glibc and libstdc++
- `/opt/cuda/lib64/libcudart.so.13`
- cuBLAS 13 and cuBLASLt 13
- locally built shared ggml libraries
- CPU-native optimization through `GGML_NATIVE=ON`

The GGUF models and preset files can be retained, but the executables and shared libraries should be rebuilt on Void.

## Recommended migration approach

1. Install **Void x86_64 glibc** on a spare disk or separate partition first.
2. Keep the current Arch installation bootable as a rollback environment.
3. Enable Void's nonfree repository and install its current modern NVIDIA driver.
4. Reboot and confirm both GPUs appear correctly:

   ```sh
   nvidia-smi
   ```

5. Verify the RTX 5060 Ti is using the required open NVIDIA kernel module and both cards can enter persistence mode.
6. Install the CUDA toolkit without replacing Void's packaged driver.
7. Confirm the compiler and runtime installation:

   ```sh
   nvcc --version
   nvidia-smi
   ```

8. Install a CUDA-compatible GCC, CMake, Ninja, Git, wget, curl, pkg-config, and the normal C/C++ build environment.
9. Clone this repository recursively and restore or copy the 103 GiB model directory.
10. Build the upstream fork first using CUDA architectures `86;120`.
11. Verify device discovery:

    ```sh
    ./versions/upstream/src/build/bin/llama-server --list-devices
    ```

12. Confirm that `CUDA0` and `CUDA1` still correspond to the GPUs expected by the presets. Update the presets if their order changed.
13. Run the upstream router manually with `model-presets.upstream.ini` before creating any boot service.
14. Load representative small and large models and monitor memory placement with `nvidia-smi` or `nvtop`.
15. Port `nvidia-persistenced` and `llama-server` startup to runit.
16. Test a complete cold boot and ensure the CUDA readiness checks prevent premature server startup.
17. Rebuild and test the experimental buun and TurboQuant+ forks only after upstream is stable.
18. Retain the Arch installation until multiple cold starts and all important model presets have been validated.

## Acceptance checks

Before considering the migration complete, verify:

- Both NVIDIA GPUs load with the intended driver after a cold boot.
- `nvidia-smi` reports both cards without errors.
- `nvidia-persistenced` remains active.
- CUDA compilation succeeds with the selected host compiler.
- Upstream llama.cpp builds with compute architectures 8.6 and 12.0.
- `llama-server --list-devices` finds both GPUs.
- Preset device assignments still map to the intended physical GPUs.
- The router starts automatically through runit.
- The server is reachable on port 8080.
- Small models load on the RTX 3070 where configured.
- Larger models load on the RTX 5060 Ti where configured.
- 64K and 128K context configurations retain sufficient VRAM headroom.
- Logging, restart behavior, and cold-start timeouts work without systemd.
- Upstream, buun, and TurboQuant+ are rebuilt rather than copied from Arch.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| CUDA does not provide an official Void package path | Use Void glibc and install the toolkit separately while retaining Void's driver |
| Toolkit installation overwrites the packaged NVIDIA driver | Use toolkit-only installation and avoid mixed driver installation methods |
| Void's default GCC is newer than CUDA supports | Install/select GCC 15 or another CUDA-supported compiler |
| RTX 5060 Ti support regresses with a driver update | Verify the current Void driver version before migration and keep a known-good kernel/driver available |
| GPU ordering changes | Check `llama-server --list-devices` and update every preset before serving models |
| Existing binaries fail because of ABI differences | Rebuild all forks on Void |
| runit starts llama-server before CUDA is ready | Retain the bounded `nvidia-smi` and `--list-devices` readiness loop |
| Model paths change | Preserve the username/repository path or update all absolute preset paths |
| Experimental fork fails while upstream works | Establish upstream first and treat buun/TurboQuant+ as separate validation stages |

## Bottom line

**Void glibc is technically capable of hosting this setup.** The hardware and llama.cpp code are not blockers. The deciding tradeoff is whether maintaining CUDA outside NVIDIA's officially supported distribution path and replacing systemd automation with runit is acceptable.

If low-maintenance CUDA updates are the priority, Arch or a supported Debian/Ubuntu installation is safer. If Void and runit are the priority, this setup is viable with repository adaptation and staged validation.

## Sources

- [Void Linux Handbook: NVIDIA](https://docs.voidlinux.org/config/graphical-session/graphics-drivers/nvidia.html)
- [NVIDIA CUDA 13.3 Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [NVIDIA Persistence Daemon documentation](https://docs.nvidia.com/deploy/driver-persistence/persistence-daemon.html)
- [NVIDIA `nvidia-persistenced` source and example init files](https://github.com/NVIDIA/nvidia-persistenced)
