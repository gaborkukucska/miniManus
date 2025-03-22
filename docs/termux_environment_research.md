# Termux Environment Research

## Filesystem Differences

1. **Non-FHS Compliant**
   - Termux does not follow the Filesystem Hierarchy Standard (FHS) unlike most Linux distributions
   - Standard directories like `/bin`, `/etc`, `/usr`, `/tmp` are not at usual locations
   - All programs must be patched and recompiled to meet Termux environment requirements
   - Scripts with standard shebangs (e.g., `#!/bin/sh`) need modification using `termux-fix-shebang`
   - The `termux-exec` package allows usage of standard shebangs

2. **File System Storage**
   - Root file system and user home directory are in a private application data directory on `/data` partition
   - Paths are exposed as `$PREFIX` and `$HOME` respectively
   - Cannot move `$PREFIX` to another location as programs expect it to remain unchanged
   - Cannot have binaries, symlinks, and other files from `$PREFIX` on sdcard (lacks unix permissions, symlinks, etc.)
   - Uninstalling Termux or wiping its data will erase `$PREFIX` and `$HOME` directories

## Bionic libc Usage

1. **Android Native Library**
   - Termux uses Bionic libc (Android's native C library) instead of glibc
   - Packages are compiled with Android NDK for best compatibility
   - Binaries are linked against Bionic libc (files libc.so, libm.so, libdl.so from /system/lib or /system/lib64)
   - Native packages from Linux distributions cannot be executed directly due to libc ABI mismatch

2. **Single-User Environment**
   - Termux runs with the same Linux user ID as the Termux application itself
   - All packages are patched to drop multiuser, setuid/setgid functionality
   - Default ports for server packages are changed (e.g., sshd uses port 8022 instead of 22)

## Python Package Compatibility

1. **Package Management**
   - Python 3.x can be installed via `pkg install python`
   - `pip` package manager is available after installation
   - Upgrading major/minor Python versions makes installed modules unusable
   - Recommended to have `build-essential` package installed for modules that compile native extensions

2. **Pre-packaged Python Libraries**
   - Some Python packages are available directly from Termux's package manager:
     - numpy: `pkg install python-numpy`
     - opencv: `pkg install opencv-python`
     - matplotlib: `pkg install matplotlib`
     - cryptography: `pkg install python-cryptography`

3. **Module Installation Challenges**
   - Some modules require special handling or environment variables:
     - pandas: requires `export CFLAGS="-Wno-deprecated-declarations -Wno-unreachable-code"`
     - pillow: 64-bit devices require `export LDFLAGS="-L/system/lib64"` before pip command
     - pyzmq: may need explicit libzmq path specification
   - Some modules may not be installable without patching and need to be installed from source
   - `termux-exec` must be working correctly to avoid shebang issues

4. **Configuration Issues**
   - Issues with platformdirs package affecting pip, virtualenv, and other tools
   - Configuration files may not be read from standard XDG directories
   - May require manual patching of platformdirs

## Resource Limitations

1. **Memory Constraints**
   - Safe memory usage by any application on Android is no more than 50% of available memory
   - Beyond 50% memory usage, probability of triggering OOM (Out Of Memory) killer drastically increases
   - Android system may forcefully terminate processes due to low memory (signal 9)

2. **CPU Limitations**
   - Android devices prioritize system stability and multitasking over processing large amounts of data
   - CPU usage reaching 100% may trigger system warnings or process termination
   - Android uses cgroups to control how much resources a process can use

3. **Process Management**
   - Android may terminate background processes to free resources
   - "Phantom Process Killer" may terminate processes deemed excessive resource users
   - Battery optimization may affect long-running processes

## Implications for miniManus

1. **Memory Efficiency**
   - Must design for minimal memory footprint
   - Implement progressive loading and unloading of resources
   - Monitor memory usage and implement safeguards before reaching critical thresholds

2. **CPU Usage**
   - Design for efficient processing with minimal CPU spikes
   - Implement background processing in small chunks rather than continuous high CPU usage
   - Consider using worker threads with controlled resource usage

3. **Package Dependencies**
   - Prefer pre-packaged Python libraries from Termux repository when available
   - Document specific installation instructions for required packages
   - Include fallback mechanisms for API providers to ensure functionality

4. **File System Considerations**
   - Respect Termux's file system structure and use `$PREFIX` and `$HOME` appropriately
   - Implement proper backup mechanisms for user data
   - Handle path differences between Termux and standard Linux environments
