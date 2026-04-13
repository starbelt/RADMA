#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Default states
DO_BUILD=true
DO_FLASH=true
DO_DATA=true

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -nb|--no-build) DO_BUILD=false; shift ;;
        -nf|--no-flash) DO_FLASH=false; shift ;;
        -nd|--no-data)  DO_DATA=false; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

if [ "$DO_BUILD" = true ]; then
    echo "=== Calculating Build Jobs ==="
    TOTAL_CORES=$(nproc)
    JOBS=$(( (TOTAL_CORES * 3 + 3) / 4 ))
    echo "Running Make with -j$JOBS"

    REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
    cd src
    
    echo "=== Configuring CMake ==="
    cmake -B out -S . \
        || { echo "ERROR: CMake configuration failed."; exit 1; }

    echo "=== Building Project ==="
    make -C out -j"$JOBS" || { echo "ERROR: Make build failed."; exit 1; }

    echo "=== Staging Flashtool Dependencies ==="
    # MUST point flashtool to 'out' so it can read CMake's target data
    # and properly build the LittleFS filesystem.
    find out/coralmicro_build -type f -name "elf_loader" -exec cp {} out/ \; || echo "Warning: elf_loader not found."
    find out/coralmicro_build -type f -name "flashloader" -exec cp {} out/ \; 2>/dev/null || true
    find out/coralmicro_build -type f -name "mklfs" -exec cp {} out/ \; || echo "Warning: mklfs not found."
else
    echo "=== Skipping Build (No-Build Flag Detected) ==="
fi


if [ "$DO_FLASH" = true ]; then
    if [ "$DO_BUILD" = false ]; then
        cd src
    fi

    echo "=== Flashing to Board ==="
    
    # Base arguments for flashtool
    FLASHTOOL_ARGS="--build_dir out --elf_path out/coralmicro-app"
    
    # Append nodata flag if requested
    if [ "$DO_DATA" = false ]; then
        echo "Note: Skipping data flash (models will not be overwritten)."
        FLASHTOOL_ARGS="$FLASHTOOL_ARGS --nodata"
    fi

    python3 ../coralmicro/scripts/flashtool.py $FLASHTOOL_ARGS || { echo "ERROR: Flashing failed."; exit 1; }
else
    echo "=== Skipping Flash (No-Flash Flag Detected) ==="
fi

echo ""
echo "=== Success! ==="