# To Clone with requisite submodules for use on Google Coral Micro:

*Linux-based or MacOS machine required*

```git clone --recursive https://github.com/starbelt/Coral-TPU-Characterization.git```

Follow the [Google Coral Micro Set-up](https://coral.ai/docs/dev-board-micro/get-started/#1-gather-requirements) documentation to build your requirements

*Note:* the requirements for hidapi listed under `coralmicro/scripts/requirements.txt` may be out of date and deprecated on newer python versions
consider changing it to `hidapi==0.14.0` once cloned prior to running `setup.sh`

# #Building

To initialize the out folder upon first clone or when changing the data source:

```cmake -B out -S .```

To build prior to flashing:

```make -C out -j$(nproc)```

simply re-run the make command above any time you make changes to the executable, and flash with the command

```python3 coralmicro/scripts/flashtool.py --build_dir out --elf_path out/coralmicro-app```

if re-flashing something with a model attached, you can add the `--nodata` flag
