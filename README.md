# To Clone with requisite submodules for use on Google Coral Micro:

```git clone --recursive https://github.com/starbelt/Coral-TPU-Characterization.git```

Follow the [Google Coral Micro Set-up].(https://coral.ai/docs/dev-board-micro/get-started/#1-gather-requirements) documentation to find requirements
*Note:* the requirements for hidapi listed under `coralmicro/scripts/requirements.txt` may be out of date and deprecated on newer python versions
consider changing it to `hidapi==0.14.0` once cloned prior to running `setup.sh`
