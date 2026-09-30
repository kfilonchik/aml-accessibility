# aml-accessibility

## 1. Install Conda without sudo

For a Linux machine, **Miniforge** is a convenient option because it installs into the user home directory and uses `conda-forge`.

Download the installer:

```bash
cd ~

wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
```

Run it:

```bash
bash Miniforge3-Linux-x86_64.sh
```

Use the default installation path, normally:

```text
~/miniforge3
```

Allow the installer to initialize Conda when prompted.

After installation, reload the shell configuration:

```bash
source ~/.bashrc
```

Verify the installation:

```bash
conda --version
```

Optional:

```bash
conda info
```

---

## 2. Create the r5py environment

Create a dedicated Conda environment with Python 3.11:

```bash
conda create -n r5geo python=3.11 -y
```

Activate it:

```bash
conda activate r5geo
```

Verify Python:

```bash
python --version
```

---

## 3. Install Java and geospatial dependencies

Install OpenJDK 21, JPype, and the geospatial/scientific Python stack from `conda-forge`:

```bash
conda install -c conda-forge openjdk=21 r5py jpype1 \
  geopandas pyproj proj proj-data shapely fiona gdal \
  pandas numpy pyarrow scikit-learn -y
```

This installs:

- `openjdk=21` — Java runtime required by R5/r5py
- `r5py` — Python interface to the R5 routing engine
- `jpype1` — Python-to-Java bridge
- `geopandas` — geospatial DataFrame support
- `pyproj`, `proj`, `proj-data` — coordinate reference system transformations
- `shapely` — geometric operations
- `fiona`, `gdal` — geospatial file access
- `pandas`, `numpy` — general data processing
- `pyarrow` — Parquet/Arrow support
- `scikit-learn` — clustering and other analysis utilities

---

## 4. Verify Java

Check the Java installation inside the environment:

```bash
java -version
```

Check where Java is coming from:

```bash
which java
```

It should normally point somewhere inside the Conda environment, for example:

```text
~/miniforge3/envs/r5geo/bin/java
```

You can also inspect `JAVA_HOME`:

```bash
echo $JAVA_HOME
```

If required, derive it from the Conda environment:

```bash
export JAVA_HOME="$CONDA_PREFIX"
```

---

## 5. Verify r5py

Start Python:

```bash
python
```

Then test the import:

```python
import r5py

print(r5py.__version__)
```

You can also test the main geospatial packages:

```python
import geopandas as gpd
import pandas as pd
import numpy as np
import shapely
import pyproj
import jpype

print("Environment OK")
```

Exit Python with:

```python
exit()
```

---

## 6. Check installed packages

List the packages installed in the environment:

```bash
conda list
```

To inspect only some relevant packages:

```bash
conda list | grep -E "r5py|openjdk|jpype|geopandas|gdal|pyproj"
```

---

## 7. Reusing the environment

Whenever you log in to the Linux machine, activate the environment before running the routing scripts:

```bash
conda activate r5geo
```

To leave the environment:

```bash
conda deactivate
```

---

## 8. Export the environment

To make the setup reproducible for any other person:

```bash
conda env export --no-builds > r5geo_environment.yml
```

A colleague can then recreate it with:

```bash
conda env create -f r5geo_environment.yml
```

and activate it with:

```bash
conda activate r5geo
```

For a cleaner, more portable dependency file, you can also export only explicitly requested packages:

```bash
conda env export --from-history > r5geo_environment_minimal.yml
```

---

## 9. How to run the webmap?

```bash
cd "/NVMESTORAGE/DATA/aml_project/webmap" && python3 -m http.server 8000
```

http://localhost:8000/aml_transit_atlas.html
 