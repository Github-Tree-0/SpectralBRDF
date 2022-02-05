# Spectral BRDF Render
Render image observation with non-Lambertian spectral reflectance with direct illumination (w/o cast shadow).
- RGB_BRDF: The BRDF dimension on the wavelength is 3.
- Spectral BRDF: The BRDF dimension on the wavelength is not limited to 3.

## What you need to prepare?
1. Surface normal map (In folder `supp_info`)
2. Lighting direction (In folder `supp_info`)
3. Spectral BRDF database (In folder `brdf_data`)


## Requirement
- ```pip install pybind11```
- Download [pybind11](https://drive.google.com/file/d/1qsYH3b55t-YpC72WQJkcL4EPSA_vfDZF/view?usp=sharing) and unzip it to folder `RGB_BRDF` and `Spectral_BRDF`

## How to use
0. ```python download_bsdf.py --obj_file supp_info/obj_isotropic.txt --out_dir brdf_data/isotropic```
1. ```python download_bsdf.py --obj_file supp_info/obj_anisotropic.txt --out_dir brdf_data/anisotropic```
2. ```pip install RGB_BRDF/setup.py```
3. ```pip install Spectral_BRDF/setup.py```
4. ```sh demo.bat```
