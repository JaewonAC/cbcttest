# Jaewon's CBCT test
this project is to test reading and processing dicom data from dental CBCT images.

### Requirement
listed items should be install as ordered
1. Python 3.6
not sure about numpy and scipy, etc. things, since I'm in vitual environment of py36, under anaconda 3.7 x64. like:
```
conda create -n py36 python=3.6
conda activate py36
```
2. Openjpeg
```
conda install -c conda-forge openjpeg
```
3. gdcm
```
conda install -c conda-forge gdcm
```
4. pydicom
```
conda install -c conda-forge pydicom
```

### Facts
1. current library setting can read dicom file that is compressed to jpeg 2000 lossless.
2. files given to me were know to be 12-it but some were 16-bit.
