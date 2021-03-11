# GF_DAE
Implementation of the paper "Guided Filter Regularization for Improved Disentanglement of Shape and Appearance in Diffeomorphic Autoencoders"

Paper: https://openreview.net/pdf?id=ILEMHPV_Lc2

# Data
The data used here are the T1 and T2 IXI images from: https://brain-development.org/ixi-dataset/
We preregister the 3D images in an affine manner and extract slice 77 and save the 2D images with the naming convention:
"IXI*-$hospital$-*$modality$_affine.png" where $hospital$ is one of [Guys, IOP, HH] and $modality$ is one of [T1, T2]. 

If you use another naming convention you will need to change the class IXI_Dataset_Grayvalues() in data.py.

Examples: 
![IXI432-Guys-0987-T1_affine](https://user-images.githubusercontent.com/52460031/110769916-fdb28480-8258-11eb-8703-1566ef369b83.png)
![IXI431-Guys-0986-T2_affine](https://user-images.githubusercontent.com/52460031/110769917-fdb28480-8258-11eb-9de5-4f7655d0eed6.png)
![IXI441-HH-2154-T1_affine](https://user-images.githubusercontent.com/52460031/110769908-fd19ee00-8258-11eb-84e3-7fd9a7b0e18c.png)
![IXI440-HH-2127-T2_affine](https://user-images.githubusercontent.com/52460031/110769912-fdb28480-8258-11eb-853c-e8f8cb6cbc0c.png)

# Usage
Use python run.py --data_path /path/to/your/images --out_path /path/to/your/out/dir
