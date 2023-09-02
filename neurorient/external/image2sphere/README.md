## Image to Sphere: Learning Equivariant Features for Efficient Pose Prediction
## Taken from commit: 5facf15af962e4b73fabbb79043337d8be703780
## Please refer to the original work
[Paper](https://openreview.net/forum?id=_2bDpAtr7PI) | [Project Page](https://dmklee.github.io/image2sphere/)

---------------------------------------------------------------------
![I2S model](assets/figure1.png)
This repository implements a hybrid equivariant model for SO(3) reasoning from 2D images for object pose estimation.
The underlying SO(3) symmetry of the pose estimation task is not accessible in an image, which can only be transformed
by in-plane rotations.  Our model, I2S, projects features from the image plane onto the sphere, which is SO(3) transformable.  Thus,
the model is able to leverage SO(3)-equivariant group convolutions which improve sample efficiency.  Conveniently,
the output of the group convolution are coefficients over the Fourier basis of SO(3), which form a concise yet expressive
representation for distributions over SO(3).  Our model can capture complex pose distributions that arise from occlusions, 
ambiguity or object symmetries.