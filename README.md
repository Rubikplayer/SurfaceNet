# SurfaceNet
M. Ji, H. Zheng, J. Gall, Y. Liu, and L. Fang. [SurfaceNet: An End-to-end 3D Neural Network for Multiview Stereopsis](https://www.researchgate.net/publication/318920947_SurfaceNet_An_End-to-end_3D_Neural_Network_for_Multiview_Stereopsis). ICCV, 2017


## Notice
* This is a messy version of the SurfaceNet code, written for myself rather than for others. 
        If you wanna have a taste on the training procedure, you can refer to this code. 
        If you only want to compare with the inference results, please go back to the [main branch](https://github.com/mjiUST/SurfaceNet/tree/github.SurfaceNet).
* So that you (even me) may not fully follow some of the details.
* This temp branch will be cut off in the future after a new clean version appears.
* For the installation please refer to the [main branch](https://github.com/mjiUST/SurfaceNet/tree/github.SurfaceNet).

## Pipeline
* _1.VGG_triplet-train_ folder: train the SimilarityNet
* _2.VGG_triplet-test_ folder: test/visualize the SimilarityNet
* _3.2D_2_3D-train_ folder: train the SurfaceNet
* _4.2D_2_3D-test_ folder: test/visualize the SurfaceNet. You'd better refer to the [main branch](https://github.com/mjiUST/SurfaceNet/tree/github.SurfaceNet) rather than this folder.
