# Disparity-based-space-vagriant-image-deblurring
![screenshot](https://github.com/hyeonsangjeon/disparity-based-space-vagriant-image-deblurring/blob/master/readme_pic/Abstract.png?raw=true)

## Abstract 
Obtaining a good-quality image requires exposure to light for an appropriate amount of time. If there is camera or object motion
during the exposure time, the image is blurred. To remove the blur, some recent image deblurring methods effectively estimate
a point spread function (PSF) by acquiring a noisy image additionally, and restore a clear latent image with the PSF. Since the
groundtruth PSF varies with the location, a blockwise approach for PSF estimation has been proposed. However, the block to
estimate a PSF is a straightly demarcated rectangle which is generally different from the shape of an actual region where the PSF
can be properly assumed constant. We utilize the fact that a PSF is substantially related to the local disparity between two views.
This paper presents a disparity-based method of space-variant image deblurring which employs disparity information in image
segmentation, and estimates a PSF, and restores a latent image for each region. The segmentation method firstly over-segments a
blurred image into sufficiently many regions based on color, and then merges adjacent regions with similar disparities. Experimental
results show the effectiveness of the proposed method.
- Keywords: Image deblurring, space-variant deblurring, disparity, segmentation, point spread function, deconvolution
## Contents
1. Blur modeling in convolution-based images
2. Introduction and problem of existing deblurring method using multiple images
- &nbsp; Spatial invariant de-blurring method using multiple images
- &nbsp; Block-based spatial variable debloring method using multiple images
3. Proposed spatially variable deblurring algorithm based on disparity-based Image segmentation using multi-image
- &nbsp; image segmentation method using multiple images
- &nbsp; PSF(Point Spread Function) estimation method by disparity using multiple images
- &nbsp; image restoration method by disparity of space variant using multiple images
4. Algorithm result and comparison with existing method
5. Conclusion
---------------------------------------

- citation count : 10 (from ~ 2018.10.22 now)  
### Blur modeling in convolution-based images
The blurred image is modeled as a linear combination of PSF, which is the trajectory of camera shake and sharp image.
![screenshot](https://github.com/hyeonsangjeon/disparity-based-space-vagriant-image-deblurring/blob/master/readme_pic/Blurred_model_of_convolution-based_images.png?raw=true)

Blind deconvolution is illposed problem
* The problem of general image blurring should be estimated by PSF and latent image.
* The problem with image restoration is that there is not enough information compared to the solution of the function.
![screenshot](https://github.com/hyeonsangjeon/disparity-based-space-vagriant-image-deblurring/blob/master/readme_pic/The_necessity_of_a_method_of_deblurring_using_multiple_images_in_a_camera.png?raw=true)

---------------------------------------
### Introduction to existing methods

##### General Method to Eliminate Space Invariant Blur using Multiple Images
Assuming that the edge distribution of the noise image is similar to the edge distribution of the clear image, PSF estimate.

It is assumed that the blurring of the image is independent of the spatial position or the amount of movement of the subject and causes the same blurring.  
![screenshot](https://github.com/hyeonsangjeon/disparity-based-space-vagriant-image-deblurring/blob/master/readme_pic/General_Method_In_space_invariant.png?raw=true)

##### Issue of Space Invariant Deblurring Method
The blurred of a typical real image differs spatially in the amount of blur.
PSF estimated from the whole image can not guarantee a stable result when the image is restored because different spreading degree according to the space is not considered.
![screenshot](https://github.com/hyeonsangjeon/disparity-based-space-vagriant-image-deblurring/blob/master/readme_pic/Issue_of_Space_Invariant_Deblurring.png?raw=true)




#### Block-based Spatial Variability Deblurring Method Using Multiple Images
Assuming that the blurring of the image depends on the spatial location to object.
Segmentation of blurred image by block, PSF estimation and image restoration by each region[2].

![screenshot](https://github.com/hyeonsangjeon/disparity-based-space-vagriant-image-deblurring/blob/master/readme_pic/block_based_deblurring.png?raw=true)

Stable PSF estimation is impossible in the case of a block including a content including a sudden change in the amount of motion of the subject.
The PSF estimation and substitution method considering only the entropy of the neighboring block [2] does not consider the amount of blurring according to the amount of movement of the subject.

![screenshot](https://github.com/hyeonsangjeon/disparity-based-space-vagriant-image-deblurring/blob/master/readme_pic/block_based_deblurring_psf.png?raw=true)

---------------------------------------
### Proposed Image Deblurring Method
Image Segmentation Space Variable Algorithm Based on the Motion Amount of Images.
![screenshot](https://github.com/hyeonsangjeon/disparity-based-space-vagriant-image-deblurring/blob/master/readme_pic/Proposed_deblurring_diagram.png?raw=true)



#### Disparity-Based Image Segmentation
The distances to the moving distance of the acquisition time difference between the blurred image and the noise image are calculated. 

![screenshot](https://github.com/hyeonsangjeon/disparity-based-space-vagriant-image-deblurring/blob/master/readme_pic/Harris_corner_calculattion_disparity.png?raw=true)

The disparity distance result of feature point of blurred image and noise image.
Movement amount differs depending on the position of the subject in the image.
![screenshot](https://github.com/hyeonsangjeon/disparity-based-space-vagriant-image-deblurring/blob/master/readme_pic/Disparity_distance_result.png?raw=true)

Approximate content-based initial segmentation using graph cut method [4].
![screenshot](https://github.com/hyeonsangjeon/disparity-based-space-vagriant-image-deblurring/blob/master/readme_pic/Init_segmentation_graph_cut.png?raw=true)

Then, Assigning the feature points corresponding to the initial partition using the graph cut to each area.
![screenshot](https://github.com/hyeonsangjeon/disparity-based-space-vagriant-image-deblurring/blob/master/readme_pic/graph_cut_feature_point_distribution.png?raw=true)

Compute the median of the disparity distance of minutiae by partition and then substitute the median value of the partition value. 
![screenshot](https://github.com/hyeonsangjeon/disparity-based-space-vagriant-image-deblurring/blob/master/readme_pic/Split_value_replaced_median_disparity_distance.png?raw=true)

Merge division value in error by setting error of substituted disparity distance.
![screenshot](https://github.com/hyeonsangjeon/disparity-based-space-vagriant-image-deblurring/blob/master/readme_pic/Merge_division_value.png?raw=true)

---------------------------------------
### Proposed Regional PSF Estimation 
Estimation of PSF considering the amount of spatial variable spreading by segment depth according to image depth.
* Estimation method used by Tikhonov method [1].
  * PSF estimation uses x and y differential images of the segmented region.
![screenshot](https://github.com/hyeonsangjeon/disparity-based-space-vagriant-image-deblurring/blob/master/readme_pic/PSF_Estimation.png?raw=true)

In the masked partial differential image, the block with the largest absolute value of the edge is scanned and the PSF is estimated.
![screenshot](https://github.com/hyeonsangjeon/disparity-based-space-vagriant-image-deblurring/blob/master/readme_pic/Block_Scanning_PSF_Estimation.png?raw=true)

PSF estimation result of segmented edge image by contents
![screenshot](https://github.com/hyeonsangjeon/disparity-based-space-vagriant-image-deblurring/blob/master/readme_pic/Estimated_PSF.png?raw=true)


Partial images with relatively coarse-grained distributions filter the PSF below a certain threadhold by the distribution of PSF using kurtosis and replace with the PSF of the most similar depth information. 
The general PSF has a very high kurtosis distribution and the kurtosis of PSFs that fail to estimate is relatively low.
* Kurtosis low threshold=20 / high threshold=300
![screenshot](https://github.com/hyeonsangjeon/disparity-based-space-vagriant-image-deblurring/blob/master/readme_pic/Kurtosis_filter.png?raw=true)


 ---------------------------------------

## Image Reconstruction method by disparity area
Merge by partition after reconstruction of each segment image using hyper-Laplacian method.
In this section, we describe image reconstruction using the blurred image
and the estimated regional PSFs. We first restore a latent image of each
segmented region by performing deconvolution of each region of the blurred
image with the corresponding PSF, and then merge all the reconstructed regions
into one (see Fig. 1). For each regional image reconstruction, we use an
FFT-based method of hyper-Laplacian regularization (p(x) ∝ exp (−k|x|^α)
where 0 < α ≤ 1) [14], and we choose L1 regularization (α = 1).
![screenshot](https://github.com/hyeonsangjeon/disparity-based-space-vagriant-image-deblurring/blob/master/readme_pic/Image_Restoration_hyper_laplacian.png?raw=true)

The above FFT-based deblurring is fast, and the use of L1 norm for regularization
preserves strong edges.
Performing deconvolution of an extremely smooth region with the estimated
PSF may cause undesirable artifact without apparent enhancement.
Protecting such smooth regions have been investigated in [32, 33] for image
restoration and sharpening, respectively.


---------------------------------------
## Experimental Results
We have used a DSLR camera (Nikon D7000) and two compact digital cameras (Samsung VLUU ST5000 and Canon Digital IXUS 110 IS) to capture real images.


### Results for artificially generated images
We made an artificial blurred image from a fairly-captured input image of
1365×1024 resolution (with exposure time of 1/50 second, relative aperture
of f/9, and ISO 1600 from Nikon D7000) and spatially-different artificial blurkernels.

![screenshot](https://github.com/hyeonsangjeon/disparity-based-space-vagriant-image-deblurring/blob/master/readme_pic/Artificially_generated_images.png?raw=true)

PSF estimate result(block size 33x33)
![screenshot](https://github.com/hyeonsangjeon/disparity-based-space-vagriant-image-deblurring/blob/master/readme_pic/PSF_estimate_result.png?raw=true)

The proposed method reduces the ringing artifact per spatial and improves the sharpness of the edge compared to the spatial invariant restoration method
![screenshot](https://github.com/hyeonsangjeon/disparity-based-space-vagriant-image-deblurring/blob/master/readme_pic/Image_Restoration_result.png?raw=true)

PSNR for each RGB channel of restored image
![screenshot](https://github.com/hyeonsangjeon/disparity-based-space-vagriant-image-deblurring/blob/master/readme_pic/PSNR.png?raw=true)

Restoration result of patch according to distance
![screenshot](https://github.com/hyeonsangjeon/disparity-based-space-vagriant-image-deblurring/blob/master/readme_pic/Restoration_of_Patch.png?raw=true)


### Results for captured real images
We have captured a pair of images whose resolution is 1365×1024: a
blurred image under long exposure and low ISO and a noisy image under
short exposure and high ISO. They are sequentially captured under a usual
hand tremor in the bracketing mode of the camera. Figure 10 shows the pair
of blurred image and registered version of noisy image. Their exposure times,
f-numbers, and ISO settings are shown in Table 2 (row of three objects).
![screenshot](https://github.com/hyeonsangjeon/disparity-based-space-vagriant-image-deblurring/blob/master/readme_pic/Result_Real_image.png?raw=true)

![screenshot](https://github.com/hyeonsangjeon/disparity-based-space-vagriant-image-deblurring/blob/master/readme_pic/Reult_Restoration.png?raw=true)

![screenshot](https://github.com/hyeonsangjeon/disparity-based-space-vagriant-image-deblurring/blob/master/readme_pic/Reult_Restoration_Compared.png?raw=true)

![screenshot](https://github.com/hyeonsangjeon/disparity-based-space-vagriant-image-deblurring/blob/master/readme_pic/Reult_Restoration_Compared_Patch.png?raw=true)


## Conclusion
We presented a disparity-based deblurring algorithm using a pair of noisy
and blurred images. Our algorithm adequately segments the image into regions
by initial graph-cut over-segmentation based on color, and disparitybased
merging. For each region a PSF is estimated, and a regional latent
image is restored. Finally the restored regional images are merged into a latent image. 
The experimental results of artificial and real sets of blurred and 
noisy images have shown that our algorithm attains better qualities than
the two existing distinguished methods. The proposed method is particularly 
useful for images with high variation of disparity.

* The disparity distance between actual images is different depending on the distance between the camera and the subject or the movement of the internal subject.
* When dividing by disparity, it is possible to divide by the accurate content considering the movement of the object by using the image segmentation and merging method based on the moving amount of the proposed multiple images.
* Possible to estimate the blurring of the image based on the amount of motion using the proposed efficient partition. 
* Compared to the conventional method, it can correct the defective ringing artifacts and obtain clear images by using exact PSF for each region of image. 


## Reference
- `[1].` L. Yuan, J. Sun, L. Quan, and H. Shum, “Image Deblurring with Blurred/Noisy Image Pairs,” ACM Trans. on 	Graphics, vol. 26, no. 3, pp. 1-10, Aug. 2007.
- `[2].` M. Sorel and F. Sroubek, “Space-variant deblurring using one blurred and one underexposed image,” 16th  IEEE 	International Conference on Image Processing, pp. 157-160, 2009.
- `[3].` B. D. Lucas and T. Kanade, “An iterative image registration technique with an application to stereo vision”, in 	Proc. 7t h IJCAI, Vancower, B. B., Canada, pp. 674-679, 1981.
- `[4].` Y.Boykov and V.Kolmogorov, “An experimental comparision of min-cut/max-flow algorithms for 	energy minim- 	ization in vision,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 26, no. 9, pp. 	1124-1137, 2004. 
- `[5].` D. Krishnan and R. Fergus, "Fast image deconvolution using hyper-Laplacian priors,” Neural  Information Proc- 	essing Systems, vol. 22, pp.1-9, 2009.
- ...
- `[14].`D. Krishnan, R. Fergus, Fast image deconvolution using hyper-Laplacian priors, in: Proc. Neural Inf. Process. Syst., pp. 1033–1041.
- ...
- `[32].`C. Wang, Z. Liu, Total variation for image restoration with smooth area protection, J. Signal Process. Syst. 61 (2010) 271–277.
- `[33].`A. Polesel, G. Ramponi, V. Mathews, Image enhancement via adaptive unsharp masking, Image Processing, IEEE Transactions on 9 (2000) 505–510.

