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


![screenshot](https://github.com/hyeonsangjeon/disparity-based-space-vagriant-image-deblurring/blob/master/readme_pic/.png?raw=true)
![screenshot](https://github.com/hyeonsangjeon/disparity-based-space-vagriant-image-deblurring/blob/master/readme_pic/.png?raw=true)
![screenshot](https://github.com/hyeonsangjeon/disparity-based-space-vagriant-image-deblurring/blob/master/readme_pic/.png?raw=true)
![screenshot](https://github.com/hyeonsangjeon/disparity-based-space-vagriant-image-deblurring/blob/master/readme_pic/.png?raw=true)
