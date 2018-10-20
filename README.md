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
