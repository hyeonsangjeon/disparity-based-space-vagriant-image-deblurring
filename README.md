# disparity-based-space-vagriant-image-deblurring

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
1. Blur modeling in camera image
2. Introduction and problem of existing deblurring method using multiple images
- Spatial invariant de-blurring method using multiple images
- Block-based spatial variable debloring method using multiple images
3. Proposed spatially variable deblurring algorithm based on disparity-based Image segmentation using multi-image
- image segmentation method using multiple images
- PSF(Point Spread Function) estimation method by disparity using multiple images
- image restoration method by disparity of space variant using multiple images
4. Algorithm result and comparison with existing method
5. Conclusion
