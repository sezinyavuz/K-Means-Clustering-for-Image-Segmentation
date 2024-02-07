# K-Means-Clustering-for-Image-Segmentation
I used K-means clustering algorithm for image segmentation by using pixel level and superpixel representation of an input image.

This code explores various image segmentation techniques using K-Means clustering and superpixels. It demonstrates how to extract different features from images, apply K-Means clustering at both pixel and superpixel levels, and visualize the segmentation results.

#####Dependencies

Python 3.x
NumPy
Matplotlib
scikit-image
scipy
Installation
To install the required dependencies, run the following command in your terminal:

Bash
pip install numpy matplotlib scikit-image scipy


##### Usage


1. Replace the placeholder in the image file path (`r"../{file}.jpg"`) (line238) according to the path of the input files you downloaded from the google drive link I added below.

-Google Drive Links:
https://drive.google.com/drive/folders/1g_EpNe3hHimtJLtpxRKcnpIiXmrHoc5j?usp=sharing


2. Run the code


##### Code Structure

The code is organized into the following functions:

rgb_color_feature: Extracts RGB color features from an image.
rgb_color_location_feature: Extracts RGB color and spatial location features from an image.
kmeans_clustering: Implements the K-Means clustering algorithm.
slic_superpixels: Generates SLIC superpixels for an image.
calculate_superpixel_features: Calculates features for each superpixel.
calculate_superpixel_rgb_mean: Calculates mean RGB values for each superpixel.
calculate_rgb_histogram: Calculates RGB histograms for each superpixel.
cluster_histogram: Clusters superpixels based on their RGB histograms.
superpixel_gabor: Applies Gabor filters to superpixels.
visualize_results: Visualizes segmentation results.
results: Main function that runs the entire image segmentation process.


##### Output
The code will display a figure with the following segmentation results:

Original image
Superpixels
Superpixel RGB means
Superpixel RGB histograms
Superpixel Gabor features
Pixel-level clustering
Pixel-level clustering with spatial location


#### Customization
You can adjust the number of clusters (k) in the results function.
You can experiment with different superpixel segmentation algorithms and feature extraction methods.



#### Author
Sezin Yavuz
