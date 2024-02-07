import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic,mark_boundaries
from skimage.color import rgb2lab
from skimage.filters import  gabor_kernel
from scipy import ndimage as ndi
from scipy.spatial.distance import cdist



def rgb_color_feature(image):
    height, width, channels = image.shape
    pixel_values = image.reshape((-1, channels))
    normalized_pixel_values = pixel_values / 255.0
    return normalized_pixel_values


def rgb_color_location_feature(image):
    height, width, channels = image.shape

    # Extract RGB features
    rgb_feature = image.reshape((-1, 3))
    normalized_rgb_values = rgb_feature / 255.0

    spatial_feature = np.column_stack((rgb_feature, np.repeat(np.arange(height), width), np.tile(np.arange(width), height)))
    spatial_feature_normalized = spatial_feature.astype(np.float64)

    # Normalize spatial features
    spatial_feature_normalized[:, :3] /= 255.0
    spatial_feature_normalized[:, -2:] /= np.array([height - 1, width - 1])

    rgb_location_features = np.column_stack((normalized_rgb_values, spatial_feature_normalized[:, -2:]))

    return np.array(rgb_location_features)


def kmeans_clustering(features, k, max_iter=100, tol=1e-4):

  n_samples, n_features = features.shape

  centroids = features[np.random.choice(n_samples, k, replace=False), :]


  for _ in range(max_iter):
    labels = np.argmin(cdist(features, centroids), axis=1)

    new_centroids = np.zeros((k, n_features))
    for i in range(k):
      cluster_points = features[labels == i]
      if len(cluster_points) > 0:
        new_centroids[i] = np.mean(cluster_points, axis=0)

    if np.linalg.norm(centroids - new_centroids) < tol:
      break

    centroids = new_centroids

  return labels

def calculate_superpixel_features(image, segments):
    num_segments = np.max(segments) + 1
    superpixel_features = np.zeros((num_segments, 3))

    # Calculate average feature values for each superpixel
    for i in range(num_segments):
        mask = (segments == i)
        superpixel_features[i, 0] = np.mean(image[mask, 0])  # L channel
        superpixel_features[i, 1] = np.mean(image[mask, 1])  # A channel
        superpixel_features[i, 2] = np.mean(image[mask, 2])  # B channel

    return superpixel_features

def slic_superpixels(image, n_segments=1000):

    image_lab = rgb2lab(image)
    segments = slic(image_lab, n_segments=n_segments, compactness=10)
    # calculate feature vectors for each superpixel
    superpixel_features = calculate_superpixel_features(image_lab, segments)

    return segments, superpixel_features



def calculate_superpixel_rgb_mean(image, segments):
    num_segments = np.max(segments) + 1
    superpixel_rgb_means = np.zeros((num_segments, 3))

    for i in range(num_segments):
        mask = (segments == i)
        if np.sum(mask) > 0:
            superpixel_rgb_means[i, 0] = np.mean(image[mask, 0])  # R channel
            superpixel_rgb_means[i, 1] = np.mean(image[mask, 1])  # G channel
            superpixel_rgb_means[i, 2] = np.mean(image[mask, 2])  # B channel
        else:
            # If the superpixel has no pixels, set mean RGB values to 0
            superpixel_rgb_means[i, :] = 0.0

    superpixel_labels = kmeans_clustering(superpixel_rgb_means, k=3)
    superpixel_mean_image = np.zeros_like(image)

    for i in range(len(superpixel_rgb_means)):
        superpixel_mean_image[segments == i] = superpixel_rgb_means[superpixel_labels[i]]

    return superpixel_mean_image


def calculate_rgb_histogram(image, segments):

    rgb_histograms = []

    for i in np.unique(segments):
        seg = (segments == i)
        superpixel = image[seg]

        if len(superpixel) > 0:
            r_values = superpixel[:, 0]
            g_values = superpixel[:, 1]
            b_values = superpixel[:, 2]

            r_hist, _ = np.histogram(r_values, bins=256, range=(0, 256), density=True)
            g_hist, _ = np.histogram(g_values, bins=256, range=(0, 256), density=True)
            b_hist, _ = np.histogram(b_values, bins=256, range=(0, 256), density=True)

            rgb_histogram = np.concatenate((r_hist, g_hist, b_hist), axis=None)
            rgb_histograms.append(rgb_histogram)
        else:
            rgb_histograms.append(np.zeros(256 * 3))

    return np.array(rgb_histograms)

def cluster_histogram(image,segments):

    rgb_histograms = calculate_rgb_histogram(image, segments)
    histogram_labels = kmeans_clustering(rgb_histograms, k=3)
    kmean= np.zeros_like(segments)

    for i, j in enumerate(np.unique(segments)):
        seg = (segments == j)
        kmean[seg] = histogram_labels[i]

    return kmean



def superpixel_gabor(image, segments):

    orientations = [0, 45, 90, 180]
    gabor_responses = []
    kernels = []
    frequency = 0.5
    for angle in orientations:
        gabor_kernel_matrix = np.real(gabor_kernel(frequency=frequency, theta=np.deg2rad(angle)))
        kernels.append(gabor_kernel_matrix)

    for kernel in kernels:
        filtered_image = np.zeros_like(image)
        for channel in range(image.shape[2]):
            filtered_image[:, :, channel] = ndi.convolve(image[:, :, channel], kernel, mode='wrap')
        gabor_responses.append(filtered_image)

    mean_response = np.mean(gabor_responses, axis=0)
    gabor_superpixel = np.empty_like(image)

    for seg in np.unique(segments):
        mask = (segments == seg)
        mean_response_segment = np.mean(mean_response[mask], axis=0)
        gabor_superpixel[mask] = mean_response_segment


    return gabor_superpixel



def visualize_results(image, labels, title, ax):
    segmented_image = labels.reshape(image.shape[:-1])
    ax.imshow(segmented_image)
    ax.set_title(title,fontsize=30)
    ax.axis("off")




def results(image_path, k=3):
    image = plt.imread(image_path)

    # Extract features
    pixel_features = rgb_color_feature(image)
    pixel_location_features = rgb_color_location_feature(image)

    segments, superpixel_features = slic_superpixels(image, n_segments=100)
    superpixel_image = mark_boundaries(image, segments)

    # Calculate superpixel RGB means
    superpixel_rgb_means = calculate_superpixel_rgb_mean(image, segments)

    # Perform K-Means clustering for pixel-level features
    pixel_labels = kmeans_clustering(pixel_features, k)
    location_labels = kmeans_clustering(pixel_location_features, k)

    # Calculate RGB Histogram
    histogram_labels = cluster_histogram(image, segments)

    #Calculate Gabor
    gabor_image = superpixel_gabor(image, segments)
    gabor_label = kmeans_clustering(gabor_image.reshape(-1, gabor_image.shape[-1]), k)
    gabor_label_image = gabor_label.reshape(image.shape[:-1])

    fig, axs = plt.subplots(2, 4, figsize=(50, 50))

    # Original Image
    axs[0, 0].imshow(image)
    axs[0, 0].set_title("Original Image",fontsize=30)
    axs[0, 0].axis("off")

    # Superpixel Image
    axs[1, 0].imshow(superpixel_image)
    axs[1, 0].set_title("Superpixel",fontsize=30)
    axs[1, 0].axis("off")

    axs[1, 1].imshow(superpixel_rgb_means, cmap='viridis')  # Use 'viridis' colormap
    axs[1, 1].set_title("Superpixel_rgb_mean",fontsize=30)
    axs[1, 1].axis("off")

    axs[1, 2].imshow(histogram_labels, cmap='viridis')
    axs[1, 2].set_title("Superpixel_rgb_histogram",fontsize=36)
    axs[1, 2].axis("off")

    axs[1, 3].imshow(gabor_label_image, cmap='viridis')
    axs[1, 3].set_title("Superpixel_gabor",fontsize=30)
    axs[1, 3].axis("off")

    # Pixel-Level Clustering
    visualize_results(image, pixel_labels, "Pixel-Level Clustering", axs[0, 1])
    visualize_results(image, location_labels, "Pixel-Level Clustering with Location", axs[0, 2])

    plt.show()

results(r"../{file}.jpg")