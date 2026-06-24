import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift



# Background: In Fourier Optics, spatial filtering is performed in the frequency domain to remove unwanted
# spatial components from an image or laser beam. A band-pass filter isolates frequencies within a specific
# annular (ring-shaped) region defined by an inner radius (bandmin) and outer radius (bandmax). 
# The filter creates a binary mask where frequencies strictly between bandmin and bandmax are passed (value=1),
# while all other frequencies are blocked (value=0). This removes both low-frequency (DC/background trends) 
# and high-frequency (noise/artifacts) components, allowing only mid-range spatial frequencies to pass.
# The process involves: (1) transforming the image to frequency domain via 2D FFT, (2) creating an annular
# mask in frequency space, (3) multiplying the frequency spectrum by the mask, and (4) transforming back
# to spatial domain via inverse FFT. The distance from the center in frequency space is computed as the 
# Euclidean distance from the origin, typically calculated using the coordinate meshgrid.

def apply_band_pass_filter(image_array, bandmin, bandmax):
    '''Applies a band pass filter to the given image array based on the frequency threshold.
    Input:
    image_array: 2D numpy array of float, the input image.
    bandmin: float, inner radius of the frequenc band
    bandmax: float, outer radius of the frequenc band
    Ouput:
    T: float,2D numpy array, The spatial filter used.
    output_image: float,2D numpy array, the filtered image in the original domain.
    '''
    
    # Get image dimensions
    rows, cols = image_array.shape
    
    # Compute the 2D FFT of the input image
    freq_domain = fftshift(fft2(image_array))
    
    # Create coordinate grids centered at the origin
    # This represents the frequency space coordinates
    row_coords = np.arange(rows) - rows // 2
    col_coords = np.arange(cols) - cols // 2
    row_mesh, col_mesh = np.meshgrid(col_coords, row_coords)
    
    # Compute the distance from the center (origin) for each point in frequency space
    distance_from_center = np.sqrt(row_mesh**2 + col_mesh**2)
    
    # Create the band-pass filter mask
    # Pass only frequencies strictly between bandmin and bandmax (exclusive bounds)
    T = np.logical_and(distance_from_center > bandmin, distance_from_center < bandmax).astype(float)
    
    # Apply the filter to the frequency domain representation
    filtered_freq_domain = freq_domain * T
    
    # Transform back to spatial domain using inverse FFT
    filtered_image = np.real(ifft2(ifftshift(filtered_freq_domain)))
    
    return T, filtered_image


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('7.1', 4)
target = targets[0]

matrix = np.array([[1, 0,0,0], [0,0, 0,1]])
bandmin,bandmax = 20,50
image_array = np.tile(matrix, (400, 200))
assert np.allclose(apply_band_pass_filter(image_array, bandmin,bandmax), target)
target = targets[1]

matrix = np.array([[1, 0,0,0], [0,0, 0,1]])
bandmin,bandmax = 40,80
image_array = np.tile(matrix, (400, 200))
assert np.allclose(apply_band_pass_filter(image_array, bandmin,bandmax), target)
target = targets[2]

matrix = np.array([[1, 0,0,0], [0,0, 0,1]])
bandmin,bandmax = 80,100
image_array = np.tile(matrix, (400, 200))
assert np.allclose(apply_band_pass_filter(image_array, bandmin,bandmax), target)
target = targets[3]

matrix = np.array([[1, 0,0,0], [0,0, 0,1]])
frequency_threshold = 100
bandmin,bandmax = 40,80
T1, filtered_image1 = apply_band_pass_filter(image_array, bandmin,bandmax)
matrix = np.array([[1, 0,0,0], [0,0, 0,1]])
bandmin,bandmax = 60, 80
image_array = np.tile(matrix, (400, 200))
T2, filtered_image = apply_band_pass_filter(image_array, bandmin,bandmax)
MaskAreaDiff = T1.sum()-T2.sum()
assert (MaskAreaDiff > 0) == target
