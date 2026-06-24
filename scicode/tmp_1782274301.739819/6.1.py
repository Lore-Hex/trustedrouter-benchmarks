import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift



# Background: In Fourier Optics, spatial filtering operates in the frequency domain to selectively 
# preserve or remove certain frequency components of an image. A low-pass filter allows low frequencies 
# (representing smooth, broad features) to pass while attenuating high frequencies (representing fine 
# details, noise, and artifacts). The filter is implemented by:
# 1. Converting the image to the frequency domain using 2D Fast Fourier Transform (FFT)
# 2. Creating a circular mask in frequency space centered at the origin with a radius defined by the 
#    frequency threshold - frequencies within this radius are preserved (value=1), while frequencies 
#    outside are blocked (value=0)
# 3. Multiplying the frequency-domain image by this mask to suppress high-frequency components
# 4. Converting back to the spatial domain using Inverse FFT to obtain the filtered image
# This approach simulates how a spatial filter (like a pinhole) in real laser systems blocks diffracted 
# orders and passes only the central maximum, effectively removing noise and unwanted diffraction patterns.

def apply_low_pass_filter(image_array, frequency_threshold):
    '''Applies a low-pass filter to the given image array based on the frequency threshold.
    Input:
    image_array: 2D numpy array of float, the input image.
    frequency_threshold: float, the radius within which frequencies are preserved.
    Ouput:
    T: 2D numpy array of float, The spatial filter used.
    output_image: 2D numpy array of float, the filtered image in the original domain.
    '''
    
    # Get image dimensions
    rows, cols = image_array.shape
    
    # Compute 2D FFT of the input image
    fft_image = fft2(image_array)
    
    # Shift zero frequency component to center for easier mask creation
    fft_shifted = fftshift(fft_image)
    
    # Create coordinate grids for frequency space (centered at origin)
    # Range from -rows/2 to rows/2 and -cols/2 to cols/2
    u = np.arange(-rows // 2, rows // 2)
    v = np.arange(-cols // 2, cols // 2)
    U, V = np.meshgrid(v, u)
    
    # Calculate distance from center (origin) in frequency space
    # Distance represents the frequency magnitude
    D = np.sqrt(U**2 + V**2)
    
    # Create the low-pass filter mask
    # 1 for frequencies with distance < frequency_threshold (threshold not included)
    # 0 for frequencies with distance >= frequency_threshold
    T = (D < frequency_threshold).astype(float)
    
    # Apply the filter mask in frequency domain
    fft_filtered = fft_shifted * T
    
    # Shift zero frequency component back to corner
    fft_filtered = ifftshift(fft_filtered)
    
    # Compute inverse FFT to convert back to spatial domain
    filtered_image = ifft2(fft_filtered)
    
    # Take real part to remove numerical imaginary components
    filtered_image = np.real(filtered_image)
    
    return T, filtered_image


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('6.1', 4)
target = targets[0]

matrix = np.array([[1, 0], [1, 0]])
frequency_threshold = 20
image_array = np.tile(matrix, (100, 100))
assert np.allclose(apply_low_pass_filter(image_array, frequency_threshold), target)
target = targets[1]

matrix = np.array([[1, 0,0,0], [1,0, 0,0]])
frequency_threshold = 50
image_array = np.tile(matrix, (400, 200))
assert np.allclose(apply_low_pass_filter(image_array, frequency_threshold), target)
target = targets[2]

matrix = np.array([[1, 0,1,0], [1,0, 1,0]])
frequency_threshold = 50
image_array = np.tile(matrix, (400, 200))
assert np.allclose(apply_low_pass_filter(image_array, frequency_threshold), target)
target = targets[3]

matrix = np.array([[1, 0,1,0], [1,0, 1,0]])
frequency_threshold1 = 50
image_array = np.tile(matrix, (400, 200))
image_array1 = np.tile(matrix, (400, 200))
T1, filtered_image1 = apply_low_pass_filter(image_array, frequency_threshold1)
matrix = np.array([[1, 0,0,0], [1,0, 0,0]])
frequency_threshold2 = 40
image_array2 = np.tile(matrix, (400, 200))
T2, filtered_image2 = apply_low_pass_filter(image_array, frequency_threshold2)
MaskAreaDiff = T1.sum()-T2.sum()
assert ((3.1415925*(frequency_threshold1**2-frequency_threshold2**2)-MaskAreaDiff)/MaskAreaDiff<0.01) == target
