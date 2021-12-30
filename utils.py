import os, cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import convolve
from scipy.signal import medfilt2d
from skimage import color
import matplotlib.pyplot as plt
# from numba import jit

def resizeImage(image):
    image = image[0:690, 180:1100]
    image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
    return image

# @jit(nopython=True)
def steerGaussFilterOrder2(image, theta, sigma):
    ## convert to radian angle
    rad_theta = -theta * (np.pi / 180)
    """[summary]
    This function implements the steerable filter
    of the second deriative of Gaussian function
    (X-Y separable version)
    Args:
        image ([ndarray]): [the input image MxNxP]
        theta ([int]): [the orientation]
        sigma ([int]): [the standard deviation of the Gaussian template]

    Return:
        output ([type]): [the response of derivative in the theta direction]
    """
    ###################### determine necessary filter ######################
    Wx = np.floor((8/2)*sigma)
    if Wx < 1:
        Wx = 1
    x = np.arange(-Wx, Wx+1)
    xx, yy = np.meshgrid(x, x)
    g0 = np.exp(-(xx**2 + yy**2) / (2*sigma**2)) / (sigma*np.sqrt(2*np.pi))
    G2a = -g0 / sigma**2 + g0 * xx**2 / pow(sigma, 4)
    G2b = g0 * xx * yy / pow(sigma, 4)
    G2c = -g0 / sigma**2 + g0 * yy**2 / pow(sigma, 4)

    G = pow(np.cos(rad_theta), 2)*G2a \
      + pow(np.sin(rad_theta), 2)*G2c \
      - 2*np.cos(rad_theta)*np.sin(rad_theta)*G2b

    I2a = ndimage.filters.convolve(\
        image, G2a, mode='nearest')
    I2b = ndimage.filters.convolve(\
        image, G2b, mode='nearest')
    I2c = ndimage.filters.convolve(\
        image, G2c, mode='nearest')

    J =   pow(np.cos(rad_theta), 2)*I2a \
        + pow(np.sin(rad_theta), 2)*I2c \
        - 2*np.cos(rad_theta)*np.sin(rad_theta)*I2b

    return J

def getBaseValue(line):
    """
    We already have formula y = ax + b
    or ax - y + b = 0
    extracting a, -1, b
    """
    slope = (line[3] - line[1]) / (line[2] - line[0])
    bias = line[1] - slope * line[0]
    base_value = np.array([slope, -1, bias])
    return base_value

def houghTransform(image):
    ## Normalize image to range 0-255
    normalize_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1).astype(np.uint8)
    ## Getting the edege with Canny transform
    canny_image = cv2.Canny(normalize_image, threshold1=40, threshold2=80)
    ## Setting parameters for Hough Transform
    rho, theta, threshold, min_line_length, max_line_gap = 3, np.pi/180, 40, 20, 30
    ## Extract lines from Hough Transform
    lines = cv2.HoughLinesP(canny_image, 
                            rho,
                            theta,
                            threshold,
                            np.array([]),
                            minLineLength=min_line_length,
                            maxLineGap=max_line_gap)

    len_array = np.zeros((len(lines), 1))
    for i in range(len(lines)):
        point_1 = lines[i][0][:2]
        point_2 = lines[i][0][2:]
        len_array[i] = np.sqrt(np.sum((point_1 - point_2)**2))
    index_max = np.argmax(len_array)
    output_line = lines[index_max][0]

    ## Get the base value of line and 
    base_value = getBaseValue(output_line)
    x_min, x_max = 0, image.shape[1]
    y_min, y_max = 0, image.shape[0]
    x2, y2 = int((y_max - base_value[2]) / base_value[0]), y_max
    x1, y1 = x_min, int(base_value[0] * x_min + base_value[2])
    output_line = x1, y1, x2, y2
    status = 1

    return output_line, status

def getGaussianFilter(shape=(3, 3), sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    H = np.exp(-(pow(x, 2) + pow(y, 2)) / (2.*pow(sigma, 2)))
    H[H < np.finfo(H.dtype).eps*H.max()] = 0
    sumh = H.sum()
    if sumh != 0:
        H /= sumh
    return H

def detectAnomalies(image, sigma_s=4, normalize=False):
    lab = image
    # get size of image
    height, width = lab.shape[:2]
    # get sigma (the standard deviation of Gaussian function)
    sigma = int(np.ceil(min(height, width) / sigma_s))
    # get kernel size
    kernel_size = 3*sigma + 1
    # get Gaussian filter
    gaussian_filter = getGaussianFilter((1, kernel_size), sigma)
    # get specific color channel in LAB space color
    L_channel, A_channel, B_channel = None, None, None
    L_channel = lab[..., 0]
    A_channel = lab[..., 1]
    B_channel = lab[..., 2]
    # apply gaussian filter to each color channel
    gaussian_L, gaussian_A, gaussian_B = None, None, None
    gaussian_L = convolve(L_channel, gaussian_filter, mode='nearest')
    gaussian_A = convolve(A_channel, gaussian_filter, mode='nearest')
    gaussian_B = convolve(B_channel, gaussian_filter, mode='nearest')
    
    # get result from Gaussian filter
    filtered_result = None
    filtered_result = pow(np.subtract(L_channel, gaussian_L), 2) \
                    + pow(np.subtract(A_channel, gaussian_A), 2) \
                    + pow(np.subtract(B_channel, gaussian_B), 2)

    # convert range result to 0-255
    filtered_result = cv2.normalize(filtered_result, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    if normalize==True:
        normalized_result = cv2.normalize(filtered_result, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8UC1)
        return normalized_result
    else:
        return filtered_result

# @jit(nopython=True)
def anomaliesRGBGenertor(image):
    ## convert image to HSV color
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ## detect anomalies in HSV image
    anomalies_HSV = detectAnomalies(hsv_image, sigma_s=4, normalize=True)
    ## convert input image to RGB color
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ## detect anomalies in input image
    anomalies_RGB = detectAnomalies(rgb_image, sigma_s=4, normalize=True)
    ## bitwise and two results: anomalies_HSV & anomalies_RGB
    anomalies_image = cv2.bitwise_or(anomalies_HSV, anomalies_RGB, mask=None)
    ## normalize anomly image to range(0, 1)
    anomalies_image = cv2.normalize(anomalies_image, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8UC1)

    return anomalies_image