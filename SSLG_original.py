import os, cv2, timeit
from utils import *
import matplotlib.pyplot as plt

### Setting folder and image path
data_path = "datasets"
image_name = "example04.png"

print(f"[INFO]...Processing image [{image_name}]")
tic = timeit.default_timer()
### Reading RGB Image
rgb_image = cv2.imread(os.path.join(data_path, "rgb", image_name))
plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
plt.show()

### Resizing RGB Image to dimension 480x640
rgb_image = resizeImage(rgb_image)
# plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
# plt.show()

### Reading the Depth Image
depth_image = cv2.imread(os.path.join(data_path, "depth_u16", image_name), -1)

norm_depth_image = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
norm_color = cv2.applyColorMap(norm_depth_image, cv2.COLORMAP_JET)
plt.imshow(cv2.cvtColor(norm_color, cv2.COLOR_BGR2RGB))
plt.show()

# plt.imshow(depth_image, cmap='gray')
# plt.show()

### Define the Intel RealSense Parameters which using for computing disparity
BASELINE = 55.0871
FOCAL_LENGTH = 1367.6650

### Filtering Object which are far than 10m
depth_image[depth_image > 10000] = 0

### Computing Disparity Map
disparity_map = np.zeros(depth_image.shape)
positive_mask = depth_image != 0
negative_mask = depth_image == 0
disparity_map[positive_mask] = np.around(FOCAL_LENGTH * BASELINE / depth_image[positive_mask]).astype('int')
disparity_map[negative_mask] = np.nan
# resize disparity map
disparity_map = resizeImage(disparity_map)
disparity_map = np.around(disparity_map).astype('int')
# disparity_map[np.isinf(disparity_map)] = 0
plt.imshow(disparity_map, cmap='gray')
plt.show()

### Computing the V-Disparity Map
height, width = disparity_map.shape[0], np.nanmax(disparity_map) + 1
print(height, width)

v_disparity = np.zeros((height, width))
for i in range(height):
    for j in range(width):
        v_disparity[i, j] = len(np.argwhere(disparity_map[i, :] == j))
v_disparity[v_disparity < 5] = 0
# v_disparity = medfilt2d(v_disparity, 5)
plt.imshow(v_disparity, cmap='gray')
plt.show()

### Apply Steerable Gaussian Filter Order
theta = [0, 45, 90]
v_disparity_steerable = np.zeros((v_disparity.shape[0], v_disparity.shape[1], 3))
for i, angle in enumerate(theta):
    v_disparity_steerable[:,:,i] = steerGaussFilterOrder2(v_disparity, angle, 3)
plt.imshow(v_disparity_steerable, cmap='gray')
plt.show()

### Based on the level of difference of the highest and the lowest gradients
### we can filter the straight lines
v_disparity_steerable_diff = np.zeros(v_disparity.shape)
for i in range(v_disparity_steerable.shape[0]):
    for j in range(v_disparity_steerable.shape[1]):
        v_disparity_steerable_diff[i, j] = np.max(v_disparity_steerable[i, j, :]) - np.min(v_disparity_steerable[i, j, :])

plt.imshow(v_disparity_steerable_diff, cmap='gray')
plt.show()

v_disparity_filter = np.zeros(v_disparity.shape)
threshold = 30
v_disparity_filter[v_disparity_steerable_diff >= threshold] = 1
plt.imshow(v_disparity_filter, cmap='gray')
plt.show()

straight_line, status = houghTransform(v_disparity_filter)
x1, y1, x2, y2 = straight_line
drivable_initial = np.zeros(disparity_map.shape)
drivable_threshold = 5
for i in range(y1, y2):
    d = (x2 - x1)/(y2 - y1)*i + (x1*y2 - x2*y1)/(y2 - y1)
    for j in range(drivable_initial.shape[1]):
        if (disparity_map[i, j] > d - drivable_threshold) and (disparity_map[i, j] < d + drivable_threshold):
            drivable_initial[i, j] = 1
plt.imshow(drivable_initial, cmap='gray')
plt.show()

### Extracting Anomalies
drivable_initial = np.uint8(drivable_initial)
_, drivable_binary = cv2.threshold(drivable_initial, 0, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(drivable_binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

contours_image = np.zeros((drivable_binary.shape[0], drivable_binary.shape[1], 3))
# filtering contours by areas
filter_contours = [contours[i] for i in range(len(contours)) if 500 <= cv2.contourArea(contours[i]) <= 50000]
cv2.drawContours(contours_image, filter_contours, -1, (0, 255, 0), 1)
plt.imshow(contours_image, cmap='gray')
plt.show()

# fill holes for contours
depth_anomalies = cv2.cvtColor(np.float32(contours_image), cv2.COLOR_BGR2GRAY)
depth_anomalies[depth_anomalies == 0.0] = 0
depth_anomalies[depth_anomalies != 0.0] = 1
depth_anomalies = ndimage.binary_fill_holes(depth_anomalies).astype(np.float32)
plt.imshow(depth_anomalies, cmap='gray')
plt.show()

### Extract Drivable Area Ignore Anomalies
depth_anomalies = cv2.normalize(depth_anomalies, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
drivable_initial = cv2.normalize(drivable_initial, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
drivable_area = cv2.bitwise_or(depth_anomalies, drivable_initial, mask=None)
drivable_area = medfilt2d(drivable_area, 5)
drivable_area = cv2.normalize(drivable_area, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_16UC1)
plt.imshow(drivable_area, cmap='gray')
plt.show()

### Extract anomalies in RGB Image
# rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2LAB).astype('float')
rgb_anomalies = detectAnomalies(rgb_image, sigma_s=4, normalize=True)
rgb_anomalies = medfilt2d(rgb_anomalies, 5)
plt.imshow(rgb_anomalies, cmap='gray')
plt.show()

### Combine Anomalies in RGB Image and in Depth Image
depth_anomalies = cv2.normalize(depth_anomalies, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8UC1)
final_anomalies = cv2.bitwise_or(rgb_anomalies, depth_anomalies)
final_anomalies = cv2.normalize(final_anomalies, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8UC1)
final_anomalies = medfilt2d(final_anomalies, 5)
plt.imshow(final_anomalies, cmap='gray')
plt.show()

### Save generated label
label = np.zeros(disparity_map.shape, dtype=np.uint8)
label[drivable_area == 1] = 1
label[final_anomalies == 1] = 2
toc = timeit.default_timer()
print(f"\tProssing Time: \t", toc - tic)
# plt.imshow(label)
# plt.show()

### Colorize Label Image And Saving
# RED (255, 0, 0), GREEN (0, 255, 0), BLUE (0, 0, 255)
label_color = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
colorized_label = rgb_image.copy()
for i in range(label_color.shape[0]):
    channel = colorized_label[:, :, i]
    channel[label == 0] = label_color[2, i]
    channel[label == 1] = label_color[1, i]
    channel[label == 2] = label_color[0, i]
    colorized_label[..., i] = channel

saving_path = os.path.join("datasets", "color_label", image_name)
cv2.imwrite(saving_path, cv2.cvtColor(colorized_label.astype(np.uint8), cv2.COLOR_RGB2BGR))

plt.imshow(colorized_label)
plt.title("Color Label Image")
plt.axis('off')
plt.show()