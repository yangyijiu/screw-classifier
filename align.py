import cv2
import numpy as np

def detect_and_compute_center_line_1(image):
    """Detect edges and compute the center line of the screw."""
    image = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(image, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 140)
    avg_theta = np.mean([line[0][1] for line in lines])
    lines = [line for line in lines if abs(line[0][1] - avg_theta) < 2]
    sorted_lines = sorted(lines, key=lambda x: x[0][0])
    line1, line2 = sorted_lines[0], sorted_lines[-1]
    theta = (line1[0][1] + line2[0][1]) / 2
    return theta
def detect_and_compute_center_line_2(image):
    """Detect edges and compute the center line of the screw."""
    image = cv2.GaussianBlur(image, (5, 5), 0)
    modified_image = image.copy()
    modified_image[:, :260] = 220
    modified_image[:, 450:] = 220
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    modified_image = clahe.apply(modified_image)
    edges = cv2.Canny(modified_image, 20, 80)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 60)
    lines = [line for line in lines if (1.55 < line[0][1] < 1.59)]
    lines = [line for line in lines if (300 < line[0][0] < 800)]
    sorted_lines = sorted(lines, key=lambda x: x[0][0])
    line1, line2 = sorted_lines[0], sorted_lines[-1]
    center_rho = (line1[0][0] + line2[0][0]) / 2
    return center_rho

def shift_x(image):
    """Detect the screw head using the brightness change along the detected lines."""
    height, width = image.shape
    center_y = height // 2
    image = cv2.medianBlur(image, 9)
    # Apply global thresholding using Otsu's binarization method
    _, image_2 = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)
    def sample_along_line(image):
        values = []
        coordinates = []
        for x in range(width):
            values.append(image[center_y, x])
            coordinates.append((x, center_y))
        return np.array(values), coordinates
    # Draw the center line on the image
    center_values, center_coords = sample_along_line(image_2)
    cv2.line(image, (0, center_y), (width, center_y), (0, 0, 255), 2)
    # Extract grayscale values along the center line
    center_diff1 = [np.max(center_values[i:i + 4]) - np.min(center_values[i:i + 4])
                    for i in range(len(center_values) - 4 + 1)]
    # Find the positions where the accumulated difference is above the threshold
    center_changes = [i + 4 // 2 for i, val in enumerate(center_diff1) if abs(val) > 30]
    shift_x = 50 - center_changes[0]
    return shift_x

def shift_y(image,original_rho):
    h, w = image.shape
    shift_y = h // 2 - int(original_rho)
    return shift_y

def shift_image(image, shift_x, shift_y, angle):
    """Shifts and rotates the image based on the given parameters."""
    h, w, c = image.shape
    center = (w // 2, h // 2)
    avg_border_color = np.mean([image[0, :], image[-1, :], image[:, 0], image[:, -1]], axis=(0, 1))
    M_rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
    M_rotation[0, 2] += shift_x
    M_rotation[1, 2] += shift_y
    transformed_image = cv2.warpAffine(image, M_rotation, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=tuple(avg_border_color))
    return transformed_image

def draw_Auxiliary_Line(transformed_image, draw_line):
    """Draws the Auxiliary Line on the image."""
    h, w, c = transformed_image.shape
    center = (w // 2, h // 2)
    if draw_line:
        cv2.line(transformed_image, (0, center[1]), (w, center[1]), (255, 0, 0), 2)
        cv2.line(transformed_image, (50, 0), (50, h), (0, 0, 255), 2)
        cv2.line(transformed_image, (960, 0), (960, h), (0, 0, 255), 2)

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Calculate average color of the borders
    avg_border_color = np.mean([image[0, :], image[-1, :], image[:, 0], image[:, -1]])

    # Get the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Rotate the image and fill the border with average border color
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=avg_border_color)
    return rotated



def rotate_and_center_screw(image, theta):

    """Compute the average gray value of a region in the image."""
    def average_gray_value(image, x_start, x_end, y_start, y_end):
        region = image[y_start:y_end, x_start:x_end]
        return np.mean(region)

    """Rotate the screw to vertical position and center it in the image."""
    rotation_angle = np.degrees(theta) - 90
    # Rotate the image to make the center line horizontal
    rotated_img = rotate_image(image, rotation_angle)
    # Compute average gray values for the two regions
    avg_l = average_gray_value(rotated_img, 100, 200, 350, 650)
    avg_r = average_gray_value(rotated_img, 800, 900, 350, 650)
    if avg_l > avg_r:
        rotated_img = rotate_image(rotated_img, 180)
        rotation_angle += 180
    return rotated_img, rotation_angle
