import pickle
import matplotlib.pyplot as plt
from align import *
import os

def load_data(filename="rotation_angle.pkl"):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError):
        return None


def classify_angle(angle):
    return (angle // 90) % 4


def compute_image_difference(result_img, img2_path):
    img2_path = img2_path.replace("archive", "archive-align")
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    diff = cv2.absdiff(result_img[:,:,0], img2)
    _, image_diff = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
    return np.sum(image_diff)


def find_closest_image(current_img_path, not_good_angle, angle_buckets, result_img):
    bucket = classify_angle(not_good_angle)
    candidates = [img for img in angle_buckets[bucket] if img != current_img_path]
    diffs = [(candidate, compute_image_difference(result_img, candidate)) for candidate in candidates]
    sorted_candidates = sorted(diffs, key=lambda x: x[1])
    return sorted_candidates[0], bucket


def apply_heatmap_to_image(img, heatmap, alpha=0.5):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    beta = 1 - alpha
    return cv2.addWeighted(img, beta, heatmap, alpha, 0)


def subtract_closest_good_image(angle_buckets):
    enhance_folder = 'archive-enhance\\'
    src_folder = 'archive\\'
    if not os.path.exists(enhance_folder):
        os.makedirs(enhance_folder)

    for dirpath, _, filenames in os.walk(src_folder):
        for filename in filenames:
            if filename.endswith('.txt'):
                continue

            image_path = os.path.join(dirpath, filename)
            print(f"Processing: {image_path}")

            image, image_gray = cv2.imread(image_path), cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            theta = detect_and_compute_center_line_1(image_gray)
            image_gray, rotation_angle = rotate_and_center_screw(image_gray, theta)

            result_img = shift_image(image, shift_x(image_gray), shift_y(image_gray, detect_and_compute_center_line_2(image_gray)), rotation_angle)
            closest_good_image_path, _ = find_closest_image(image_path, rotation_angle, angle_buckets, result_img)

            closest_good_image_path = closest_good_image_path[0].replace("archive", "archive-align")
            good_image = cv2.imread(closest_good_image_path, 0)

            difference = cv2.absdiff(good_image, cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY))
            difference = cv2.convertScaleAbs(difference, alpha=3.5, beta=0)

            heatmap = cv2.cvtColor((plt.get_cmap('inferno')(difference) * 255).astype(np.uint8), cv2.COLOR_RGBA2BGR)
            weighted_overlayed = apply_heatmap_to_image(result_img, heatmap, alpha=0.6)

            result_dir = os.path.join(enhance_folder, os.path.relpath(dirpath, src_folder))
            os.makedirs(result_dir, exist_ok=True)
            cv2.imwrite(os.path.join(result_dir, filename), weighted_overlayed)


if __name__ == "__main__":
    train_rotation_angle = load_data()
    angle_buckets = {i: [] for i in range(4)}
    for img_path, (angle) in train_rotation_angle.items():
        angle_buckets[classify_angle(angle)].append(img_path)
    subtract_closest_good_image(angle_buckets)
