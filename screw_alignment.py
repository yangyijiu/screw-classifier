import os
import pickle
import argparse
from align import *


def save_data(start_positions, filename="rotation_angle.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(start_positions, f)


def process_image(src_folder, file_path, dest_folder, draw_line):
    image = cv2.imread(file_path)
    image_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    # Detect, rotate, and shift the screw for alignment
    theta = detect_and_compute_center_line_1(image_gray)
    image_gray, rotation_angle = rotate_and_center_screw(image_gray, theta)
    center_rho = detect_and_compute_center_line_2(image_gray)
    shift_y1 = shift_y(image_gray, center_rho)
    shift_x1 = shift_x(image_gray)
    result_img = shift_image(image, shift_x1, shift_y1, rotation_angle)
    draw_Auxiliary_Line(result_img, draw_line)

    # Save the alignment image
    relative_dir = os.path.relpath(os.path.dirname(file_path), src_folder)
    result_dir = os.path.join(dest_folder, relative_dir)
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, os.path.basename(file_path))
    cv2.imwrite(result_path, result_img)

    return rotation_angle if "archive\\train\\good" in file_path else None


def main(src_folder, dest_folder, draw_line):
    train_rotation_angle = {}
    for dirpath, _, filenames in os.walk(src_folder):
        for filename in filenames:
            if filename.endswith('.txt'):
                continue
            dirpath = os.path.normpath(dirpath)
            file_path = os.path.join(dirpath, filename)
            print(f"Processing: {file_path}")
            angle = process_image(src_folder, file_path, dest_folder, draw_line)
            if angle is not None:
                train_rotation_angle[file_path] = angle

    save_data(train_rotation_angle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align screws in images.")
    parser.add_argument("--old", default='archive', type=str, help="Path to the original screws folder.")
    parser.add_argument("--align", default='archive-align', type=str, help="Path to the aligned screws folder.")
    parser.add_argument("--draw_line", default=False, action="store_true", help="Flag to draw line or not")

    args = parser.parse_args()
    main(args.old, args.align, args.draw_line)
    print("Screw Alignment successfully!")
