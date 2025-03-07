import cv2
import os
import glob
import numpy as np

def process_image(input_path, output_dir):
    filename = os.path.basename(input_path)
    base_filename = os.path.splitext(filename)[0]
    
    output_dir = os.path.join(output_dir, base_filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Processing {filename}...")
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image {input_path}")
        return

    cv2.imwrite(os.path.join(output_dir, "original.jpg"), img)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_dir, "grayscale.jpg"), gray)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    cv2.imwrite(os.path.join(output_dir, "enhanced.jpg"), enhanced)
    
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY_INV, 21, 1)
    cv2.imwrite(os.path.join(output_dir, "threshold.jpg"), thresh)
    
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 35))
    
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=3)
    cv2.imwrite(os.path.join(output_dir, "horizontal_lines.jpg"), horizontal_lines)
    
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=3)
    cv2.imwrite(os.path.join(output_dir, "vertical_lines.jpg"), vertical_lines)
    
    grid = cv2.add(horizontal_lines, vertical_lines)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    grid = cv2.dilate(grid, kernel, iterations=2)
    cv2.imwrite(os.path.join(output_dir, "grid_lines.jpg"), grid)
    
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("No grid contours found. Using full image as fallback.")
        h, w = img.shape[:2]
        x, y, w, h = 0, 0, w, h
    else:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        main_grid = contours[0]
        x, y, w, h = cv2.boundingRect(main_grid)
    
    bbox_img = img.copy()
    cv2.rectangle(bbox_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.imwrite(os.path.join(output_dir, "bounding_box.jpg"), bbox_img)
    
    main_grid_img = img[y:y+h, x:x+w].copy()
    cv2.imwrite(os.path.join(output_dir, "main_grid.jpg"), main_grid_img)
    
    num_rows = 14
    num_cols = 10
    
    cell_height = h // num_rows
    cell_width = w // num_cols
    
    cell_count = 0
    for row in range(num_rows):
        for col in range(num_cols):
            cell_x = x + col * cell_width
            cell_y = y + row * cell_height
            
            cell_img = img[cell_y:cell_y+cell_height, cell_x:cell_x+cell_width].copy()
            
            cell_filename = f"cell_r{row}_c{col}.jpg"
            cv2.imwrite(os.path.join(output_dir, cell_filename), cell_img)
            
            cell_count += 1
    
    visualization = main_grid_img.copy()
    for row in range(num_rows):
        for col in range(num_cols):
            start_point = (col * cell_width, row * cell_height)
            end_point = ((col + 1) * cell_width, (row + 1) * cell_height)
            cv2.rectangle(visualization, start_point, end_point, (0, 255, 0), 2)
    
    cv2.imwrite(os.path.join(output_dir, "cell_boundaries.jpg"), visualization)
    
    print(f"Extracted and saved {cell_count} cells to '{output_dir}' directory")
    print(f"Also saved main grid, grid lines, and visualization images")

def main():
    output_base_dir = "extracted_cells"
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
    
    sample_dir = "./data/samples"
    image_pattern = os.path.join(sample_dir, "*.png")
    
    image_files = glob.glob(image_pattern)
    image_files = ['./data/samples/Parent2-pg1.png', './data/samples/Parent2-pg2.png']
    print(image_files)
    
    if not image_files:
        print(f"No PNG files found in {sample_dir}")
        return
    
    print(f"Found {len(image_files)} PNG files to process")
    
    for image_path in image_files:
        process_image(image_path, output_base_dir)
    
    print("Processing complete!")

if __name__ == "__main__":
    main()