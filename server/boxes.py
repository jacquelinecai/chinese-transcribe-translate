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
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_dir, "grayscale.jpg"), gray)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    cv2.imwrite(os.path.join(output_dir, "enhanced.jpg"), enhanced)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY_INV, 21, 1)
    cv2.imwrite(os.path.join(output_dir, "threshold.jpg"), thresh)
    
    # Detect grid lines
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
    
    # Find the main grid
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
    
    # Define grid dimensions
    num_rows = 14
    num_cols = 10
    
    cell_height = h // num_rows
    cell_width = w // num_cols
    
    # Create a copy for visualization
    visualization = main_grid_img.copy()
    
    # Create a separate threshold image for the entire grid for better character detection
    main_grid_gray = cv2.cvtColor(main_grid_img, cv2.COLOR_BGR2GRAY)
    _, main_grid_thresh = cv2.threshold(main_grid_gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.imwrite(os.path.join(output_dir, "main_grid_thresh.jpg"), main_grid_thresh)
    
    cell_count = 0
    for row in range(num_rows):
        for col in range(num_cols):
            cell_x = x + col * cell_width
            cell_y = y + row * cell_height
            
            # Extract the cell from the original image
            cell_img = img[cell_y:cell_y+cell_height, cell_x:cell_x+cell_width].copy()
            
            # Save the original cell
            orig_cell_filename = f"cell_orig_r{row}_c{col}.jpg"
            cv2.imwrite(os.path.join(output_dir, orig_cell_filename), cell_img)
            
            # Convert cell to grayscale for character detection
            cell_gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
            
            # Apply blur to reduce noise
            cell_blur = cv2.GaussianBlur(cell_gray, (5, 5), 0)
            
            # Apply Otsu's thresholding to isolate the character
            _, cell_thresh = cv2.threshold(cell_blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            
            # Save thresholded cell for debugging
            thresh_filename = f"thresh_r{row}_c{col}.jpg"
            cv2.imwrite(os.path.join(output_dir, thresh_filename), cell_thresh)
            
            # Find non-zero pixels (the character)
            non_zero = cv2.findNonZero(cell_thresh)
            
            if non_zero is not None and len(non_zero) > 0:
                # Get the bounding box of all non-zero pixels
                x_c, y_c, w_c, h_c = cv2.boundingRect(non_zero)
                
                # Add padding around the character
                padding = 5
                min_x = max(0, x_c - padding)
                min_y = max(0, y_c - padding)
                max_x = min(cell_img.shape[1], x_c + w_c + padding)
                max_y = min(cell_img.shape[0], y_c + h_c + padding)
                
                # Crop to the character's bounding box
                char_img = cell_img[min_y:max_y, min_x:max_x]
                
                # Draw the detected character boundary on the visualization
                abs_min_x = (col * cell_width) + min_x
                abs_min_y = (row * cell_height) + min_y
                abs_max_x = (col * cell_width) + max_x
                abs_max_y = (row * cell_height) + max_y
                cv2.rectangle(visualization, (abs_min_x, abs_min_y), (abs_max_x, abs_max_y), (0, 0, 255), 2)
                
                # Save the cropped character
                char_filename = f"char_r{row}_c{col}.jpg"
                cv2.imwrite(os.path.join(output_dir, char_filename), char_img)
            else:
                # If no character found, save the original cell
                char_filename = f"char_r{row}_c{col}.jpg"
                cv2.imwrite(os.path.join(output_dir, char_filename), cell_img)
            
            # Draw the cell boundary on the visualization
            start_point = (col * cell_width, row * cell_height)
            end_point = ((col + 1) * cell_width, (row + 1) * cell_height)
            cv2.rectangle(visualization, start_point, end_point, (0, 255, 0), 1)
            
            cell_count += 1
    
    cv2.imwrite(os.path.join(output_dir, "char_boundaries.jpg"), visualization)
    
    print(f"Extracted and saved {cell_count} cells to '{output_dir}' directory")
    print(f"Cropped characters based on their actual boundaries")

def main():
    output_base_dir = "uploaded_extracted_cells"
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
    
    sample_dir = "./uploads"
    image_pattern = os.path.join(sample_dir, "*.png")
    
    image_files = glob.glob(image_pattern)
    
    # If you want to process specific files only, uncomment and adjust this:
    # image_files = ['./data/samples/Prompts 3-1.png', './data/samples/Prompts 3-2.png', './data/samples/Prompts 3-3.png']
    
    if not image_files:
        print(f"No PNG files found in {sample_dir}")
        return
    
    print(f"Found {len(image_files)} PNG files to process")
    
    for image_path in image_files:
        process_image(image_path, output_base_dir)
    
    print("Processing complete!")

if __name__ == "__main__":
    main()