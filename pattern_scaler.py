from PIL import Image
import math
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch, cm
from reportlab.lib.pagesizes import A4
import os
import tempfile
import numpy as np

class PatternScaler:
    def __init__(self, page_size=A4, margin_cm=1.0, overlap_cm=1.0):
        self.page_size = page_size
        self.margin = margin_cm * cm
        self.overlap = overlap_cm * cm
        self.printable_width = self.page_size[0] - (2 * self.margin)
        self.printable_height = self.page_size[1] - (2 * self.margin)

    def is_empty_image(self, img, threshold=0.99):
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Convert to numpy array for faster processing
        img_array = np.array(img)
        
        # Check for transparency
        if img_array.shape[2] == 4:  # RGBA
            # Count pixels that are fully transparent
            transparent_pixels = (img_array[:, :, 3] == 0).sum()
            total_pixels = img_array.shape[0] * img_array.shape[1]
            
            if transparent_pixels / total_pixels > threshold:
                return True
        
        # Check for white pixels
        white_threshold = 250  # Close to white (255)
        is_white = (img_array[:, :, :3] > white_threshold).all(axis=2)
        white_pixel_ratio = is_white.sum() / (img_array.shape[0] * img_array.shape[1])
        
        return white_pixel_ratio > threshold

    def create_tiled_pdf(self, image_path, real_width_cm, real_height_cm, output_pdf_path):
        try:
            # Create a temporary directory for cropped images
            with tempfile.TemporaryDirectory() as temp_dir:

                with Image.open(image_path) as img:
                    # Convert real dimensions to points
                    real_width_pts = real_width_cm * cm
                    real_height_pts = real_height_cm * cm
                    
                    scale_width = real_width_pts / img.size[0]
                    scale_height = real_height_pts / img.size[1]
                    scale = min(scale_width, scale_height)
                    
                    scaled_width = int(img.size[0] * scale)
                    scaled_height = int(img.size[1] * scale)
                    
                    printable_width_no_margin = self.printable_width - self.overlap
                    printable_height_no_margin = self.printable_height - self.overlap
                    
                    pages_across = math.ceil(scaled_width / printable_width_no_margin)
                    pages_down = math.ceil(scaled_height / printable_height_no_margin)
                    
                    scaled_img = img.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)

                    c = canvas.Canvas(output_pdf_path, pagesize=self.page_size)
                    
                    pages_created = 0
                    page_positions = []  # Store positions of non-empty pages
                    
                    for row in range(pages_down):
                        for col in range(pages_across):
                            left = col * (self.printable_width - self.overlap)
                            top = row * (self.printable_height - self.overlap)
                            right = min(left + self.printable_width, scaled_width)
                            bottom = min(top + self.printable_height, scaled_height)
                            
                            # Ensure coordinates are valid
                            left = max(0, min(left, scaled_width))
                            top = max(0, min(top, scaled_height))
                            right = max(left, min(right, scaled_width))
                            bottom = max(top, min(bottom, scaled_height))
                            
                            # Crop the scaled image
                            crop_box = (int(left), int(top), int(right), int(bottom))
                            cropped = scaled_img.crop(crop_box)
                            
                            # Check if the cropped section contains content
                            if not self.is_empty_image(cropped):
                                page_positions.append((row, col, crop_box, cropped.width, cropped.height))
                    
                    total_pages = len(page_positions)
                    
                    # Second pass: create PDF with only non-empty pages
                    for page_num, (row, col, crop_box, width, height) in enumerate(page_positions, 1):
                        # Get the cropped section
                        cropped = scaled_img.crop(crop_box)
                        
                        # Save temporary cropped image
                        temp_image_path = os.path.join(temp_dir, f'temp_{row}_{col}.png')
                        cropped.save(temp_image_path)
                        
                        # Draw the cropped image section
                        c.drawImage(temp_image_path,
                                  self.margin,
                                  self.page_size[1] - self.margin - height,
                                  width=width,
                                  height=height)
                        
                        # Add page information and assembly guides
                        self._add_page_info(c, page_num, total_pages, row + 1, col + 1)
                        self._add_assembly_guides(c)
                        
                        c.showPage()
                    
                    c.save()
                    
                    print(f"PDF created successfully at {output_pdf_path}")
                    print(f"Pattern dimensions: {real_width_cm:.1f}cm x {real_height_cm:.1f}cm")
                    print(f"Total pages created: {total_pages}")
                
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            raise

    def _add_page_info(self, canvas, page_num, total_pages, row, col):
        canvas.setFont("Helvetica", 10)
        text = f"Page {page_num} of {total_pages} (Row {row}, Column {col})"
        canvas.drawString(self.margin, self.page_size[1] - 20, text)

    def _add_assembly_guides(self, canvas):
        canvas.setDash(6, 3)
        canvas.setStrokeColorRGB(0.7, 0.7, 0.7)
        
        # Draw crop marks at corners
        for x in [self.margin, self.page_size[0] - self.margin]:
            for y in [self.margin, self.page_size[1] - self.margin]:
                canvas.line(x - 10, y, x + 10, y)
                canvas.line(x, y - 10, x, y + 10)