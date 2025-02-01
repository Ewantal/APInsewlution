import cv2
import numpy as np
import warnings
import math

from PIL import Image

from pattern_scaler import PatternScaler
from pattern_piece import PatternPiece
from bust_pattern import BustPattern
from custom_ellipse import CustomEllipseCurve
from custom_line import CustomLine
from custom_polyline import CustomPolyline

class Affichage : 
    def __init__(self, distances, pattern : PatternPiece, name : str):
        self.name = name

        self.distances = distances
        self.pattern : BustPattern = pattern

        try :
            self.width, self.height = self.pattern.get_width_height()
        except :
            self.width, self.height = 600, 800
        
        self.is_initailizing = True

        self.is_saved = False
        self.test_image = np.ones((self.height, self.width, 3), np.uint8) * 255  # 4 canaux (RGBA)


    def print_pattern(self) : 
        cv2.destroyAllWindows()
        # Create a window
        cv2.namedWindow(self.name)

        # Create trackbars for height and width
        for distance_name, distance in self.distances.items() : 
            cv2.createTrackbar(distance_name, self.name, distance, self.width, self.update_image)

        self.is_initailizing = False

        self.print_points_as_dots()
        self.print_links()

    def save_pattern_image(self):

        # In millimiters
        max_height = abs(self.pattern.get_point_y_value("A") - self.pattern.get_point_y_value("F"))
        max_width = abs(self.pattern.get_point_x_value("A") - self.pattern.get_point_x_value("A2"))

        print(f"max_height : {max_height}, max_width : {max_width}")

        if not cv2.imwrite('./results/test_pattern.png', self.test_image):
            raise Exception("Could not write image")

        pdf_path = "./results/test_pdf.pdf"
        
        scaler = PatternScaler()
        scaler.create_tiled_pdf('./results/test_pattern.png', (max_width*1.2)/10, (max_height*1.2)/10, pdf_path)
            

    def reset_image(self) :
        self.test_image = np.ones((self.height, self.width, 3), np.uint8) * 255
    
    # Callback function for the trackbars
    def update_image(self, val):

        if self.is_initailizing:
            self.pattern.create_body_pattern(distances=self.distances)
            return
        
        # Get current positions of the trackbars

        for distance_name, distance in self.distances.items() : 
            self.distances[distance_name] = cv2.getTrackbarPos(distance_name, self.name)
        
        # Redraw the image with updated values
        try : 
            self.reset_image()
            self.pattern.create_body_pattern(distances=self.distances)
            self.print_points_as_dots()
            self.print_links()
        except : 
            print("THIS VALUE IS NOT FAISABLE")


    def print_points_as_dots(self) : 
        print(self.pattern.points)

        # Define text parameters
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (0, 0, 255)  # Red color for text
        font_thickness = 1
        text_offset = (5, 5)  # Offset for text placement relative to point

        for point_name, coordinates in self.pattern.points.items():
            # Convert coordinates to integers for cv2
            x, y = int(coordinates[0]), int(coordinates[1])
            point_pos = (x, y)
            
            # Draw the point as a filled circle
            cv2.circle(self.test_image, point_pos, 3, (0, 255, 0), -1)  # Green dot
            
            # Add the point name next to the dot
            text_pos = (x + text_offset[0], y + text_offset[1])
            cv2.putText(self.test_image, point_name, text_pos, 
                    font, font_scale, font_color, font_thickness)
        
        print(self.pattern.name)
        cv2.putText(self.test_image, self.pattern.name, (int(self.width/6), int(self.height/2)), 
                font, font_scale, font_color, font_thickness)  
        cv2.putText(self.test_image, self.pattern.style, (int(self.width/6), int(self.height/1.8)), 
                font, font_scale, font_color, font_thickness)  
        


    def print_links(self) : 
        for link_name, link in self.pattern.links.items() : 
            if isinstance(link, CustomLine) :
                print("DRAWING LINE")
                self.draw_line(link)
            if isinstance(link, CustomEllipseCurve) :
                print("DRAWING CURVE")
                self.draw_ellipse(link)
            if isinstance(link, CustomPolyline) :
                print("DRAWING POLYLINE")
                self.draw_polyline(link)


    def draw_line(self, line : CustomLine) : 
        cv2.line(self.test_image, self.pattern.points.get(line.start_point), self.pattern.points.get(line.end_point), (255, 0, 0, 255), 3)


    def draw_ellipse(self, ellipse : CustomEllipseCurve) : 
        cv2.ellipse(self.test_image, self.pattern.points.get(ellipse.center), ellipse.axes, 0, ellipse.compute_start_angle(self.pattern), ellipse.compute_end_angle(self.pattern), (255, 0, 0, 255), 3)
        #Here : cv2.ellipse(image, center_coordinates as (x,y), axesLength as (a,b), angle, startAngle, endAngle, color, thickness) 

    def draw_polyline(self, polyline : CustomPolyline) : 
        curve_points = polyline.compute_bezier_interpolation_curve(pattern=self.pattern)
        cv2.polylines(self.test_image, [curve_points], isClosed=False, color=(225, 0, 0, 255), thickness=3)