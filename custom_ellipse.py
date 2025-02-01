from sympy import Point, Ellipse, Line
import math

class CustomEllipseCurve() : 
    def __init__(self, axes, center, start_point, end_point):
        self.axes = axes
        self.center = center
        self.start_point = start_point
        self.end_point = end_point

    def compute_start_angle(self, pattern) :

        # Get center and start points
        center_point = pattern.points.get(self.center)
        start_point = pattern.points.get(self.start_point)
        
        # Create a horizontal reference line and the line to start point
        reference_line = Line(Point(center_point[0], center_point[1]), 
                            Point(center_point[0] + 1, center_point[1]))
        point_line = Line(Point(center_point[0], center_point[1]), 
                        Point(start_point[0], start_point[1]))
        
        # Calculate angle between lines
        angle_rad = float(point_line.angle_between(reference_line))
        
        # Convert to degrees
        angle_deg = math.degrees(angle_rad)
        
            
        print(f"START ANGLE IS : {angle_deg}")
        return angle_deg

    def compute_end_angle(self, pattern) :

        # Get center and start points
        center_point = pattern.points.get(self.center)
        end_point = pattern.points.get(self.end_point)
        
        # Create a horizontal reference line and the line to start point
        reference_line = Line(Point(center_point[0], center_point[1]), 
                            Point(center_point[0] + 1, center_point[1]))
        point_line = Line(Point(center_point[0], center_point[1]), 
                        Point(end_point[0], end_point[1]))
        
        # Calculate angle between lines
        angle_rad = float(point_line.angle_between(reference_line))
        
        # Convert to degrees
        angle_deg = math.degrees(angle_rad)
    
            
        print(f"END ANGLE IS : {angle_deg}")
        return angle_deg
    
    def get_a_b(self) :
        return self.axes
    


    """

    angle_start = 90
    line1 = Line(Point(mid_point[0], mid_point[1]), Point(tan_end_point[0], tan_end_point[1]))
    line2 = Line(Point(mid_point[0], mid_point[1]), Point(mid_point[0]+50, mid_point[1]))
    angle_rad = line1.angle_between(line2)

    # Convert the angle to degrees
    angle_end = angle_rad.evalf() * 180 / 3.141592653589793
    angle_end = float(angle_end) * -1
    
    """
