"""
Hérite de pattern_piece.py et implémente le patron particulier d'un buste
"""
from pattern_piece import PatternPiece
from custom_line import CustomLine
from custom_ellipse import CustomEllipseCurve
from custom_polyline import CustomPolyline

from math import sqrt
from sympy import symbols, solve, Eq, sqrt
import numpy as np

from shapely.geometry import Point, Polygon
from shapely.plotting import plot_polygon, plot_points


class BustPattern(PatternPiece) :

    BUFFER_OFFSET = 2

    def __init__(self, style, sleeves):
        super().__init__(name="BustPattern")
        self.style = style
        self.sleeves = sleeves
        if style == 'loose' :
            letter_points = "ABCDEFGHIJ"
        else :
            letter_points = "ABCDEFGHIJKL"
        self.initiate_points(letter_points=letter_points)
        print("DEBUG : this works")

    def create_body_pattern(self, distances) : 
        print("CREATING BODY PATTERN")
        self.set_body_pattern_points(distances)
        self.set_body_pattern_links(distances)
        self.center_points_on_screen(100)

    def set_body_pattern_points(self, distances) :
        polygon = {}

        polygon["A"] = [0, distances.get("hauteur_buste")]
        polygon["B"] = [distances.get("inter_aisselles"), distances.get("hauteur_buste")]
        polygon["C"] = [polygon["B"][0], distances.get("f-g")]
        polygon["D"] = [polygon["B"][0], 0]
        polygon["E"] = [distances.get("f-e"), polygon["D"][1]]
        polygon["F"] = [0,0]
        polygon["G"] = [distances.get("inter_aisselles") - distances.get("inter_epaules"), (polygon["C"][1] / 2)]
        polygon["I"] = [0, distances.get("f-i")]
        polygon["H"] = [distances.get("largeur_epaules"), (polygon["G"][1]+polygon["I"][1])/2] # replace with real values
        polygon["J"] = [polygon["H"][0]*1.7, polygon["I"][1]] #replace with real values

        if self.style != 'loose' :
            polygon["K"] = [distances.get("f-k"), (abs(polygon["A"][1] + polygon["I"][1]))/2]
            polygon["L"] = [distances.get("f-l"), (abs(polygon["A"][1] + polygon["I"][1]))/2]

        self.get_whole_polygon(polygon)

        # Add the points to self
        for key, values in polygon.items():
            self.set_point(key, values)
        
    def get_whole_polygon(self, polygon):
        
        self.mirror_points(polygon)
        print(f"\nInitial polygon : {polygon}")
        self.buffering(polygon)
        print(f"\nBuffered polygon : {polygon}")
        
    def mirror_points(self, polygon):
        # Order of the points in the polygon
        if self.style == 'loose' :
            original_points = ["C", "D", "E", "F", "G", "H", "J", "I", "A", "B"]
        else :
            original_points = ["C", "D", "E", "F", "G", "H", "J", "I", "L", "A", "B"]

        xx = [polygon[letter][0] for letter in original_points]
        yy = [polygon[letter][1] for letter in original_points]

        mirror_x = []
        mirror_y = []

        # Normalisation
        norm_x = xx[0]
        norm_y = yy[0]
        xx = [xx[index] - norm_x for index in range(len(xx))]
        yy = [yy[index] - norm_y for index in range(len(yy))]

        # Duplicate points
        count = len(xx) - 1
        while count != 1 :
            mirror_x.append(xx[count]*(-1))
            mirror_y.append(yy[count])
            count = count - 1

        # Denormalisation
        mirror_x = [mirror_x[index] + norm_x for index in range(len(mirror_x))]
        mirror_y = [mirror_y[index] + norm_y for index in range(len(mirror_y))]

        count = len(mirror_x) - 1
        for letter in original_points[2:-1]: # Remove C, D and B
            polygon[letter+"2"] = [mirror_x[count], mirror_y[count]]
            count = count - 1
    
    def buffering(self, polygon):
        max_x = polygon["F2"][0]
        max_y = polygon["A"][1]

        for key in polygon.keys():
            # Need integer for ellipse, CV2 does not take float
            polygon[key][0] = int(polygon[key][0] + (polygon[key][0]/max_x) * 4 * self.BUFFER_OFFSET)
            polygon[key][1] = int(polygon[key][1] + (polygon[key][1]/max_y) * 4 * self.BUFFER_OFFSET)

    def set_body_pattern_links(self, distances) : 
        a = abs(self.get_point_x_value("E") - self.get_point_x_value("D"))
        b = abs(self.get_point_y_value("C") - self.get_point_y_value("D"))
        a2 = abs(self.get_point_x_value("E2") - self.get_point_x_value("D"))
        b2 = abs(self.get_point_y_value("C") - self.get_point_y_value("D"))
        axes_ce = (a, b)
        axes_ce_2 = (a2, b2)
        self.links = {
            "AB" : CustomLine("A", "B"),
            "BA2" : CustomLine("A2","B"),
            "CE" : CustomEllipseCurve(axes_ce, "D", "C", "E"),
            "C2E2" : CustomEllipseCurve(axes_ce_2, "D", "C", "E2"),
            "EG" : CustomLine("E", "G"),
            "E2G2" : CustomLine("E2", "G2"),
            "GI" : CustomPolyline(start_point="I", end_point="G", through_point="J"),
            "G2I2" : CustomPolyline(start_point="I2", end_point="G2", through_point="J2"),
        }
        if self.style == 'loose' :
            self.links["IA"] = CustomLine("I", "A")
            self.links["I2A2"] = CustomLine("I2", "A2")
        else :
            self.links["IA"] = CustomPolyline(start_point="I", end_point="A", through_point="L")
            self.links["I2A2"] = CustomPolyline(start_point="I2", end_point="A2", through_point="L2")

    def compute_h_point(self, distances) : 
        return [50, 230]
    
    def compute_e_point(self, distances) : 
        """
        Trouvée en faisant le théorème de pythagore dans le triangle EGF car E-G est connue et G-F est connue
        """
        return int(sqrt(distances.get("cou_epaule")**2 - (distances.get("f-g"))**2))
    
    def get_width_height(self) :
        width = abs(self.get_point_x_value("A") - self.get_point_x_value("A2"))
        height = abs(self.get_point_y_value("A") - self.get_point_y_value("F"))
        return width, height