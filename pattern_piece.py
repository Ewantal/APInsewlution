""" Classe mère implémentant un bout de patron indépendant
Attributs : 
- une liste de points (x, y) contenant les coordonnées des points nécéssaires à la création du patron
- une liste de liens spécifiant quels points sont reliés et comment (ligne, courbe, ellipse ?) 
 """

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any


class PatternPiece(ABC) : 
    def __init__(self, name : str):
        self.name = name
        # points contains all the points present in the pattern in a dict as {"A" : (xa, ya), "B" : (xb, yb) ...}
        self.points : Dict[str, Tuple[float, float]] = {}
        # links contains all the links that need to be drawn between the points as {"AB" : line, "BC" : ellipse ...}
        self.links: Dict[str, str] = {}

    def initiate_points(self, letter_points) : 
        for i in letter_points : 
            self.points[i] = (None, None)

    def set_point(self, point_to_set : str, value : Tuple[float, float]) : 
        self.points[point_to_set] = value

    def center_points_on_screen(self, margin) : 
        """
            Used to center the points on the screen as they are initated in the left-upper corner
        """
        for point_name, coordinates in self.points.items():
            x, y = int(coordinates[0]), int(coordinates[1])
            self.points[point_name] = [x+margin, y+margin]

    def get_point_x_value(self, point : str) :
        return self.points.get(point)[0]
    
    def get_point_y_value(self, point : str) :
        return self.points.get(point)[1]