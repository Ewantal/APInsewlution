import numpy as np

class CustomPolyline() : 
    def __init__(self, start_point, end_point, through_point):
        self.start_point = start_point
        self.end_point = end_point
        self.through_point = through_point

    def compute_quadratic_interpolation_curve(self, pattern) :
        """
        Create a quadratic interpolation curve passing through three specified points.
        
        Parameters:
        start_point: tuple (x, y) - starting point (armpit)
        end_point: tuple (x, y) - ending point (shoulder)
        through_point: tuple (x, y) - point the curve must pass through
        image_size: tuple (width, height) - size of the output image
        
        Returns:
        numpy array (image with curve)
        """
        # Convert points to numpy arrays
        P0 = np.array(pattern.points.get(self.start_point), dtype=np.float64)
        P2 = np.array(pattern.points.get(self.end_point), dtype=np.float64)
        P1 = np.array(pattern.points.get(self.through_point), dtype=np.float64)

        print(f"Specified start point:  {self.start_point} : {P0}")
        print(f"Specified through point:  {self.through_point} : {P1}")
        print(f"Specified end point:  {self.end_point} : {P2}")
        
        # Compute quadratic interpolation coefficients
        def compute_parabola_coefficients(x0, y0, x1, y1, x2, y2):
            """
            Compute coefficients for a parabola passing through three points
            """
            denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
            if abs(denom) < 1e-10:
                # Fallback if denominator is too close to zero
                return lambda x: np.mean([y0, y1, y2])
            
            A = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
            B = (x2*x2 * (y0 - y1) + x1*x1 * (y2 - y0) + x0*x0 * (y1 - y2)) / denom
            C = (x1 * x2 * (x1 - x2) * y0 + x2 * x0 * (x2 - x0) * y1 + x0 * x1 * (x0 - x1) * y2) / denom
            
            return lambda x: A*x*x + B*x + C
        
        # Generate x values
        x_values = np.linspace(P0[0], P2[0], 100)
        
        # Compute the parabolic function
        y_func = compute_parabola_coefficients(P0[0], P0[1], P1[0], P1[1], P2[0], P2[1])
        
        # Generate curve points
        curve_points = np.zeros((len(x_values), 2), dtype=np.int32)
        for i, x in enumerate(x_values):
            curve_points[i] = [int(x), int(y_func(x))]

        curve_length = np.sum(np.sqrt(np.sum(np.diff(curve_points, axis=0)**2, axis=1)))

        return curve_points
    
    def compute_bezier_interpolation_curve(self, pattern) :
        """
        Generate points along a quadratic Bézier curve
        
        Args:
        - start_point: Starting point (G)
        - control_point: Control point (H)
        - end_point: Ending point (I)
        - num_points: Number of points to generate along the curve
        
        Returns:
        - Array of points on the Bézier curve
        """
        I = np.array(pattern.points.get(self.start_point))  # start point
        H = np.array(pattern.points.get(self.through_point))  # control point
        G = np.array(pattern.points.get(self.end_point))  # end point


        control_point = 2 * np.array(H) - 0.5 * (np.array(I) + np.array(G))

        num_points = 100
        t_values = np.linspace(0, 1, num_points)
        curve_points = np.zeros((num_points, 2))
        
        for i, t in enumerate(t_values):
            # Quadratic Bézier curve formula
            curve_points[i] = (1-t)**2 * I + \
                            2 * (1-t) * t * H + \
                            t**2 * G
        
        return curve_points.astype(np.int32)