Point = tuple[float, float, float]
_EPS = 1e-12
_LINE_ANGLE_TOL = 1e-8   # total tangent rotation (rad) below which the arc is a straight line
_COLLINEAR_TOL = 1e-6
DEFAULT_COMPLETION = 0.999   # fraction of the movement considered "completed"