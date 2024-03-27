import numpy as np
import math

class Walker:
    MOVE_UP = 0
    MOVE_RIGHT = 1
    MOVE_DOWN = 2
    MOVE_LEFT = 3
    
    MOVE_DIST = 1.0 # Distance of each move command
    MOVE_DIST_NOISE = 0.2 # Standard deviation of the distance
    MOVE_ANGLE_NOISE = 0.1 # Standard deviation of the angle
    
    MOVE_ANGLE_BIAS = [math.pi / 12, 0.0, -math.pi / 24, math.pi/20]
    
    def __init__(self, xpos = 0.0, ypos = 0.0):
        self.xpos = xpos
        self.ypos = ypos
        
    def command(self, command):
        angle = {
            Walker.MOVE_UP: math.pi / 2,
            Walker.MOVE_RIGHT: 0.0,
            Walker.MOVE_LEFT: math.pi,
            Walker.MOVE_DOWN: 3 * math.pi / 2
        }
        self._move(angle[command] + Walker.MOVE_ANGLE_BIAS[command])
            
    def _move(self, angle):
        angle = np.random.normal(loc = angle, scale = Walker.MOVE_ANGLE_NOISE)
        dist = np.random.normal(loc = Walker.MOVE_DIST, scale = Walker.MOVE_DIST_NOISE)
        self.xpos += math.cos(angle) * dist
        self.ypos += math.sin(angle) * dist
        
if __name__ == "__main__":
    import pygame
    pygame.init()
    screen = pygame.display.set_mode([500, 500])