import numpy as np

class Kernel:
    # attributes:
    # weights
    # r_xy: influence radius in xy dir
    # r_z: influence radius in z dir

    def __init__(self, p):

        # calculate boundaries based on min influence value
        self.r_xy = int(np.floor(np.sqrt(-np.log(p.min_infl_factor) * 2 * p.sigma**2)))
        self.r_z = int(- np.log(p.min_infl_factor)/(p.atten_coef * p.layer_height))

        # initialize kernel
        self.weights = np.zeros((2*self.r_xy + 1, 2*self.r_xy + 1, 2*self.r_z + 1))

        # calculate entries of kernel
        for i in range(-self.r_xy, self.r_xy + 1):
            for j in range(-self.r_xy, self.r_xy + 1):
                for k in range(0, self.r_z + 1): # z dir: only values in positive direction, values for negative k stay zero
                    infl = np.exp(-((i)**2 + (j)**2)/(2 * p.sigma**2)  # pixel size not included
                              - p.atten_coef*k*p.layer_height) / (2*np.pi*p.sigma**2)
                    if infl > p.min_infl_factor:
                        self.weights[i+self.r_xy, j+self.r_xy, k+self.r_z] = infl
