# object for process/input parameters

class Param:
    # parameters set by the user
    layer_height = 0.05               # layer height in mm
    exposure_time = 2                 # final exposure time in s

    # internal process parameters
    atten_coef = 11.82                # attenuation coefficient [1/mm]
    sigma = 2                         # standard deviation of gauss filter [mm]
    I_max = 1.93                      # default intensity for G = 1 [mW/cm2]
    E_crit = 0.97                     # energy exposure for a voxel to cure [mJ/cm2]
    alpha = 2.2702                    # grayscale-intensity interpolation curve fit [-]
    pxl_size = 0.05                   # pixel size of LCD screen [mm]

    # numerical modelling
    min_infl_factor = 0.001           # minimum influence factor for energy exposure
