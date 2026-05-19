import numpy as np
def apply_kernel(dtm, center, k_size):
    """
    Extract a subsection of the dtm centered at the given point 
    by the given kernel size
    """
    x_0 = center[0] - k_size // 2
    y_0 = center[1] - k_size // 2

    x_0 = 0 if x_0 < 0 else x_0
    y_0 = 0 if y_0 < 0 else y_0

    return dtm[x_0: center[0] + k_size // 2 + 1, y_0: center[1] + k_size // 2 + 1]

def adaptive_dtm_filter(dtm):
    """
    Find noisy pixels in the dtm based on abnormal height values
    and replace them with the mean of the pixels in a window 
    that are below the some percentile
    """
    k_size = 7
    dtm = dtm.copy()

    for i in range(0, dtm.shape[0]):
        for j in range(0, dtm.shape[1]): 
            subsection = apply_kernel(dtm, (i, j), k_size).flatten()

            if subsection.shape[0] > 0:
                le_per = np.percentile(subsection, [50])

                if dtm[i,j] > le_per[0]:
                    dtm[i,j] = subsection[np.where(subsection <= le_per[0])].mean()

    return dtm

def hillshade(array,azimuth=90,angle_altitude=60):
    # array = (array- array.min())/(np.max(array)-np.min(array))
    azimuth = 360.0 - azimuth 
    
    x, y = np.gradient(array)
    slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)
    azm_rad = azimuth*np.pi/180. #azimuth in radians
    alt_rad = angle_altitude*np.pi/180. #altitude in radians
 
    shaded = np.sin(alt_rad)*np.sin(slope) + np.cos(alt_rad)*np.cos(slope)*np.cos((azm_rad - np.pi/2.) - aspect)
    
    return 255*(shaded + 1)/2

def calcSurf(cubo, porcentaje):
    barras = np.arange(0,100,0.5)
    cumula = np.cumsum(cubo,axis=0)
    maximos = np.max(cumula,axis=0)
    maximos[np.where(maximos == 0)] = 1
    cumula = cumula / maximos

    surf = cumula > porcentaje
    surf = np.argmax(surf, axis = 0)

    return surf

def createCHM(cubo, porcentaje= 0.95,porcentaje_dtm = 0.05):
    dtm = calcSurf(cubo,porcentaje_dtm)
    dtm = adaptive_dtm_filter(dtm)
    dem = calcSurf(cubo,porcentaje)
    chm = dem - dtm

    return chm, dtm, hillshade(dtm), dem

