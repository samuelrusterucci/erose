import numpy as np
import matplotlib.pyplot as plt

from .coordinate_transform import CoordinateTransformer

class HistogramOnTheSky():
    '''
    This class allows one to create a 2D histogram of the density of stars on the sky
    projected onto a plane tangential to the central point
    
    Parameters
    ----------
    phi, theta : np.ndarrays
        phi and theta are the spherical coordinates of the input. They can either 
        correspond to (ra, dec) or to (l, b). 
    
    phi_c, theta_c : floats
        Centre of the region we want to study

    span : int 
        Span of the histogram, e.g. 5 would mean it goes from -5 to 5

    bin_size : float
        This is the size of the bins, given in arcminutes

    xi, eta : np.ndarrays
        If input (xi, eta), no need for the _Projection() method

    Returns
    -------
    X, Y, Z : np.ndarrays 
        They are 2D numpy arrays storing the (xi, eta) positions and the kernel "value"
        Their dimensions goes to (2*3sig by 2*3sig)
    '''
    def __init__(self, phi: np.ndarray, theta: np.ndarray,
        phi_c: float, theta_c: float, span: float, bin_size: float,
        xi: np.ndarray=np.array([]), 
        eta: np.ndarray=np.array([]),
        weight: np.ndarray=np.array([])):

        #Initialiser
        self.phi    = phi     ;   self.theta    = theta
        self.phi_c  = phi_c   ;   self.theta_c  = theta_c
        self.span   = span    ;   self.bin_size = bin_size/60
        self.weight = weight

        if len(xi) == 0:
            self._projection()  
        else:
            self.xi = xi    ;   self.eta = eta

        if len(weight) == 0:
            self.weight = np.ones(len(xi), dtype=np.float32)
                
        self._make_the_histrogram()

    def _projection(self) -> None:
        '''
        Selects and projects the stars from the input in the correct transverse plane

        Returns
        -------
        self.rough_mask, self.exact_mask : np.ndarrays 
            Masks selecting the stars around the central point

        self.xi, self.eta : np.ndarrays :
            Star coordinates in the (xi, eta) plane
        '''
        x_stars_deg = CoordinateTransformer.right_ascension_recentering(phi=self.phi,
                                            phi_c=self.phi_c)
        y_stars_deg = self.theta
        
        #Pre-selection of the stars in a region around the central object
        self.rough_mask = (x_stars_deg > -2*self.span/np.cos(np.deg2rad(self.theta_c))
                      ) & (x_stars_deg < 2*self.span/np.cos(np.deg2rad(self.theta_c))
                      ) & (y_stars_deg > self.theta_c - 2*self.span
                      ) & (y_stars_deg < self.theta_c + 2*self.span)

        xi, eta = CoordinateTransformer.sphere_to_plane(phi=x_stars_deg[self.rough_mask],
                             theta=y_stars_deg[self.rough_mask], 
                             phi_c=0, 
                             theta_c=self.theta_c)
        self.exact_mask   = (xi > -self.span) & (xi < self.span
                        ) & (eta > - self.span) & (eta < self.span)
        self.xi, self.eta = xi[self.exact_mask], eta[self.exact_mask]

    def _make_the_histrogram(self):
        '''
        Makes a histogram of the stars previously selected

        Returns
        -------
        self.x_edges, self.y_edges, self.x_centre, self.y_centre : np.ndarrays 
            Edges and centres of the bins of the generated histogram (in (xi, eta)
            coordinates)

        self.H : np.ndarray 
            2D final histogram
        '''
        #Now we create the arrays defining the edges of the histogram
        self.x_edges = np.arange(start=-self.span,
                                 stop=self.span+0.00001, 
                                 step=self.bin_size)
        self.y_edges = np.arange(start=-self.span, 
                                 stop=self.span+0.00001, 
                                 step=self.bin_size)
        
        #The centres of the bins of the histogram and a meshgrid of those
        x_centre = self.x_edges + self.bin_size/2
        y_centre = self.y_edges + self.bin_size/2
        self.x_centre, self.y_centre = np.meshgrid(x_centre[:-1], y_centre[:-1])
        
        #Makes the actual histogram
        H, _, _ = np.histogram2d(self.xi, self.eta, bins=(self.x_edges, self.y_edges),
                                 density=False, weights=self.weight)
        self.H = np.transpose(H)

    def digitalise(self, entry_mask: np.ndarray):
        '''
        This method is used to isolate the stars under a given mask

        Parameters
        ----------
        entry_mask : np.ndarray
            Mask for which we want to know which stars are under

        Returns
        -------
        self.star_blob_mask : np.ndarray 
            Actual mask of the stars under the mask

        self.xi_blob, self.eta_blob : np.ndarrays 
            (xi, eta) coordinates of the stars under the mask
        '''
        #Computes bin indices
        x_bin_indices = np.digitize(self.xi, self.x_edges) - 1
        y_bin_indices = np.digitize(self.eta, self.y_edges) - 1

        # Get the bin indices for the histogram
        y, x = np.indices(np.shape(self.H.T))

        # Create a boolean mask of the selected bins
        selected_bins = set(zip(x[entry_mask].ravel(), y[entry_mask].ravel()))

        # Vectorized check for membership
        self.star_blob_mask = np.array([(xi, yi) in selected_bins for xi, yi in zip(x_bin_indices, y_bin_indices)], dtype=bool)
        self.xi_blob        = self.xi[self.star_blob_mask] 
        self.eta_blob       = self.eta[self.star_blob_mask]    

    def visualiser(self, plot_the_stars: bool=False):   
        '''
        This allows one to easily visualise the histogram in the (xi, eta)
        
        Parameters
        ----------
        plot_the_stars : bool
            If True, the individual scattered stars will be visible

        Returns
        -------
        A matplotlib plot to visualise the kernel
        '''
        fig, ax, = plt.subplots(1, 1, figsize=(5, 4), dpi=221)

        pcm  = ax.pcolormesh(self.x_centre, self.y_centre, 
                             self.H, cmap='Greys', rasterized=True)
        cbar = plt.colorbar(pcm, ax=ax)
        cbar.set_label('Number of counts')

        if plot_the_stars == True:
            ax.scatter(self.xi, self.eta, c='#C90B0B', s=1, linewidths=0)

        ax.invert_xaxis()
        ax.set_xlabel(r"$\xi$ (deg)")   ;   ax.set_ylabel(r"$\eta$ (deg)")



def completeness_map(phi: np.ndarray, theta: np.ndarray, 
                     phi_c: float, theta_c: float,
                     span: float, size_small_bins: int=1, size_big_bins: int=8,
                     xi: np.ndarray=np.array([]), 
                     eta: np.ndarray=np.array([])):
    '''
    This class allows one to create a 2D histogram of the density of stars on the sky
    projected onto a plane tangential to the central point
    
    Parameters
    ----------
    phi, theta : np.ndarrays
        phi and theta are the spherical coordinates of the input. They can either 
        correspond to (ra, dec) or to (l, b). 
    
    phi_c, theta_c : floats
        Centre of the region we want to study

    span : int 
        Span of the histogram, e.g. 5 would mean it goes from -5 to 5

    size_small_bins, size_big_bins : int
        If used along with the "HistogramOnTheSky" class, size_small_bins should have 
        the same size as the bin_size from that class. The size_big_bins corresponds 
        to the bins over which the completness is checked. If no objects are found in
        the big bins, returns 0 and vice-versa. Both are given in arcmin

    xi, eta : np.ndarrays
        If input (xi, eta), no need for (phi, theta)
    
    Returns
    -------
    completness_mask_small : np.ndarrays 
        Mask of the size of the histogram of small bins but accounting for the
        completness deduced from the histogram with the big bins
    '''
    if len(xi) == 0:
        histogram_big_bins   = HistogramOnTheSky(phi=phi, theta=theta, 
                                                 phi_c=phi_c, theta_c=theta_c,
                                                 span=span, bin_size=size_big_bins)
        
        histogram_small_bins = HistogramOnTheSky(phi=phi, theta=theta, 
                                                 phi_c=phi_c, theta_c=theta_c,
                                                 span=span, bin_size=size_small_bins)

    else:
        histogram_big_bins   = HistogramOnTheSky(phi=phi, theta=theta,
                                                 phi_c=phi_c, theta_c=theta_c,
                                                 span=span, bin_size=size_big_bins,
                                                 xi=xi, eta=eta)
        histogram_small_bins = HistogramOnTheSky(phi=phi, theta=theta, 
                                                 phi_c=phi_c, theta_c=theta_c,
                                                 span=span, bin_size=size_small_bins, 
                                                 xi=xi, eta=eta)

    completness_mask_big   = histogram_big_bins.H.copy()
    completness_mask_big[completness_mask_big > 0.1] = 1
    
    completness_mask_small = histogram_small_bins.H.copy()
    completness_mask_small[completness_mask_small > 0.1] = 1

    #Finds which of the small bins are within or not the footprint
    x_bin_indices = np.digitize(x=histogram_small_bins.x_centre.flatten(), 
                                bins=histogram_big_bins.x_edges) - 1
    y_bin_indices = np.digitize(x=histogram_small_bins.y_centre.flatten(), 
                                bins=histogram_big_bins.y_edges) - 1

    #Get the bin indices for the histogram
    y, x = np.indices(np.shape(histogram_big_bins.H))

    #Creates a boolean mask of the selected bins
    selected_bins = set(zip(x[completness_mask_big.astype(bool)].ravel(), y[completness_mask_big.astype(bool)].ravel()))

    #Checks the membership 
    mask = np.array([(xi, yi) in selected_bins for xi, yi in zip(x_bin_indices, y_bin_indices)], dtype=bool)
    mask = mask.reshape(np.shape(completness_mask_small))

    completness_mask_small[mask] = 1

    return completness_mask_small.astype(np.int16), histogram_small_bins.x_centre, histogram_small_bins.y_centre

def redimension_kernel_to_image_size(x_centre: int, 
                                     y_centre: int,
                                     kernel_array: np.ndarray, 
                                     image_array: np.ndarray) -> np.ndarray:
    '''
    Creates a mask of the size of an input image with the shape of an input kernel

    Parameter x_centre, y_centre, kernel_array, image_array
    ----------
    x_centre, y_centre: int
        Image coordinates where we want to centre the input kernel

    kernel_array: np.ndarray
        Input kernel is a 2D numpy array 

    image_array: np.ndarray
        Input image is a 2D numpy array

    Returns
    -------
    masked_image : np.ndarray
        Mask of kernel of the size of the input image
    '''
    masked_image = np.ones(np.shape(image_array))
    x1, x2 = y_centre - len(kernel_array)//2 , y_centre + len(kernel_array)//2 + 1
    y1, y2 = x_centre - len(kernel_array)//2 , x_centre + len(kernel_array)//2 + 1

    kernel_x_min, kernel_x_max = 0, len(kernel_array)
    kernel_y_min, kernel_y_max = 0, len(kernel_array) 

    #Dealing with boundary effects
    if x1 < 0:
        kernel_x_min = abs(x1)
        x1 = 0

    if y1 < 0:
        kernel_y_min = abs(y1)
        y1 = 0

    if x2 >= len(masked_image):
        kernel_x_max = len(kernel_array) - abs(x2 - len(masked_image)) - 1
        x2 = len(masked_image) - 1

    if y2 >= len(masked_image):
        kernel_y_max = len(kernel_array) - abs(y2 - len(masked_image)) - 1
        y2 = len(masked_image) - 1

    masked_image[x1:x2, y1:y2] *= (~kernel_array.astype(bool)).astype(int)[kernel_x_min:kernel_x_max, kernel_y_min:kernel_y_max]
    masked_image = ~masked_image.astype(bool)
    
    return masked_image

def make_elliptical_masks(xi_arcmin: np.ndarray, eta_arcmin: np.ndarray, 
                          radius: np.float64, angle_deg: np.float64,
                          ellipticity: np.float64):
    '''
    Creates an elliptical mask, and provide other properties about the resulting ellipse

    Parameter 
    ----------
    xi_arcmin, eta_arcmin : np.ndarray
        Input stars projected in such a way that they are centred on 0

    radius : np.float64
        Length of the semi-major axis of the ellipse

    angle_deg : np.float64
        Angle of the ellipse

    ellipticity : np.float64
        If 0 circle, if 1 line

    Returns
    -------
    mask_central_ellipse, mask_annulus_ellipse : np.ndarrays
        Masks of the central ellipse (containing the overdensity) and the annulus 
        ellipse supposedly being representative of the background. The inner annulus 
        radius stars at 3 times the radius of the central ellipse 

    others : np.float64
        The following parameters are also returned and correspond to the "diameter" of 
        the semi-minor/major axis of the inner and annulus ellipse:
            inner_circle_a, inner_circle_b, inner_annulus_a, 
            inner_annulus_b, outer_annulus_a, outer_annulus_b
    '''
    # Convert coordinates to arcminutes
    xi, eta = xi_arcmin, eta_arcmin

    # Compute semi-minor axis
    b = radius * (1 - ellipticity)

    # Compute area of central ellipse
    central_area = np.pi * radius * b

    # Annulus: starts 3σ (radii) away, equal area
    inner_annulus_a = 3 * radius
    inner_annulus_b = inner_annulus_a * (1 - ellipticity)
    outer_annulus_a = np.sqrt(inner_annulus_a**2 + (central_area / (np.pi*(1 - ellipticity))))
    outer_annulus_b = outer_annulus_a * (1 - ellipticity)

    # Rotate coordinates by the given angle
    theta   = np.deg2rad(angle_deg)
    xi_rot  = xi * np.cos(theta) + eta * np.sin(theta)
    eta_rot = -xi * np.sin(theta) + eta * np.cos(theta)

    # Elliptical radius squared
    r_ell_sq = (xi_rot / radius)**2 + (eta_rot / b)**2
    r_inner_sq = (xi_rot / inner_annulus_a)**2 + (eta_rot / inner_annulus_b)**2
    r_outer_sq = (xi_rot / outer_annulus_a)**2 + (eta_rot / outer_annulus_b)**2

    # Masks
    mask_central_ellipse  = r_ell_sq < 1
    mask_annulus_ellipse = (r_inner_sq > 1) & (r_outer_sq < 1)

    return mask_central_ellipse, mask_annulus_ellipse, 2*radius, 2*b, 2*inner_annulus_a, 2*inner_annulus_b, 2*outer_annulus_a, 2*outer_annulus_b