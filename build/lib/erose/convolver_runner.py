import numpy as np
import scipy
from scipy.stats import gamma
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

erose_density = LinearSegmentedColormap.from_list(
    'truncated_bone', 
    ["#000000", "#526594", "#B14E4E", "#D1D0B3", '#FFFFFF'], 
    N=256
)
erose_density.set_bad(color='gray')

erose_density2 = LinearSegmentedColormap.from_list(
    'truncated_bone', 
    ["#000000", "#213A40", "#5D5349", "#EB6A24", "#F2A979", "#F8F0EF"], 
    N=256
)
erose_density2.set_bad(color='gray')

from .coordinate_transform import CoordinateTransformer
from .kernels import KernelGenerator
from .map_processing import HistogramOnTheSky, completeness_map, redimension_kernel_to_image_size

class ConvolverRunner():
    '''
    This class runs the convolution enhancing overdensities
    
    Parameters
    ----------
    phi, theta : np.ndarrays
        Input spherical coordinates of the sources, e.g.: (ra, dec) or (l, b)

    phi_c, theta_c : floats
        Centre around which the coordinates will be transformed from spherical 
        coordinates to planar using a gnomonic projection

    resolution : float
        Resolution of the underlying histogram (determining the counts) 
        This must be given in arcmin

    span : float
        The sources that will be consider for the analyses will go from
        (phi_c - span, phi_c - span) along x and (theta_c - span, theta_c - span)
        along the y axis

    footprint_resolution : int
        How much bigger are the pixels used to make the footprint compared to the 
        pixels used to make the analysis, that is the resolution parameter

    Returns
    -------
    self.stacked_map : dict 
        A dictionary listing the masks on stars from the input table
    '''
    def __init__(self, phi: np.ndarray,    theta: np.ndarray,
                       phi_c: float=209.5, theta_c: float=12.8,
                       resolution: float=1, span: float=4, 
                       footprint_resolution: int=8):
        #Initialises the limits of the histogram
        self.phi        = np.array(phi)          ;   self.theta   = np.array(theta)
        self.phi_c      = phi_c                  ;   self.theta_c = theta_c
        self.resolution = resolution             ;   self.span    = span      

        self.local_selector()                     
        
        if len(self.xi) == 0:
            self.fraction_covered = 0
        
        else:
            self.histogram_and_footprint(footprint=True, 
                                       footprint_resolution_factor=footprint_resolution)
            self.fraction_covered = len(self.completness_mask[self.completness_mask == 1])/len(self.completness_mask.flatten())
          
    def local_selector(self):
        '''
        Method selecting the stars in the region that we have selected

        Returns
        -------
        self.region_mask : np.ndarray
            Mask of the stars in the field

        self.xi, self.eta : np.ndarrays
            Arrays of the (xi, eta) coordinates of the selected stars
        '''
        x_stars_deg = CoordinateTransformer.right_ascension_recentering(self.phi,
                                                                        self.phi_c)
        y_stars_deg = self.theta
        
        #Pre-selection of the stars in a region around the central object
        rough_mask = (x_stars_deg > -2*self.span/np.cos(np.deg2rad(self.theta_c))
                 ) & (x_stars_deg < 2*self.span/np.cos(np.deg2rad(self.theta_c))
                 ) & (y_stars_deg > self.theta_c - 2*self.span
                 ) & (y_stars_deg < self.theta_c + 2*self.span)

        #Projecting stars from a sphere to a plane
        xi, eta = CoordinateTransformer.sphere_to_plane(x_stars_deg[rough_mask],
                                                        y_stars_deg[rough_mask], 
                                                        0, self.theta_c)
        
        #Finalises the mask really selecting the sources in a square around
        exact_mask   = (xi > -self.span) & (xi < self.span
                   ) & (eta > - self.span) & (eta < self.span)
        
        self.xi, self.eta = xi[exact_mask], eta[exact_mask]
        self.region_mask  = rough_mask.copy()
        self.region_mask[rough_mask] = exact_mask

    def histogram_and_footprint(self, distance_mask: np.ndarray=np.array([]),
                                      isochrone_weights: np.ndarray=np.array([]),
                                      footprint: bool=True,
                                      footprint_resolution_factor: int=8):
        '''
        Returns the histogram of the stars given as input (selected by the 
        distance_mask) and also returns a map approximating the footprint

        Parameters
        ----------
        distance_mask : np.ndarray
            Mask of the stars that should be selected (size of the input 
            self.phi/self.theta)

        footprint : bool
            If true, computes an approximate footprint automatically

        footprint_resolution_factor : int
            How much bigger are the pixels used to make the footprint compared to the 
            pixels used to make the analysis

        Returns
        -------
        self.hist : np.ndarray
            Histogram of the counts of stars

        self.completness_mask : np.ndarray
            Provides the pixels from self.hist for which there seems to be acquired data
        '''
        if len(distance_mask) == 0:
            self.hist = HistogramOnTheSky(np.array([]), np.array([]),
                                          self.phi_c, self.theta_c, 
                                          self.span, self.resolution,
                                          self.xi, self.eta)
            
        else:
            self.hist = HistogramOnTheSky(np.array([]), np.array([]),
                                        self.phi_c, self.theta_c, 
                                        self.span, self.resolution,
                                        self.xi[distance_mask], 
                                        self.eta[distance_mask],
                                        isochrone_weights[distance_mask])
             
        if footprint == True:
            self.completness_mask, _, _ = completeness_map(
                            np.array([]), np.array([]),
                            self.phi_c, self.theta_c, 
                            self.span, size_small_bins=self.resolution,
                            size_big_bins=footprint_resolution_factor*self.resolution, 
                            xi=self.xi, eta=self.eta)
            self.completness_mask_bool = self.completness_mask.astype(bool)

    def order_convolution(self, signal_kernel: np.ndarray, noise_kernel: np.ndarray):
        '''
        Makes the convolutions and estimates of the background

        Parameters
        ----------
        signal_kernel, noise_kernel : np.ndarrays
            2D numpy arrays corresponding to the kernels that will be used

        Returns
        -------
        self.prob : np.ndarray
            Represents the cumulative distribution function
        '''
        #Signal convolution_____________
        im1_num = scipy.signal.fftconvolve(self.hist.H, signal_kernel, mode='same')
        self.Z1 = np.array(im1_num.reshape(np.shape(self.hist.H)))*self.completness_mask    
        
        #Initialisation of the noise kernel_____________
        im1_den       = scipy.signal.fftconvolve(self.completness_mask,
                                                  noise_kernel, mode='same')
        self.Noise_nb = np.array(im1_den.reshape(np.shape(self.completness_mask))) 
        self.Noise_nb[abs(self.Noise_nb) <= 0] = 0.0000001 

        #Determination of the mean and of the variance_____________
        Z_1  = scipy.signal.fftconvolve(self.Z1,    noise_kernel, mode='same') 
        Z_1 /= self.Noise_nb
        Z_2  = scipy.signal.fftconvolve(self.Z1**2, noise_kernel, mode='same') 
        Z_2 /= self.Noise_nb
        var  = Z_2 - (Z_1)**2                    
        var  = np.maximum(var, 0) 
        
        #"Renormalisation" to deal with 0s_____________
        self.mean = (Z_1)*self.completness_mask  
        self.mean[self.mean == 0] = 0.0000000001
        self.var  = (var)*self.completness_mask  
        self.var[self.var == 0]   = 0.0000000001
        
        #Estimation of the shape (k) and scale (theta) parameters_____________
        self.k_param     = self.mean**2 / self.var
        self.theta_param = self.var / self.mean
        
        #Final map_____________
        self.prob = gamma.cdf(self.Z1, a=self.k_param, loc=0, scale=self.theta_param)
        self.prob *= self.completness_mask

    def multi_kernel_and_distances(self):
        '''
        Launches the different convolutions and stores the output maps

        Returns
        -------
        self.stacked_max : np.ndarray
            2D maps storing the maximum values reached by each of the individual maps

        self.which_kernel, self.which_distance : np.ndarrays
            2D maps storing which kernel/distance has maximised the self.stacked_max in
            each pixels
        '''
        if self.fraction_covered > 0.05:
            self.stacked_max    = np.zeros(np.shape(self.hist.H))
            self.which_kernel   = np.zeros(np.shape(self.hist.H))
            self.which_distance = np.zeros(np.shape(self.hist.H))

            if self.progress_bar == True:
                progress_bar = tqdm(total=len(self.signal_kernel_sizes) * len(self.distance_keys),
                                    desc='Convolution in progress, please wait an instant', ascii=" ▖▘▝▗▚▞█")
                
            for kernels in self.signal_kernel_sizes:
                signal_kernel, _, _ = KernelGenerator.gaussian_kernel(kernels,
                                                                      kernels)
                noise_kernel,  _, _ = KernelGenerator.annulus_kernel(self.background_annulus_size[0],
                                                                     self.background_annulus_size[1])
                for distances in self.distance_keys:
                    self.histogram_and_footprint(distance_mask=self.distance_dictionary[distances],
                                               isochrone_weights=self.isochrone_weights[distances], 
                                               footprint=False)
                    self.order_convolution(signal_kernel, noise_kernel)

                    #Temporarily stores the previous map of maximums and calculates the new
                    temp = self.stacked_max.copy()  
                    self.stacked_max = np.maximum(self.stacked_max, self.prob)

                    #Which kernels/distance maximise the detection
                    self.which_kernel   = np.where(self.stacked_max != temp,   kernels, self.which_kernel)
                    self.which_distance = np.where(self.stacked_max != temp, distances, self.which_distance)

                    if self.progress_bar == True:
                        progress_bar.update(1)
                
            if self.progress_bar == True:
                progress_bar.close()

            #Deals with nans in some uncomplete regions
            self.stacked_max[np.isnan(self.stacked_max)] = 0

    def noise_completeness(self, input_xi:float, 
                                 input_eta: float,
                                 noise_kernel: np.ndarray):
        '''
        Determines the fraction of the input noise kernel having access to data
        around an input (xi, eta) coordinate
        
        Parameters
        ----------
        input_xi, input_eta : float
            After finding the pixel closest to (input_xi, input_eta), does the fraction 
            determination

        noise_kernel : np.ndarray
            2D numpy arrays corresponding to the noise kernel

        Returns
        -------
        _ : float
            fraction of the noise kernel containing data
        '''
        pix_x = int((input_xi + self.span)/(self.resolution/60))
        pix_y = int((input_eta + self.span)/(self.resolution/60))

        image = self.Z1 
        kernel = noise_kernel  
        ker_l = len(kernel[0])

        tot_bins_kernel = len(noise_kernel[noise_kernel == 1])

        min_x, max_x = pix_y - ker_l//2, pix_y + ker_l//2 + 1
        min_y, max_y = pix_x - ker_l//2, pix_x + ker_l//2 + 1

        if min_x < 0: min_x = 0  
        if (max_x > len(image)): max_x = len(image)
        if min_y < 0: min_y = 0  
        if (max_y > len(image)): max_y = len(image)

        local_image = image[min_x:max_x, min_y:max_y]
        
        ker_temp = redimension_kernel_to_image_size(pix_x, pix_y,
                                                    kernel, self.Z1)
        kernel = ker_temp[min_x:max_x, min_y:max_y]
        
        distribution = (local_image*kernel).flatten() 
        distribution = distribution[distribution!=0]
        tot_bins_used = len(distribution)
        
        return tot_bins_used/tot_bins_kernel

    def launch_erose(self, signal_kernel_sizes: np.ndarray, 
                     background_annulus_size: np.ndarray,
                     distance_dictionary: dict=None, 
                     isochrone_weights: dict=None,
                     progress_bar: bool=True):
        '''
        Launches the erose enhancing method 

        Parameters
        -------
        signal_kernel_sizes : np.ndarray
            Array of the sizes of the signal kernels, e.g.: np.array([2,5,7])

        background_annulus_size : np.ndarray
            The first element correspond to the inner radius of the annulus, while the
            second to the outer, e.g.: np.array([inner_radius=25, outer_radius=40])

        distance_dictionary : dict
            The keys correspond to the distances (in kpc). The values correspond to 
            np.ndarrays of the size of the "input_colour" and "input_magnitude", 
            masking stars falling along an isochrone

        progress_bar : bool
            If true, a progress bar will be shown indicating the advancement of the 
            process
        '''
        if distance_dictionary:
            self.distance_dictionary = distance_dictionary
            if isochrone_weights:
                self.isochrone_weights = isochrone_weights
            else:
                self.isochrone_weights = {}
                for keys in list(distance_dictionary.keys()):
                    self.isochrone_weights[keys] = np.ones(len(distance_dictionary[keys]), dtype=np.float16)
        
        else:
            if isochrone_weights:
                self.isochrone_weights = isochrone_weights
                self.distance_dictionary = {}
                for keys in list(isochrone_weights.keys()):
                    self.distance_dictionary[keys] = np.ones(len(isochrone_weights[keys]), dtype=bool)

            else:
                distance_dictionary       = {}
                distance_dictionary['-1'] = np.ones(len(self.xi), dtype=bool)
                self.distance_dictionary  = distance_dictionary

                isochrone_weights       = {}
                isochrone_weights['-1'] = np.ones(len(self.xi), dtype=np.float16)
                self.isochrone_weights  = isochrone_weights

        self.distance_keys           = list(self.distance_dictionary.keys())
        self.signal_kernel_sizes     = signal_kernel_sizes
        self.background_annulus_size = background_annulus_size
        self.progress_bar            = progress_bar

        self.multi_kernel_and_distances()

        if self.fraction_covered > 0.05:
            self.stacked_max[self.stacked_max >= 1] = 1 - 10**(-15)
            self.log_map = -np.log(1 - self.stacked_max)   

    def visualise_local_distribution(self, xi: float=0.0, eta: float=0.0,
                                           nb_bins: int=15, cmap: str=erose_density2):
        '''
        Allows to visualise the gamma distribution fitted around a point
        
        Note: should be used to adapt the size of the noise kernel

        Parameters
        -------
        xi, eta : floats
            (xi, eta) coordinates (given in degrees) of the central pixel around which
            we would like to look at the local gamma distribution

        nb_bins : int
            Number of bins used for the histogram
        '''
        input_xi, input_eta = xi, eta
        pix_x = int((input_xi + self.span)/(self.resolution/60))
        pix_y = int((input_eta + self.span)/(self.resolution/60))

        image        = self.Z1 
        kernel, _, _ = KernelGenerator.annulus_kernel(self.background_annulus_size[0],
                                         self.background_annulus_size[1])
        ker_l = len(kernel[0])

        min_x, max_x = pix_y - ker_l//2, pix_y + ker_l//2 + 1
        min_y, max_y = pix_x - ker_l//2, pix_x + ker_l//2 + 1

        if min_x < 0: min_x = 0  
        if (max_x > len(image)): max_x = len(image)
        if min_y < 0: min_y = 0  
        if (max_y > len(image)): max_y = len(image)

        local_image = image[min_x:max_x, min_y:max_y]
        
        ker_temp = redimension_kernel_to_image_size(pix_x, pix_y,
                                                kernel, self.Z1)
        kernel = ker_temp[min_x:max_x, min_y:max_y]

        #Here, distribution has the size of the flattened kernel
        distribution = (local_image*kernel)[kernel!=0].flatten()
        hist_, bins_ = np.histogram(distribution, bins=nb_bins) 
        inter_bin    = (bins_[1] - bins_[0])/2

        #On the other hand, here, every 0 --either corresponding to 0s due to the kernel or not stars-- are removed 
        distribution2 = (local_image*kernel).flatten() 
        distribution2 = distribution2[distribution2!=0]
        hist_, bins_  = np.histogram(distribution2, bins=nb_bins) 
        inter_bin     = (bins_[1] - bins_[0])/2

        fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=300)
        pcm  = ax[0].pcolormesh(self.hist.x_centre[min_x:max_x, min_y:max_y], 
                                self.hist.y_centre[min_x:max_x, min_y:max_y], 
                                kernel*local_image, cmap=cmap, 
                                vmin=np.min(self.Z1), vmax=np.max(self.Z1), 
                                rasterized=True)
        cbar = plt.colorbar(pcm, ax=ax[0])

        ax[0].set_xlabel(r"$\xi$ (deg)") 
        ax[0].set_ylabel(r"$\eta$ (deg)") 
        ax[0].invert_xaxis()

        ax[1].stairs(hist_/len(distribution), bins_, color='black')
        x = np.linspace(min(bins_) + inter_bin, max(bins_) + inter_bin, len(bins_))
        y = gamma.pdf(x, self.k_param[pix_y][pix_x], loc=0,
                      scale=self.theta_param[pix_y][pix_x])
        ax[1].plot(x, y/sum(y),'black', lw=2, alpha=1, label='FFT', ls='-')

        plt.show()

    def visualiser(self, cmap: str=erose_density2, savefig: str='', dpi: int=321):
        '''
        This allows one to easily visualise the final map overdensity map
        
        Parameters
        ----------
        cmap, dpi : same as matplotlib counterparts

        savefig : str
            The plot will be saved as "savefig.pdf"

        Returns
        -------
        A matplotlib plot to visualise the kernel
        '''
        fig, ax, = plt.subplots(1, 1, figsize=(5, 4), dpi=221)

        ax.pcolormesh(self.hist.x_centre, self.hist.y_centre, 
                      ~self.completness_mask_bool, rasterized=True, cmap=cmap, alpha=1)
        pcm  = ax.pcolormesh(self.hist.x_centre, self.hist.y_centre,
                             self.log_map, cmap=cmap,
                             vmin=0, vmax=36.74, rasterized=True, alpha=0.85)      
        cbar = plt.colorbar(pcm, ax=ax)

        ax.set_xlabel(r"$\xi$ (deg)") ; ax.set_ylabel(r"$\eta$ (deg)")
        ax.invert_xaxis()
        cbar.set_label(r'$-\log(1 - \mathrm{CDF})$')

        if len(savefig) != 0:
            plt.savefig(savefig +'.pdf', dpi=dpi)
            
        plt.show()