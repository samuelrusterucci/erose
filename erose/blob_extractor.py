import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle

white_red  = LinearSegmentedColormap.from_list('truncated_bone', ["#EAEAEA",  "#C8C8C8" , "#D42C2C"], N=256)
white_red.set_bad(color='gray')

blue_red  = LinearSegmentedColormap.from_list('truncated_bone', ["#526594", "#000000", "#B14E4E"], N=256)
blue_red.set_bad(color='gray')

from .coordinate_transform import CoordinateTransformer
from .kernels import KernelGenerator
from .map_processing import make_elliptical_masks
from .splining_isochrone import open_and_spline_parsec

class BlobExtractor():
    '''
    This class extracts blob overdensities from a map obtained with the 
    'ConvolverRunner' class
    
    Parameters
    ----------
    CR_Obj : ConvolverRunner
        An object from the 'ConvolverRunner' class

    threshold : floats
        The blobs will be selected until a certain threshold. If a blob is below
        that threshold, it will not be listed in the output file        

    min_nb_stars : float
        Minimum number of stars in the overdensity, that is the number of stars 
        in the pixels above the aforementioned 'threshold' and within a same 
        overdensity (blob)

    field_edge_value : float
        Says if an overdensity is within field_edge_value degree of the edge

    Returns
    -------
    self.blob_properties : pd.core.frame.DataFrame 
        The different columns of the dataframe are the following:
            - peak_max : maximum value of the blob according to the density map
                        the maximum value is 36.74
            - ra : right ascension
            - dec : declination 
            - blob_dist : distance that maximised the detection of the blob
            - blob_size : size of the kernel that maximised the detection of the blob
            - noise_com : fraction of the noise kernel--which estimates the background--
                        for which we have data
            - field_edg : boolean signaling if the detection is at the edge of the field
    '''
    def __init__(self, CR_Obj: 'ConvolverRunner', threshold: float=11.,
                       min_nb_stars: float=5, field_edge_value: float=0.5):
        self.CR_Obj           = CR_Obj 
        self.min_nb_stars     = min_nb_stars
        self.field_edge_value = field_edge_value

        self.extraction(threshold)
        self.blob_classifier()
        self.blob_properties()
        self.empty_blob_remover()

    def extraction(self, threshold: float=11.):
        '''
        Method making masks of the blobs using scipy
        
        Parameters
        ----------
        threshold : floats
            The blobs will be selected until a certain threshold. If a blob is below
            that threshold, it will not be listed in the output file       

        Returns
        -------
        self.blob_labels : np.ndarray
            2D map where individual overdensities are identified by integers 
        '''
        #Data smoothing to "remove" the effects of the CCD gaps
        data   = sp.ndimage.uniform_filter(self.CR_Obj.log_map, 5) #5 is the smoothing_radius
        #Making the blob by bringing the values below the threshold to the threshold
        thresh = data > threshold
        #Fills internal holes within detected blobs
        filled = sp.ndimage.binary_fill_holes(thresh)
        #Labels the blobs, the result is stored in blobs, classified from lower left to upper right
        self.blob_labels, _ = sp.ndimage.label(filled)

    def blob_classifier(self):
        '''
        Classifies the blobs from the most promising one to the less

        Returns
        -------
        self.blob_slices : np.ndarray
            Returns a local mask of the overdensities
        '''
        #Dictionary of the max values of each blobs (key=label: value=maximum)
        label_to_max = {}
        for label in range(1, np.max(self.blob_labels) + 1):
            label_to_max[str(label)] = np.max(self.CR_Obj.log_map[(self.blob_labels == label)])
        label_to_max = dict(sorted(label_to_max.items(), key=lambda item: item[1], reverse=True))

        #Once dictionary ordered, we recreate the self.blob_label array with the re-ordered blobs now
        blob_array = np.zeros(np.shape(self.blob_labels), dtype=np.int16)
        for blob in range(len(label_to_max)):
            temp_mask = (self.blob_labels == int(list(label_to_max.keys())[blob]))
            blob_array[temp_mask] = blob + 1
        self.blob_labels = blob_array    ;    del blob_array

        #Gives a selection rectangle (Python slice) around each of the blobs
        self.blob_slices = sp.ndimage.find_objects(self.blob_labels)
        self.max_values  = np.array(list(label_to_max.values()))

    def blob_properties(self):
        '''
        Extracts the different properties of the blobs   

        Returns
        -------
        self.blob_properties : pd.core.frame.DataFrame 
            For more details see the description of the class
        '''
        #Finds the "average centroid" of each blob and then converts them from (xi, eta) to (ra, dec)
        blob_centres_ra = np.zeros(len(self.blob_slices))
        blob_centres_de = np.zeros(len(self.blob_slices))
        ra_centre       = np.zeros(len(self.blob_slices))
        de_centre       = np.zeros(len(self.blob_slices))
        for blob in range(len(self.blob_slices)):
            blob_centres_ra[blob] = np.median(self.CR_Obj.hist.x_centre[self.blob_slices[blob][0], self.blob_slices[blob][1]])
            blob_centres_de[blob] = np.median(self.CR_Obj.hist.y_centre[self.blob_slices[blob][0], self.blob_slices[blob][1]])
            ra_centre[blob], de_centre[blob] = CoordinateTransformer.plane_to_sphere(blob_centres_ra[blob],
                                                        blob_centres_de[blob], self.CR_Obj.phi_c, self.CR_Obj.theta_c)

        #Finds which kernel/distance have maximised that overdensity
        blob_dist = np.zeros(len(self.blob_slices), dtype='<U32')
        blob_size = np.zeros(len(self.blob_slices))
        for blob in range(len(self.blob_slices)):
            arr = self.CR_Obj.which_distance[self.blob_slices[blob][0], self.blob_slices[blob][1]].flatten()
            unique_values, counts = np.unique(arr[arr != '0.0'], return_counts=True) 
            blob_dist[blob]       = unique_values[np.argmax(counts[counts != 0])]
            
            arr = self.CR_Obj.which_kernel[self.blob_slices[blob][0], self.blob_slices[blob][1]].flatten()
            unique_values, counts = np.unique(arr[arr != 0], return_counts=True) 
            blob_size[blob]       = unique_values[np.argmax(counts[counts != 0])]*self.CR_Obj.resolution

        #Number of stars
        field_edg = np.zeros(len(self.blob_slices), dtype=np.int8) 
        dist_2edg = int((self.field_edge_value*60)/self.CR_Obj.resolution) 
        size_hist = len(self.CR_Obj.hist.x_centre) 
        for blob in range(len(self.blob_slices)):
            pix_x_centre = int((self.blob_slices[blob][1].stop + self.blob_slices[blob][1].start)/2)
            pix_y_centre = int((self.blob_slices[blob][0].stop + self.blob_slices[blob][0].start)/2)
            if (pix_x_centre < dist_2edg) or (pix_x_centre > size_hist - dist_2edg) or (pix_y_centre < dist_2edg) or (pix_y_centre > size_hist - dist_2edg):
                field_edg[blob] = 1

        #What is the completeness of the noise kernel around each blob
        noise_completness   = np.zeros(len(self.blob_slices))
        noise_kernel,  _, _ = KernelGenerator.annulus_kernel(self.CR_Obj.background_annulus_size[0],
                                                            self.CR_Obj.background_annulus_size[1])
        for blob in range(len(self.blob_slices)):
            noise_completness[blob] = self.CR_Obj.noise_completeness(blob_centres_ra[blob], blob_centres_de[blob], noise_kernel)
            
        d = {'peak_max':   self.max_values, 'ra':       ra_centre, 'dec':    de_centre, 
             'blob_dist': blob_dist, 'blob_size': blob_size, 'noise_com': noise_completness,
             'field_edg': field_edg}
        self.blob_properties = pd.DataFrame(data=d).round(2)
        mask = (self.blob_properties['blob_dist'] != ''
           ) & (self.blob_properties['blob_size'] != 0)
        self.blob_properties = self.blob_properties[mask]#.reset_index(drop=True)
    
    def empty_blob_remover(self):
        '''
        Removes overdensities containing less than self.min_nb_stars (default=5) stars

        Returns
        -------
        self.blob_properties : pd.core.frame.DataFrame 
            For more details see the description of the class
        '''
        #When the mask of a blob is empty it gets automatically removed
        blobs_to_remove       = []
        number_of_stars       = np.zeros(len(self.blob_properties), dtype=np.int32)
        self.visualise_field = np.zeros(np.shape(self.blob_labels))

        # for blob in range(len(self.blob_properties)):
        for idx, blob in enumerate(self.blob_properties.index):
            #Dictionary key for the distance which maximised the blob detection
            key_distance = str(self.blob_properties['blob_dist'].loc()[blob])

            #Mask of the current blob
            current_blob_map = np.zeros(np.shape(self.blob_labels))
            current_blob_map[(self.blob_labels == blob + 1)] += 1

            #Histogram of the stars in the region at the maximised distance
            self.CR_Obj.histogram_and_footprint(self.CR_Obj.distance_dictionary[key_distance], 
                                                self.CR_Obj.isochrone_weights[key_distance],
                                                False)

            #Number of stars in the overdensity
            number_of_stars[idx] = np.sum((current_blob_map*self.CR_Obj.hist.H)[(current_blob_map*self.CR_Obj.hist.H > 0)])

            if number_of_stars[idx] < self.min_nb_stars:
                blobs_to_remove.append(blob)
            else:
                self.visualise_field += current_blob_map
        
        self.blob_properties['nb_src'] = number_of_stars
        self.blob_properties           = self.blob_properties.drop(blobs_to_remove)

    def field_visualiser(self):
        '''
        This allows one to easily visualise the different blobs listed in 
        self.blob_properties

        Returns
        -------
        A matplotlib plot to visualise the kernel
        '''
        fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=151)

        ax.pcolormesh(self.CR_Obj.hist.x_centre, self.CR_Obj.hist.y_centre,
                       self.visualise_field, cmap='Greys')
        xlim, ylim = ax.get_xlim(), ax.get_ylim()

        for blob in self.blob_properties.index:
            x_centre = np.mean(self.CR_Obj.hist.x_centre[self.blob_slices[blob][0],
                                                          self.blob_slices[blob][1]])
            y_centre = np.mean(self.CR_Obj.hist.y_centre[self.blob_slices[blob][0],
                                                          self.blob_slices[blob][1]])

            angle = np.arctan2(y_centre, x_centre)

            if angle > 0:
                if angle < np.pi/2:
                    text_pos_x = -(np.max(self.CR_Obj.hist.x_centre[self.blob_slices[blob][0],self.blob_slices[blob][1]]) - np.min(self.CR_Obj.hist.x_centre[self.blob_slices[blob][0], self.blob_slices[blob][1]]))/2
                    text_pos_y = -(np.max(self.CR_Obj.hist.y_centre[self.blob_slices[blob][0], self.blob_slices[blob][1]]) - np.min(self.CR_Obj.hist.y_centre[self.blob_slices[blob][0], self.blob_slices[blob][1]]))/2
                    ax.text(x_centre + text_pos_x, y_centre + text_pos_y, f"{blob}", color="#000000", fontsize=10, horizontalalignment='left', verticalalignment='top') 
                else:
                    text_pos_x = (np.max(self.CR_Obj.hist.x_centre[self.blob_slices[blob][0], self.blob_slices[blob][1]]) - np.min(self.CR_Obj.hist.x_centre[self.blob_slices[blob][0], self.blob_slices[blob][1]]))/2
                    text_pos_y = -(np.max(self.CR_Obj.hist.y_centre[self.blob_slices[blob][0], self.blob_slices[blob][1]]) - np.min(self.CR_Obj.hist.y_centre[self.blob_slices[blob][0], self.blob_slices[blob][1]]))/2
                    ax.text(x_centre + text_pos_x, y_centre + text_pos_y, f"{blob}", color="#000000", fontsize=10, horizontalalignment='right', verticalalignment='top') 
            else:
                if angle < np.pi/2:
                    text_pos_x = -(np.max(self.CR_Obj.hist.x_centre[self.blob_slices[blob][0], self.blob_slices[blob][1]]) - np.min(self.CR_Obj.hist.x_centre[self.blob_slices[blob][0], self.blob_slices[blob][1]]))/2
                    text_pos_y = (np.max(self.CR_Obj.hist.y_centre[self.blob_slices[blob][0], self.blob_slices[blob][1]]) - np.min(self.CR_Obj.hist.y_centre[self.blob_slices[blob][0], self.blob_slices[blob][1]]))/2
                    ax.text(x_centre + text_pos_x, y_centre + text_pos_y, f"{blob}", color="#000000", fontsize=10, horizontalalignment='left', verticalalignment='bottom') 
                else:
                    text_pos_x = (np.max(self.CR_Obj.hist.x_centre[self.blob_slices[blob][0], self.blob_slices[blob][1]]) - np.min(self.CR_Obj.hist.x_centre[self.blob_slices[blob][0], self.blob_slices[blob][1]]))/2
                    text_pos_y = (np.max(self.CR_Obj.hist.y_centre[self.blob_slices[blob][0], self.blob_slices[blob][1]]) - np.min(self.CR_Obj.hist.y_centre[self.blob_slices[blob][0], self.blob_slices[blob][1]]))/2
                    ax.text(x_centre + text_pos_x, y_centre + text_pos_y, f"{blob}", color="#000000", fontsize=10, horizontalalignment='right', verticalalignment='bottom') 

            ax.scatter(x_centre, y_centre, s=1, c='#C90C0B', alpha=0.5) 

        ax.set_xlabel(r'$\xi$ (deg)')   ;   ax.set_ylabel(r'$\eta$ (deg)')
        ax.set_xlim(xlim[1], xlim[0]) ; ax.set_ylim(ylim[0], ylim[1])
        plt.show()

    def zoom_on_overdensity(self, blob_id: int=0, distance_around: float=18,
                            mode: str='arcmin',
                            s: float=1, dpi: int=221, savefig: str=''):
        '''
        Visualise the individual stars of the overdensity with a zoom in the region

        Parameters
        -------
        blob_id : int
            Blob index given in the self.blob_properties table

        distance_around : float 
            Plus and minus limits of the plot given in arcmin. Default is 18' = 0.3 deg

        mode : str
            View displayed in either 'arcmin' or 'deg'

        s, dpi : same as matplotlib counterparts

        savefig : str
            The plot will be saved as "savefig.pdf"
        '''
        centre_x = np.median(self.CR_Obj.hist.x_centre[self.blob_slices[blob_id][0], 
                                                       self.blob_slices[blob_id][1]])
        centre_y = np.median(self.CR_Obj.hist.y_centre[self.blob_slices[blob_id][0], 
                                                       self.blob_slices[blob_id][1]])

        final_mask = np.zeros_like(self.CR_Obj.xi, dtype=bool)
        final_mask = (abs(self.CR_Obj.xi - centre_x) < distance_around/60
                 ) & (abs(self.CR_Obj.eta - centre_y) < distance_around/60)
        
        fig, ax = plt.subplots(figsize=(4,4), dpi=221)

        if mode == 'arcmin':
            ax.scatter(x=(self.CR_Obj.xi[final_mask] - centre_x)*60, 
                    y=(self.CR_Obj.eta[final_mask] - centre_y)*60,
                    c='#000000', s=s, linewidths=0)
            ax.set_xlabel(r"$\xi$ (arcmin)")   
            ax.set_ylabel(r"$\eta$ (arcmin)")
            ax.set_xlim(-distance_around, distance_around)
            ax.set_ylim(-distance_around, distance_around)

        else:
            ax.scatter(x=self.CR_Obj.xi[final_mask] - centre_x, 
                    y=self.CR_Obj.eta[final_mask] - centre_y,
                    c='#000000', s=s, linewidths=0)
            ax.set_xlabel(r"$\xi$ (deg)")   
            ax.set_ylabel(r"$\eta$ (deg)")
            ax.set_xlim(-distance_around/60, distance_around/60)
            ax.set_ylim(-distance_around/60, distance_around/60)

        ax.invert_xaxis()

        if len(savefig) != 0:
            plt.savefig(savefig +'.pdf', dpi=dpi)

        plt.show()

    def diagnostic_plot(self, blob_id: int, IS_obj: 'IsochroneSelector',
                        isochrone_distance: float=None,
                        blob_size: float=None,
                        min_col_cmd: float=None,
                        max_col_cmd: float=None,
                        min_mag_cmd: float=10, 
                        max_mag_cmd: float=30,
                        colour_label: str=None,
                        mag_label: str=None,
                        s: float=1, dpi: int=300, savefig: str=None):
        '''
        Spatial visualisation of the overdensity showing what its colour-magnitude 
        diagram looks like in comparison to the field

        Parameters
        -------
        blob_id : int
            Blob index given in the self.blob_properties table

        IS_obj : 'IsochroneSelector' 
            IsochroneSelector objects providing the isochrone, the input colours and 
            magnitudes or stars

        isochrone_distance : float
            Isochrone shift provided in kpc

        blob_size : float
            Size of the central "candidate" region. If None, the value from the 
            blob_properties variable will be used

        min_col_cmd, max_col_cmd : floats
            minimum and maximum colour values in the CMD

        min_mag_cmd, max_mag_cmd : floats
            minimum and maximum magnitude values in the CMD

        colour_label, mag_label : strs
            Strings providing the labels on the x and y axis of the CMD

        s, dpi : float, int
            same as matplotlib counterparts

        savefig : str
            The plot will be saved as "savefig.pdf"
        '''
        r_h = blob_size or self.blob_properties.loc()[blob_id]['blob_size']
        inner_annulus = 3. * r_h
        outer_annulus = np.sqrt((r_h)**2 + (3.*(r_h))**2)
        
        #Pre-selection of the data 
        centre_x = np.median(self.CR_Obj.hist.x_centre[self.blob_slices[blob_id][0], 
                                                       self.blob_slices[blob_id][1]])
        centre_y = np.median(self.CR_Obj.hist.y_centre[self.blob_slices[blob_id][0], 
                                                       self.blob_slices[blob_id][1]])

        final_mask = np.zeros_like(self.CR_Obj.xi, dtype=bool)
        final_mask = (abs(self.CR_Obj.xi - centre_x) < 1.2*outer_annulus
                 ) & (abs(self.CR_Obj.eta - centre_y) < 1.2*outer_annulus)
        
        xi  = (self.CR_Obj.xi[final_mask] - centre_x)*60
        eta = (self.CR_Obj.eta[final_mask] - centre_y)*60

        mag1_name = list(IS_obj.isochrone_colour_mag.values())[0][0]
        mag2_name = list(IS_obj.isochrone_colour_mag.values())[0][1]

        temp_colour = (IS_obj.isochrone_table[mag1_name] - IS_obj.isochrone_table[mag2_name]).copy()
        temp_mag    = (IS_obj.isochrone_table[mag1_name]).copy()

        if min_col_cmd is None: min_col_cmd = np.mean(temp_colour) - 1e3
        if max_col_cmd is None: max_col_cmd = np.mean(temp_colour) + 1e3

        if isochrone_distance:
            temp_mag += 5*np.log10(isochrone_distance*1000) - 5 
        else:
            temp_mag += 5*np.log10(float(self.blob_properties.loc()[blob_id]['blob_dist'])*1000) - 5 

        inner_mask   = (xi**2 + eta**2) < (r_h)**2
        annulus_mask = ((xi**2 + eta**2) > (inner_annulus)**2
                   ) & ((xi**2 + eta**2) < (outer_annulus)**2)
        
        transparency = 0.2

        # Draw the ellipses (rotated by 'angle')
        circle1 = Circle((0, 0), radius=r_h, angle=0,
                        color="#3C55D2", alpha=transparency)
        circle2 = Circle((0, 0), radius=outer_annulus, angle=0,
                        color="#3C55D2", alpha=transparency)
        circle3 = Circle((0, 0), radius=inner_annulus, angle=0,
                        facecolor='white', alpha=1, edgecolor="#3C55D2", lw=0)
        circle4 = Circle((0, 0), radius=inner_annulus, angle=0,
                        facecolor='white', alpha=transparency, edgecolor="#3C55D2", lw=1)

        fig = plt.figure(figsize=(8.2, 3.5), dpi=221)
        gs = fig.add_gridspec(1, 4, width_ratios=[1, 0.25, 0.8, 0.8], wspace=0.)

        # First plot: independent y-axis
        ax0 = fig.add_subplot(gs[0, 0])

        ax_off = fig.add_subplot(gs[0, 1])
        ax_off.set_axis_off()

        # Three plots sharing y-axis
        ax1 = fig.add_subplot(gs[0, 2])
        ax2 = fig.add_subplot(gs[0, 3], sharey=ax1)

        for e in [circle2, circle3, circle4, circle1]:
            ax0.add_patch(e)

        ax0.scatter(xi, eta, c='black', s=s, linewidths=0.0)
        ax0.set_xlim(-1.2*outer_annulus, 1.2*outer_annulus)
        ax0.set_ylim(-1.2*outer_annulus, 1.2*outer_annulus)
        ax0.invert_xaxis()
        ax0.set_title(f"({self.blob_properties.loc()[blob_id]['ra']}, {self.blob_properties.loc()[blob_id]['dec']})deg / {self.blob_properties.loc()[blob_id]['peak_max']}", 
                      fontsize=13)
        ax0.set_xlabel(r'$\xi$ (arcmin)')
        ax0.set_ylabel(r'$\eta$ (arcmin)')

        ax1.scatter(temp_colour, temp_mag, s=1/2, linewidths=0, c='black')
        ax1.scatter(
            (list(IS_obj.input_stars_colour_mag.values())[0])[final_mask][inner_mask], 
            (list(IS_obj.input_stars_colour_mag.values())[-1])[final_mask][inner_mask],
            s=s, edgecolor='black', linewidths=0.
        )

        xlim_cst = (np.max(temp_colour) - np.min(temp_colour))*0.05
        ax1.set_xlim(
            max(np.min(temp_colour) - xlim_cst, min_col_cmd),
            min(np.max(temp_colour) + xlim_cst, max_col_cmd)
        )

        ax1.set_ylim(
            max(np.min((list(IS_obj.input_stars_colour_mag.values())[-1])[final_mask][inner_mask]), min_mag_cmd),
            min(np.max((list(IS_obj.input_stars_colour_mag.values())[-1])[final_mask][inner_mask]), max_mag_cmd)
        )
        ax1.invert_yaxis()

        ax1.set_title('Candidate', loc='right', fontsize=12)

        if colour_label: ax1.set_xlabel(colour_label)
        else: ax1.set_xlabel(f"{list(IS_obj.isochrone_colour_mag.keys())[0]}")
        
        if mag_label: ax1.set_ylabel(mag_label)
        else: ax1.set_ylabel(f"{list(IS_obj.isochrone_colour_mag.keys())[-1]}")

        ax2.scatter(temp_colour, temp_mag, s=1/2, linewidths=0, c='black')
        ax2.scatter(
            (list(IS_obj.input_stars_colour_mag.values())[0])[final_mask][annulus_mask], 
            list(IS_obj.input_stars_colour_mag.values())[-1][final_mask][annulus_mask],
            s=s, edgecolor='black', linewidths=0.
        )
        
        ax2.set_xlim(
            max(np.min(temp_colour) - xlim_cst, min_col_cmd),
            min(np.max(temp_colour) + xlim_cst, max_col_cmd)
        )

        ax2.set_title('Field', loc='right', fontsize=12)
        
        if colour_label: ax2.set_xlabel(colour_label)
        else: ax2.set_xlabel(f"{list(IS_obj.isochrone_colour_mag.keys())[0]}")
        
        ax2.set_ylabel("") 
        ax2.tick_params(labelleft=False)  

        plt.tight_layout()

        if savefig:
            plt.savefig(f'{savefig}.pdf', dpi=300)
        plt.show()