import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from tqdm import tqdm
# from tqdm.notebook import tqdm

from .convolver_runner import ConvolverRunner
from .isochrone_cutter import IsochroneSelector
from .blob_extractor import BlobExtractor
from .coordinate_transform import CoordinateTransformer

try:
    from fastparquet import ParquetFile
    has_fastparquet = True
except ImportError:
    fastparquet     = None
    has_fastparquet = False

class SurveyRunner():
    def __init__(self, inter_field_span: float=5,
                 available_memory: str='medium',
                 name_output_file: str='output',
                 parquet_file_name: str=None,
                 additional_cut: tuple=None,
                 restrart_from_dec: float=None):
        '''
        The search is performed along declination stripes and the final result is stored
        as an ouput csv file

        Parameters 
        ----------
        signal_kernel_sizes, background_annulus_size : np.ndarray
            Same as in the launch_erose method

        inter_field_span : float
            Space (in deg) between the different fields

        available_memory : str
            Estimated memory usage to store the weigths of the stars. In order to 
            estimate the expected space taken up by the weights, please run the 
            erose.isochrone_cutter.estimated_memory_required() method
                'low': computes the weights of stars for each field, many stars have 
                    weights that are recomputed many times
                'medium': computes the weights of stars for each declination stripe, 
                    resulting is less recompute of star weights
                'high': computes the weights of stars for the whole catalogue at once,
                    resulting in no recomputing
                
        name_output_file : str
            Name of the output file

        additional_cut : list
            List of tuples that will be applied to the input .parquet file if this mode
            is choosen. E.g., [('gmag', '<', 27)]

        restrart_from_dec : float
            erose scans the sky by declination stripes from -90 to 90 deg. If an error 
            occur, one can restart the program from a different minimum declination. A
            simple estimate of the dec is enough as the code automatically converts the 
            value to the "nearest" declination stripe used

        Returns
        -------
        self.sky_overdensities : pd.DataFrame
            Table storing all of the overdensities that were found
        '''
        #Initialisation of the final table
        self.sky_overdensities = pd.DataFrame()
        
        self.available_memory = available_memory
        self.name_output_file = name_output_file

        self.parquet_file_name = parquet_file_name
        self.additional_cut    = additional_cut

        self.restrart_from_dec = restrart_from_dec
        self.inter_field_span = inter_field_span
        self._create_tessellation()

    def _create_tessellation(self):
        '''
        Creates the centres of the different fields that will be used.
        More details about the self.inter_field_span parameter in launch_full_sky_search

        Returns
        -------
        self.ra_overdensity_field, self.dec_overdensity_field : np.ndarray, np.ndarray
            Returns the centres of the field that will be explored
        '''
        ra_list, de_list = [], []  

        de_temp = np.linspace(-89.999 + self.inter_field_span/2, 
                    89.999 - self.inter_field_span/2, 
                    int(180 // self.inter_field_span)
        )

        #Converts declination estimate to real declination array value
        if self.restrart_from_dec:
            ar_diff = np.abs(de_temp - (self.restrart_from_dec))
            de_temp = de_temp[np.argmin(ar_diff):] 
        
        for declination in de_temp:
            ra = np.linspace(0.0001, 359.999, 
                int(360 // (self.inter_field_span/np.cos(np.deg2rad(declination))) + 1)
            )
            ra_list += list(ra)
            for _ in range(len(ra)): de_list += [declination]
            
        self.ra_overdensity_field, self.dec_overdensity_field = np.array(ra_list), np.array(de_list)
        
    def _cut_dec_stripe(self, current_dec: float):
        '''
        Makes a mask

        Parameters 
        ----------
        current_dec : float
            Centre declination of the current declination stripe

        Returns
        -------
        self.mask_current_stripe : np.ndarray, np.ndarray
            Mask the stars outside of the current declination stripe
        '''
        self.mask_current_stripe = (self.theta > current_dec - self.span
                               ) & (self.theta < current_dec + self.span)
        
    def isochrone_selector(self, distances: np.ndarray,
                           isochrone_table: pd.DataFrame,
                           isochrone_colour_mag: dict,
                           input_stars_colour_mag: dict,
                           paths_to_uncertainties: dict,
                           mask_size:list, 
                           mode: str='joblib',
                           apply_mask: bool=False):
        '''
        Uses the same arguments as the 'IsochroneSelector' class, more details there.

        Parameters 
        ----------
        input_stars_colour_mag : dict
            If used in the .parquet mode, its use is different as in the normal
            'IsochroneSelector' class. The keys remain the same, however instead of the 
            star colours and magnitudes, we have a list of the column names 
                e.g.: {'g-r': ['gmag', 'rmag']}, where 'gmag' is the name of the column
                in the input "parquet_file_name" file
            
        apply_mask : float
            If True, applies an additional mask to the weights
        '''
        self.distances              = distances
        self.isochrone_table        = isochrone_table
        self.isochrone_colour_mag   = isochrone_colour_mag
        self.input_stars_colour_mag = input_stars_colour_mag
        self.paths_to_uncertainties = paths_to_uncertainties
        self.mask_size              = mask_size
        self.mode                   = mode
        self.apply_mask             = apply_mask

    def convolver_runner(self, phi: np.ndarray | dict,
                         theta: np.ndarray | dict,
                         resolution: float=0.5,
                         footprint_resolution: int=8,
                         span: float=4):
        '''
        Uses the same arguments as the 'ConvolverRunner' class, more details there.

        Parameters 
        ----------
        phi, theta : np.ndarray | dict
            If np.ndarray, they are simply arrays storing the positions of the stars
            If dict, we are in the .parquet mode, and they store the column names of the
            input file, e.g., phi = {'ra': 'RA'} if 'RA' is the column name in the input
            "parquet_file_name" file
        '''
        self.phi                  = phi
        self.theta                = theta
        self.resolution           = resolution
        self.footprint_resolution = footprint_resolution
        self.span                 = span

    def _compute_masks_and_probs(self):
        '''
        Selects and computes the masks and weights according to the memory mode used
        '''
        if self.available_memory == 'low':
            current_dec_star_colours = {}
            for keys in self.input_stars_colour_mag:
                current_dec_star_colours[keys] = self.input_stars_colour_mag[keys][self.mask_current_stripe][self.CR_obj.region_mask]

        elif self.available_memory == 'medium':
            current_dec_star_colours = {}
            for keys in self.input_stars_colour_mag:
                current_dec_star_colours[keys] = self.input_stars_colour_mag[keys][self.mask_current_stripe]

        elif self.available_memory == 'high':
            current_dec_star_colours = {}
            for keys in self.input_stars_colour_mag:
                current_dec_star_colours[keys] = self.input_stars_colour_mag[keys]
            
        IS_obj = IsochroneSelector(
                    distances=self.distances,
                    isochrone_table=self.isochrone_table,
                    isochrone_colour_mag=self.isochrone_colour_mag,
                    input_stars_colour_mag=current_dec_star_colours,
                    paths_to_uncertainties=self.paths_to_uncertainties,
                    mask_size=self.mask_size
        )

        if self.apply_mask:
            if self.mode == 'joblib': self.masks, self.probs = IS_obj.masks_and_probs_joblib()
            if self.mode == 'multiprocessing': self.masks, self.probs = IS_obj.masks_and_probs_multiprocessing()
        
        else:
            if self.mode == 'joblib': _, self.probs = IS_obj.masks_and_probs_joblib()
            if self.mode == 'multiprocessing': _, self.probs = IS_obj.masks_and_probs_multiprocessing()

    def _current_masks(self):
        '''
        Resizes the mask dictionary that is going to be given to launch_erose
        '''
        current_masks = {}

        if self.available_memory == 'low':
            for keys in self.masks:
                current_masks[keys] = self.masks[keys]

        elif self.available_memory == 'medium':
            for keys in self.masks:
                current_masks[keys] = self.masks[keys][self.CR_obj.region_mask]

        elif self.available_memory == 'high':
            for keys in self.masks:
                current_masks[keys] = self.masks[keys][self.mask_current_stripe][self.CR_obj.region_mask]

        return current_masks
    
    def _current_probs(self):
        '''
        Resizes the probs dictionary that is going to be given to launch_erose
        '''
        current_probs = {}

        if self.available_memory == 'low':
            for keys in self.probs:
                current_probs[keys] = self.probs[keys]

        elif self.available_memory == 'medium':
            for keys in self.probs:
                current_probs[keys] = self.probs[keys][self.CR_obj.region_mask]

        elif self.available_memory == 'high':
            for keys in self.probs:
                current_probs[keys] = self.probs[keys][self.mask_current_stripe][self.CR_obj.region_mask]

        return current_probs
    
    def _open_parquet(self, current_declination: float=None):
        '''
        Opens either a slice or an entire parquet file with or without filters

        Returns
        -------
        self.phi, self.theta, self.input_stars_colour_mag : np.ndarrays
            Instead of the input dictionaries those are converted into the format 
            actually containing the data
        '''
        if (self.available_memory == 'low') | (self.available_memory == 'medium'):
            #Only a stripe of declination is open
            applied_mask = [(list(self.theta_name.values())[0], '>', current_declination - self.inter_field_span
                        ), (list(self.theta_name.values())[0], '<', current_declination + self.inter_field_span)]
            
            if isinstance(self.additional_cut, list):
                applied_mask.extend(self.additional_cut)
            
            temp_tab = pd.read_parquet(
                path=self.parquet_file_name,
                filters=applied_mask,
                columns=list(set(self.columns_provided_by_user)) #similar to np.unique
            )
            temp_tab = temp_tab.dropna().reset_index(drop=True)
            
            self.phi   = temp_tab[list(self.phi_name.values())[0]].to_numpy()
            self.theta = temp_tab[list(self.theta_name.values())[0]].to_numpy()

            for key, value in self.input_stars_colour_mag_name.items():
                if len(value) > 1:
                    self.input_stars_colour_mag[key] = (temp_tab[value[0]] - temp_tab[value[1]]).to_numpy()
                else:
                    self.input_stars_colour_mag[key] = (temp_tab[value[0]]).to_numpy()
            del temp_tab

        if (self.available_memory == 'high'):
            if isinstance(self.additional_cut, list):
                temp_tab =  pd.read_parquet(
                    path=self.parquet_file_name,
                    columns=list(set(self.columns_provided_by_user)), #similar to np.unique
                    filters=self.additional_cut
                )
            else:
                temp_tab =  pd.read_parquet(
                    path=self.parquet_file_name,
                    columns=list(set(self.columns_provided_by_user)), #similar to np.unique
                )
            temp_tab = temp_tab.dropna().reset_index(drop=True)

            self.phi   = temp_tab[list(self.phi_name.values())[0]].to_numpy()
            self.theta = temp_tab[list(self.theta_name.values())[0]].to_numpy()

            for key, value in self.input_stars_colour_mag_name.items():
                if len(value) > 1:
                    self.input_stars_colour_mag[key] = (temp_tab[value[0]] - temp_tab[value[1]]).to_numpy()
                else:
                    self.input_stars_colour_mag[key] = (temp_tab[value[0]]).to_numpy()
            del temp_tab

    @staticmethod
    def remove_doubles(input_table: pd.DataFrame):
        '''
        Removes doubles in a logical way:
            1) Finds all occurences of the same object in the detection (5' conesearch)
            2) Sorts those by the fraction of the background kernel in the data
                3) If only edge detection: keeps detection with highest noise_com
                        if multiple share the same noise_com, then peak_max
                3) If at least one none edge detection: highest noise_com is kept
                        if multiple share the same noise_com, then peak_max

        Parameters 
        ----------
        input_table : pd.DataFrame
            Table resulting from the raw file output by launch_full_sky_search.
            Must contain the following columns: peak_max, ra, dec, ,noise_com, field_edg

        Returns
        -------
        input_table : pd.DataFrame
            The input table is returned but sorted
        '''
        index = 0

        while True:
            try:
                tab_idx = np.array(input_table.index)[index]

                #Table of all similar sources sorted by noise_com
                temp_tab = input_table[CoordinateTransformer.cone_search(input_table['ra'], input_table['dec'], 
                                    input_table['ra'].loc()[tab_idx], input_table['dec'].loc()[tab_idx], 5)].sort_values('noise_com', ascending=False)

                #Is it an edge detection or not?
                field_edg_mask = (temp_tab['field_edg'] == 0)

                #Index to be kept
                idx_kept = 0

                #If only edge detection, simply take highest edge detection value
                if len(temp_tab[field_edg_mask]) == 0:
                    #Masks all similar values of noise completeness equal to the first
                    mask_eq_noise_com = temp_tab[~field_edg_mask]['noise_com'] == np.array(temp_tab[~field_edg_mask]['noise_com'])[0]

                    #If multiple, then use the peak max value
                    if len(temp_tab[~field_edg_mask][mask_eq_noise_com]) != 0:
                        idx_kept = np.array(temp_tab[~field_edg_mask][mask_eq_noise_com].sort_values('peak_max', ascending=False).index)[0]
                    
                    #Else just the noise completeness
                    else:
                        idx_kept = np.array(temp_tab[~field_edg_mask].index)[0]

                #If not edge detection, simply take highest value
                elif len(temp_tab[field_edg_mask]) != 0:
                    mask_eq_noise_com = temp_tab[field_edg_mask]['noise_com'] == np.array(temp_tab[field_edg_mask]['noise_com'])[0]

                    if len(temp_tab[field_edg_mask][mask_eq_noise_com]) != 0:
                        idx_kept = np.array(temp_tab[field_edg_mask][mask_eq_noise_com].sort_values('peak_max', ascending=False).index)[0]
                    
                    else:
                        idx_kept = np.array(temp_tab[field_edg_mask].index)[0]

                idx_kept_mask = (temp_tab.index == idx_kept)

                input_table = input_table.drop(index=temp_tab.index[~idx_kept_mask])
                index += 1

            except:
                break

        input_table = input_table.sort_values('peak_max', ascending=False).reset_index(drop=True)

        input_table = input_table.drop_duplicates(subset=['ra', 'dec'])
        input_table = input_table[input_table['noise_com'] > 0.21].reset_index(drop=True)

        temp_ra = np.rad2deg(np.deg2rad(input_table['ra'].to_numpy()) % (2*np.pi))
        input_table['ra'] = temp_ra

        return input_table
    
    def diagnostic_plot(self, blob_id: int,
                        min_mag_cmd: float=10,
                        max_mag_cmd: float=30,
                        s: float=1):
        '''
        Spatial visualisation of the overdensity showing what its colour-magnitude 
        diagram looks like in comparison to the field

        Parameters
        -------
        blob_id : int
            Blob index given in the self.blob_properties table

        min_mag_cmd, max_mag_cmd : float, float
            minimum and maximum magnitude values in the CMD

        s : float
            Sizes of the markers for the stars, same as matplotlib
        '''
        r_h = self.sky_overdensities.loc()[blob_id]['blob_size']
        inner_annulus = 3. * r_h
        outer_annulus = np.sqrt((r_h)**2 + (3.*(r_h))**2)

        #Projects the spherical coordinates into the tangent plane
        xi, eta = CoordinateTransformer.spheretoplane(
            self.phi[self.mask_current_stripe][self.around_density_mask], 
            self.theta[self.mask_current_stripe][self.around_density_mask],
            self.sky_overdensities['ra'].loc()[blob_id],
            self.sky_overdensities['dec'].loc()[blob_id]
        )

        xi *= 60 ; eta *= 60

        final_mask = (abs(xi) < 1.5*outer_annulus
                    ) & (abs(eta) < 1.5*outer_annulus)

        mag1_name = list(self.isochrone_colour_mag.values())[0][0]
        mag2_name = list(self.isochrone_colour_mag.values())[0][1]
        temp_colour = (self.isochrone_table[mag1_name] - self.isochrone_table[mag2_name]).copy()
        temp_mag    = (self.isochrone_table[mag1_name]).copy()
        temp_mag += 5*np.log10(float(self.sky_overdensities.loc()[blob_id]['blob_dist'])*1000) - 5 

        inner_mask   = (xi[final_mask]**2 + eta[final_mask]**2) < (1.5*r_h)**2
        annulus_mask = ((xi[final_mask]**2 + eta[final_mask]**2) > (inner_annulus)**2
                    ) & ((xi[final_mask]**2 + eta[final_mask]**2) < (outer_annulus)**2)
        
        transparency = 0.2

        # Draw the ellipses (rotated by 'angle')
        circle1 = Circle((0, 0), radius=r_h, angle=0,
                        color="#3C55D2", alpha=transparency)
        circle2 = Circle((0, 0), radius=outer_annulus, angle=0,
                        color="#3C55D2", alpha=transparency)
        circle3 = Circle((0, 0), radius=inner_annulus, angle=0,
                        facecolor='white', alpha=1, edgecolor="#3C55D2", lw=0)
        circle4 = Circle((0, 0), radius=inner_annulus, angle=0,
                        facecolor='white', alpha=transparency, 
                        edgecolor="#3C55D2", lw=1)

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
        ax0.set_title(f"({self.sky_overdensities['ra'].loc()[blob_id]}, {self.sky_overdensities['dec'].loc()[blob_id]})deg / {self.sky_overdensities['peak_max'].loc()[blob_id]}", 
                      fontsize=13)
        ax0.set_xlabel(r'$\xi$ (arcmin)')
        ax0.set_ylabel(r'$\eta$ (arcmin)')

        ax1.scatter(temp_colour, temp_mag, s=1/2, linewidths=0, c='black')
        ax1.scatter(
            (list(self.input_stars_colour_mag.values())[0])[self.mask_current_stripe][self.around_density_mask][final_mask][inner_mask], 
            list(self.input_stars_colour_mag.values())[-1][self.mask_current_stripe][self.around_density_mask][final_mask][inner_mask],
            s=s, edgecolor='black', linewidths=0.
        )

        xlim_cst = (np.max(temp_colour) - np.min(temp_colour))*0.05
        ax1.set_xlim(
            np.min(temp_colour) - xlim_cst,
            np.max(temp_colour) + xlim_cst
        )

        ax1.set_ylim(
            max(np.min(list(self.input_stars_colour_mag.values())[-1][self.mask_current_stripe][self.around_density_mask][final_mask][inner_mask]), min_mag_cmd),
            min(np.max(list(self.input_stars_colour_mag.values())[-1][self.mask_current_stripe][self.around_density_mask][final_mask][inner_mask]), max_mag_cmd)
        )
        ax1.invert_yaxis()
        ax1.set_title('Candidate', loc='right', fontsize=12)
        ax1.set_xlabel(f"{list(self.isochrone_colour_mag.keys())[0]}")
        ax1.set_ylabel(f"{list(self.isochrone_colour_mag.keys())[-1]}")

        ax2.scatter(temp_colour, temp_mag, s=1/2, linewidths=0, c='black')
        ax2.scatter(
            (list(self.input_stars_colour_mag.values())[0])[self.mask_current_stripe][self.around_density_mask][final_mask][annulus_mask], 
            list(self.input_stars_colour_mag.values())[-1][self.mask_current_stripe][self.around_density_mask][final_mask][annulus_mask],
            s=s, edgecolor='black', linewidths=0.
        )
        
        ax2.set_xlim(
            np.min(temp_colour) - xlim_cst,
            np.max(temp_colour) + xlim_cst
        )

        ax2.set_title('Field', loc='right', fontsize=12)
        ax2.set_xlabel(f"{list(self.isochrone_colour_mag.keys())[0]}")
        ax2.set_ylabel("") ; ax2.tick_params(labelleft=False)  

        plt.tight_layout()

        plt.savefig(f'{blob_id}.pdf', dpi=300)
        plt.close()

    def launch_full_sky_search(self, signal_kernel_sizes: np.ndarray, 
                               background_annulus_size: np.ndarray,
                               diagnostic_plots: bool=False,
                               detection_threshold: float=13.,
                               min_nb_stars: int=5):
        '''
        The search is performed along declination stripes and the final result is stored
        as an ouput csv file

        Parameters 
        ----------
        signal_kernel_sizes, background_annulus_size : np.ndarray
            Same as in the launch_erose method

        detection_threshold : float
            Same as defined in the BlobExtractor class

        Returns
        -------
        self.sky_overdensities : pd.DataFrame
            Table storing all of the overdensities that were found
        '''
        self.signal_kernel_sizes     = signal_kernel_sizes
        self.background_annulus_size = background_annulus_size
        self.detection_threshold     = detection_threshold 
        self.min_nb_stars            = min_nb_stars

        self.parquet_check = False
        if isinstance(self.input_stars_colour_mag[list(self.input_stars_colour_mag)[0]][0], str):
            assert has_fastparquet, 'Please install the fastparquet package to proceed'
            
            #Checks if the input columns can be found into the input parquet file
            pf = ParquetFile(self.parquet_file_name)
            column_names = pf.columns
            self.columns_provided_by_user = list(self.phi.values()) + list(self.theta.values()) + sum(self.input_stars_colour_mag.values(), [])
            for user_cols in self.columns_provided_by_user:
                assert user_cols in column_names, f"It seems the column '{user_cols}' doesn't exist in '{self.parquet_file_name}'."

            #Creates a copy of the names, so phi/theta/... can be used as in the normal case
            self.phi_name                    = self.phi.copy()
            self.theta_name                  = self.theta.copy()
            self.input_stars_colour_mag_name = self.input_stars_colour_mag.copy()

            #Sets a boolean to say everything's good to run the method using parquet
            self.parquet_check = True

        status_line = tqdm(
            total=0,
            bar_format="{desc}",
            position=0
        )

        progress_bar = tqdm(
            total=len(self.ra_overdensity_field),
            desc="Fields treated",
            ascii=" ▖▘▝▗▚▞█",
            position=1,
            leave=True
        )

        #If the available memory is high, then the weights of all stars are pre-computed
        if self.available_memory == 'high': 
            if self.parquet_check:
                status_line.set_description_str(
                    f"Opening the full catalogue..."
                )
                self._open_parquet()

            status_line.set_description_str(
                f"Computing weights... (full input catalogue)"
            )
            self._compute_masks_and_probs()
        
        #Loop over the declinations_____________________________________________________
        for _, current_dec in enumerate(np.unique(self.dec_overdensity_field)):
            progress_bar.set_postfix_str(
                f"(current declination: {current_dec:.2f} deg)"
            )
            if ((self.available_memory == 'medium') | (self.available_memory == 'low')): 
                if self.parquet_check:
                    status_line.set_description_str(
                        f"Opening the current stripe..."
                    )
                    self._open_parquet(current_dec)

            self._cut_dec_stripe(current_dec=current_dec)

            #Checks if the current stripe is empty or not
            if len(self.phi[self.mask_current_stripe]) > 0:
                
                #Only the weights of stars from the current declination stripe are pre-computed
                if self.available_memory == 'medium': 
                    status_line.set_description_str(
                        f"Computing weights... (for the current declination stripe)"
                    )
                    self._compute_masks_and_probs()
                    
                #Loop over the right-ascensions_________________________________________
                for _, current_ra in enumerate(self.ra_overdensity_field[(self.dec_overdensity_field == current_dec)]):  
                    self.CR_obj = ConvolverRunner(
                        phi=self.phi[self.mask_current_stripe],                     
                        theta=self.theta[self.mask_current_stripe],
                        phi_c=current_ra,
                        theta_c=current_dec, 
                        resolution=self.resolution,
                        footprint_resolution=self.footprint_resolution
                    )

                    if self.CR_obj.fraction_covered > 0.05:  

                        if self.available_memory == 'low': 
                            status_line.set_description_str(
                                f"Computing weigths... (for the current field)"
                            )
                            self._compute_masks_and_probs()

                        current_probs = self._current_probs()

                        if self.apply_mask: current_masks = self._current_masks
                        else: current_masks = None

                        status_line.set_description_str(
                            f"Enhancing overdensities in the current field..."
                        )

                        self.CR_obj.launch_erose(
                            signal_kernel_sizes=self.signal_kernel_sizes,
                            background_annulus_size=self.background_annulus_size,
                            distance_dictionary=current_masks,
                            isochrone_weights=current_probs,
                            progress_bar=False,
                        )

                        BE_obj = BlobExtractor(CR_Obj=self.CR_obj, 
                                               threshold=self.detection_threshold,
                                               min_nb_stars=self.min_nb_stars)
                        self.sky_overdensities = pd.concat((self.sky_overdensities, 
                                                            BE_obj.blob_properties))
                    progress_bar.update(1)
                    self.sky_overdensities.to_csv(f'./raw_{self.name_output_file}.csv', 
                                                  index=False)
                    
            else:
                progress_bar.update(len(self.ra_overdensity_field[(self.dec_overdensity_field == current_dec)]))

        progress_bar.set_postfix_str(
            f" "
        )

        #Resets the indices of the final table
        self.sky_overdensities = self.sky_overdensities.reset_index(drop=True)

        status_line.set_description_str(
            f"Removing multiple detections..."
        )

        #Final table with all overdensities
        self.sky_overdensities = SurveyRunner.remove_doubles(self.sky_overdensities)
        self.sky_overdensities.to_csv(f'./{self.name_output_file}.csv', index=False)

        if diagnostic_plots:
            status_line.set_description_str(
                f"Making diagnostic plots..."
            )

            for plot in range(len(self.sky_overdensities)):
                current_dec = self.sky_overdensities['dec'].loc()[plot]

                if self.parquet_check:
                        self._open_parquet(current_dec)
                self._cut_dec_stripe(current_dec=current_dec)

                self.around_density_mask = CoordinateTransformer.cone_search(
                    self.phi[self.mask_current_stripe],
                    self.theta[self.mask_current_stripe],
                    self.sky_overdensities['ra'].loc()[plot],
                    self.sky_overdensities['dec'].loc()[plot],
                    self.sky_overdensities['blob_size'].loc()[plot]*7
                )

                self.diagnostic_plot(plot)
        
        status_line.set_description_str(
            f"Everything ran correctly."
        )
        status_line.close()
        progress_bar.close()