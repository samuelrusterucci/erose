import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from scipy.stats import multivariate_normal

#Used for parallel computing
import multiprocessing as mp
from multiprocessing import Pool
_pool = None

try:
    import joblib
    from joblib import Parallel, delayed
    has_joblib = True
except ImportError:
    joblib     = None
    has_joblib = False

class ExponentialFitting():
    def __init__(self, magnitude: np.ndarray, uncertainty_in_mag: np.ndarray):
        self.magnitude          = magnitude
        self.uncertainty_in_mag = uncertainty_in_mag
        self.fitting()
        self.create_model()

    def exponential_model(self, x, a, b, c):
        return a * np.exp(b * x) + c          

    def fitting(self):
        popt, _ = curve_fit(self.exponential_model, self.magnitude, 
                            self.uncertainty_in_mag)
        self.final_parameters = popt

    def from_magnitude_to_uncertainties(self, magnitude: np.ndarray):
        return self.exponential_model(magnitude, *self.final_parameters)
    
    def create_model(self):
        x_modeled = np.linspace(min(self.magnitude), 28, 5100).astype(np.float32)
        y_modeled = self.exponential_model(x_modeled,
                                          *self.final_parameters).astype(np.float32)
        y_modeled = np.where(y_modeled < 1e-5, 0, y_modeled)
        self.final_table = pd.DataFrame({'mag': x_modeled, 'u_mag': y_modeled})

    def save_model(self, name: str="hihi"):
        self.final_table.to_csv(name+'.csv', index=False)
    
    def visualiser(self):
        x_modeled = np.linspace(min(self.magnitude), max(self.magnitude), 100)
        y_modeled = self.exponential_model(x_modeled, *self.final_parameters)

        fig, ax = plt.subplots(figsize=(4.5,3.5), dpi=121)

        ax.scatter(self.magnitude, self.uncertainty_in_mag,
                    c='black', s=0.4, linewidths=0)
        ax.plot(x_modeled, y_modeled, c='#3A54A4')
        ax.set_xlabel('Magnitude')
        ax.set_ylabel('Magnitude uncertainty')

        plt.show()

def estimated_memory_required():
    nb_objects   = int(input('What is the number of objects in the input catalogue?'))
    nb_distances = int(input('How many distance shifts are you planning to probe?'))

    print(f"Expected memory usage: {nb_distances*np.zeros(nb_objects, dtype=np.float64).nbytes / (1024 ** 3):.5f} GB")

class Isochroner():
    '''
    Generates a mask around an input isochrone

    Parameters 
    ----------
    isochrone_table : pandas.dataFrame
        Pandas data frame containing the magnitudes of a single stellar track

    isochrone_colour_mag: list
        List of lists of the different colours of the CMD in which the search will be 
        made. E.g.: if one wants to use both the (g-r) and the (u-g) colours from the 
        SDSS, colour_names = [ ['g', 'r'], ['u', 'g'] ]

    mag_name : str
        Which magnitude should be used for the CMD

    distance : float
        Distance of the isochrone (given in kpc)
    '''
    def __init__(self, isochrone_table: 'pandas.core.frame.DataFrame'=None,
                 isochrone_colour_mag: dict={'g-r': ['gmag', 'rmag']},
                 distance: float=10.):
        #Properties common to all types of isochrones
        self.colour_mag    = isochrone_colour_mag
        self._colour_names = {}
        self._mag_name     = '' 
        for _, (key, item) in enumerate(isochrone_colour_mag.items()):
            if len(item) == 2:
                self._colour_names[key] = item
            else:
                self._mag_name = item[0]

        self._distance              = distance*1000
        self._input_isochrone_table = isochrone_table

        self.iso_mags   = {}
        self.u_iso_mags = {}

        self.iso_colours   = {}
        self.u_iso_colours = {}
            
        unique_input_names = np.unique(np.array(sum(list(self._colour_names.values()) + [[self._mag_name]], [])))
        for unique_names in unique_input_names:
            self.iso_mags[unique_names] = 5*np.log10(self._distance) - 5 + isochrone_table[unique_names]
        del unique_input_names


    def _mask_maker(self, input_stars_colour_mag: dict,
                    paths_to_uncertainties: dict=None,
                    mask_size: list=[0.01, 0.01, 0.01], 
                    test_on_subset: int=None, correlation: bool=True):
        '''
        Wraps the three functions actually making the mask around the isochrone

        Parameters 
        ----------
        input_colours : list
            List of numpy arrays, each

        colour_names: list
            List of lists of the different colours of the CMD in which the search will be 
            made. E.g.: if one wants to use both the (g-r) and the (u-g) colours from the 
            SDSS, colour_names = [ ['g', 'r'], ['u', 'g'] ]

        mag_name : str
            Which magnitude should be used for the CMD

        distance : float
            Distance of the isochrone (given in kpc)

        correlation : bool 
            If true, accounts for correlations in the CMD

        Returns
        -------
        self.iso_colour, self.iso_mag : np.array, np.array
            Fit of the isochrone

        self.selected_star_colour, self.selected_star_mag : np.array, np.array
            Stars contained within the region around the isochrone
        '''
        self.input_colours    = {}
        self.input_magnitude = np.array([])

        if test_on_subset is None:
            test_on_subset = len(list(input_stars_colour_mag.values())[0])

        for idx, (key, item) in enumerate(input_stars_colour_mag.items()):
            if (idx < len(input_stars_colour_mag) - 1):
                self.input_colours[key] = item[:test_on_subset]
            else:
                self.input_magnitude = item[:test_on_subset]

        self._distance_to_isochrone()
        self._open_uncertainty_table(paths_to_uncertainties)
        self._making_the_mask(mask_size, correlation)

    def _distance_to_isochrone(self):
        '''
        Each points of the input star is effectively associated to a point of the input
        stellar track 

        Returns
        -------
        self.dist_from_iso : np.ndarray
            Distance of the input point to the closest point of the isochrone

        self.idx_closest_iso_point : np.ndarray
            Index of the isochrone point to which the input point is closest
        '''
        for key in self.input_colours.keys():
            assert key in self._colour_names.keys(), f"'{key}' isn\'t define as a colour nor a magnitude in the input 'colour_mag' dictionary"

        isochrone_points   = []
        self._input_points = []
        for _, (key, item) in enumerate(self._colour_names.items()):
            isochrone_points.append(self.iso_mags[item[0]] - self.iso_mags[item[1]])
            self._input_points.append(self.input_colours[key])
        isochrone_points.append(self.iso_mags[self._mag_name])
        self._input_points.append(self.input_magnitude)

        if len(self._input_points) == len(isochrone_points):
            tree = cKDTree(np.column_stack(isochrone_points)) 
            self.dist_from_iso, self.idx_closest_iso_point = tree.query(
                np.column_stack(self._input_points), k=1,
                distance_upper_bound=np.inf
                )
        else:
            print("The inputs were not provided in the correct format.")
              
    def _open_uncertainty_table(self, paths_to_uncertainties: dict=None):
        if paths_to_uncertainties is not None:
            for item in sum(self.colour_mag.values(), []):
                assert item in paths_to_uncertainties.keys(), f"you have not provided uncertainties for '{item}'"
                
            table_of_uncertainties = pd.DataFrame()
            for name in paths_to_uncertainties:
                table_of_uncertainties = pd.read_csv(paths_to_uncertainties[name])
                indices = np.searchsorted(table_of_uncertainties["mag"][:-1],
                                            self.iso_mags[name])
                self.u_iso_mags[name] = np.array(table_of_uncertainties["u_mag"][indices])

            for _, item in enumerate(self._colour_names.values()):
                self.iso_colours[item[0] + '-' + item[1]]   = self.iso_mags[item[0]] - self.iso_mags[item[1]]
                self.u_iso_colours[item[0] + '-' + item[1]] = np.sqrt(self.u_iso_mags[item[0]]**2 + self.u_iso_mags[item[1]]**2)
            self._paths_to_uncertainties = paths_to_uncertainties
    
        else:
            for _, item in enumerate(self._colour_names.values()):
                self.iso_colours[item[0] + '-' + item[1]]   = self.iso_mags[item[0]] - self.iso_mags[item[1]]
                self.u_iso_colours[item[0] + '-' + item[1]] = np.zeros(len(self.iso_mags[item[0]]))
            self.u_iso_mags[self._mag_name] = np.zeros(len(self.iso_mags[item[0]]))
            self._paths_to_uncertainties = None

    def _making_the_mask(self, mask_size: list=[0.01, 0.01, 0.01], 
                         correlation: bool=True):
        assert len(mask_size) == len(self.colour_mag), f"the mask size doesn\'t match the number of dimensions provided in the colour-magnitude space"
        
        points_to_check = np.unique(self.idx_closest_iso_point)
        self.total_mask = np.zeros(len(self.input_magnitude), dtype=bool)
        self.point_prob = np.zeros(len(self.input_magnitude), dtype=np.float64)

        #Determination of the Jacobian used to get the final covariance matrix
        jacobian     = np.zeros((len(self.iso_colours) + 1, len(self.iso_colours) + 1))
        unique_bands = np.unique(sum(list(self.colour_mag.values()), []))

        for columns in range(len(jacobian) - 1):
            for rows in range(len(jacobian)):
                if unique_bands[rows] == list(self.colour_mag.values())[columns][0]:
                    jacobian[rows][columns] = 1

                if unique_bands[rows] == list(self.colour_mag.values())[columns][1]:
                    jacobian[rows][columns] = -1

        for rows in range(len(jacobian)):
            if unique_bands[rows] == list(self.colour_mag.values())[len(jacobian)-1][0]:
                jacobian[rows][len(jacobian)-1] = 1

        #Determination of the \tilde{\Sigma}_j matrix
        if correlation == False:
            for iso_point_idx in points_to_check:
                mean = np.zeros(len(self.iso_colours) + 1)
                cov  = np.zeros((len(self.iso_colours) + 1, len(self.iso_colours) + 1))

                for idx, diff_colours in enumerate(list(self.iso_colours)):
                    mean[idx]     = self.iso_colours[diff_colours][iso_point_idx]
                    cov[idx][idx] = np.sqrt(mask_size[idx]**2 + self.u_iso_colours[diff_colours][iso_point_idx]**2)
                mean[idx+1]       = self.iso_mags[self._mag_name][iso_point_idx]
                cov[idx+1][idx+1] = np.sqrt(mask_size[idx+1]**2 + self.u_iso_mags[self._mag_name][iso_point_idx]**2)

                current_mask = (self.idx_closest_iso_point == iso_point_idx)
                temp_list   = []
                for input_axes in self._input_points:
                    temp_list.append(input_axes[current_mask])

                final_cov = cov

                pdf_vals = multivariate_normal.pdf(x=np.array(temp_list).T, 
                                                mean=mean, cov=final_cov)
                pdf_max  = multivariate_normal.pdf(mean, mean=mean, cov=final_cov)
                pdf_normalised = pdf_vals / pdf_max

                self.total_mask[current_mask] += (pdf_normalised > 0.6065) #value 1sig
                self.point_prob[current_mask] = pdf_normalised 

        elif correlation == True:
            for iso_point_idx in points_to_check:
                mean = np.zeros(len(self.iso_colours) + 1)
                cov  = np.zeros((len(self.iso_colours) + 1, len(self.iso_colours) + 1))
                
                for idx, diff_colours in enumerate(list(self.iso_colours)):
                    mean[idx]     = self.iso_colours[diff_colours][iso_point_idx]
                mean[idx+1]       = self.iso_mags[self._mag_name][iso_point_idx]
                
                for idx, diff_mags in enumerate(unique_bands):
                    # cov[idx][idx] = np.sqrt(mask_size[idx]**2 + self.u_iso_mags[diff_mags][iso_point_idx]**2)
                    cov[idx][idx] = mask_size[idx]**2 + self.u_iso_mags[diff_mags][iso_point_idx]**2

                current_mask = (self.idx_closest_iso_point == iso_point_idx)
                temp_list   = []
                for input_axes in self._input_points:
                    temp_list.append(input_axes[current_mask])

                final_cov = np.matmul(jacobian.T, np.matmul(cov, jacobian))

                pdf_vals = multivariate_normal.pdf(x=np.array(temp_list).T, 
                                                mean=mean, cov=final_cov)
                pdf_max  = multivariate_normal.pdf(mean, mean=mean, cov=final_cov)
                pdf_normalised = pdf_vals / pdf_max

                self.total_mask[current_mask] += (pdf_normalised > 0.6065) #value 1sig
                self.point_prob[current_mask] = pdf_normalised 
        self._mask_size = mask_size

class IsochroneSelector():
    def __init__(self, distances: np.ndarray,
                isochrone_table: pd.DataFrame,
                isochrone_colour_mag: dict,
                input_stars_colour_mag: dict,
                paths_to_uncertainties: dict=None,
                mask_size: list=None,
                cmd_correlation: bool=True):
        '''
        Class defining a isochrone selector object used to determine probabilities.

        Parameters
        -------
        distances : np.float64
            Distance shift of the input isochrone in kpc.
            Should be given as an array, e.g.: np.array([10., 20.])

        isochrone_table : pd.DataFrame
            Pandas data frame of containing the absolute magnitudes of the input isochrone

        isochrone_colour_mag : dict
            Dictionary listing the names of the columns of 'isochrone_table'. For instance,
            if working in the 3D (multi-)colour-magnitude space (g-r, u-g, g):
                {"g-r": ["gmag", "rmag"], 
                "u-g": ["umag", "gmag"],
                "g":   ["gmag"]}
            Note: the keys of the dictionary must be the same as the keys of 
                'input_stars_colour_mag'

        input_stars_colour_mag : dict
            Dictionary containing the data for the stars of interest. If working in the 3D 
            (multi-)colour-magnitude space (g-r, u-g, g):
                {"g-r": array_g_r_data, 
                "u-g": array_u_g_data,
                "g":   array_g_data}

        paths_to_uncertainties : dict
            Dictionary containing the paths to the uncertainties. For instance, if working 
            in the 3D (multi-)colour-magnitude space (g-r, u-g, g):
                {"gmag": "./g_mag.csv",
                 "rmag": "./r_mag.csv",
                 "umag": "./u_mag.csv"}
            Note: their order doesn't matter, but each key should correspond to one column 
                of the input 'isochrone_table'

        mask_size : list
            Sizes of additional scatter added to the mask in each of the different 
            dimensions. For instance, if working in the 3D (multi-)colour-magnitude space 
            (g-r, u-g, g), supposing we want all scatter to be the same [0.1, 0.1, 0.1]
            Note: the number of elements must match the number of dimension of the (multi-)
                colour-magnitude space

        cmd_correlation : bool
            If true, accounts for correlations in the CMD
        '''
        self.distances              = distances
        self.isochrone_table        = isochrone_table
        self.isochrone_colour_mag   = isochrone_colour_mag
        self.input_stars_colour_mag = input_stars_colour_mag
        self.paths_to_uncertainties = paths_to_uncertainties
        self.mask_size              = mask_size
        self.cmd_correlation        = cmd_correlation

    #SINGLE______
    def masks_and_probs(self):
        masks = {}
        probs = {}

        for dist in self.distances:
            iso_obj = Isochroner(
                self.isochrone_table, 
                self.isochrone_colour_mag, dist
            )
            
            iso_obj._mask_maker(
                input_stars_colour_mag=self.input_stars_colour_mag, 
                paths_to_uncertainties=self.paths_to_uncertainties, 
                mask_size=self.mask_size, 
                correlation=self.cmd_correlation
            )
            
            masks[str(dist)] = iso_obj.total_mask
            probs[str(dist)] = iso_obj.point_prob

        return masks, probs
        
    #MULTIPROCESS___________________________________________________________________
    @staticmethod
    def _get_pool(n_jobs=None):
        '''
        Used by masks_and_probs_multiprocessing
        '''
        global _pool
        if _pool is None:
            if n_jobs is None:
                n_jobs = max(1, mp.cpu_count() - 1)
            _pool = mp.Pool(processes=n_jobs)
        return _pool
    
    @staticmethod
    def _close_pool():
        '''
        Used by masks_and_probs_multiprocessing
        '''
        global _pool
        if _pool is not None:
            _pool.terminate()   
            _pool.join()        
            _pool = None

    @staticmethod
    def _launcher_multiprocessing(args):
        '''
        Used by masks_and_probs_multiprocessing
        '''
        dist, isochrone_table, isochrone_colour_mag, input_stars_colour_mag, paths_to_uncertainties, mask_size, cmd_correlation = args
        iso_obj = Isochroner(
            isochrone_table, 
            isochrone_colour_mag, 
            dist
        )
        
        iso_obj._mask_maker(
            input_stars_colour_mag=input_stars_colour_mag,
            paths_to_uncertainties=paths_to_uncertainties, 
            mask_size=mask_size,
            correlation=cmd_correlation
        )
        
        return iso_obj.total_mask, iso_obj.point_prob, dist
    
    def masks_and_probs_multiprocessing(self, n_jobs: int=None):
        '''
        Provides a mask and probabilities to stars from the object

        Parameters
        -------
        n_jobs : int
            Number of threads on which the code will be ran, the default is the number 
            of cores on the machine -1

        Returns
        -------
        masks : dict
            Dictionary masking sources for which the probability to belong to the isochrone
            is lower than 68%. The keys correspond to the input distances (in kpc)
        
        probs : dict
            Dictionary providing the probability of a source to belong to the input 
            isochrone. The keys correspond to the input distances (in kpc)
        '''
        if n_jobs is None:
            n_jobs = max(1, mp.cpu_count() - 1)  # number of cores on machine minus 1

        masks = {}
        probs = {}

        args_list = [
            (dist, self.isochrone_table, self.isochrone_colour_mag,
            self.input_stars_colour_mag, self.paths_to_uncertainties, 
            self.mask_size, self.cmd_correlation)
            for dist in self.distances
        ]

        pool = self._get_pool(n_jobs)
        results = pool.map(self._launcher_multiprocessing, args_list)

        for mask, prob, dist in results:
            masks[str(dist)] = mask
            probs[str(dist)] = prob

        return masks, probs

    #JOBLIB_________________________________________________________________________
    @staticmethod
    def _launcher_joblib(args): 
        '''
        Used by masks_and_probs_joblib
        '''
        dist, isochrone_table, isochrone_colour_mag, input_stars_colour_mag, paths_to_uncertainties, mask_size, cmd_correlation= args
        iso_obj = Isochroner(
            isochrone_table, 
            isochrone_colour_mag, dist
        )
        
        iso_obj._mask_maker(
            input_stars_colour_mag=input_stars_colour_mag,
            paths_to_uncertainties=paths_to_uncertainties, 
            mask_size=mask_size,
            correlation=cmd_correlation
        )
        
        return iso_obj.total_mask, iso_obj.point_prob, dist 

    def masks_and_probs_joblib(self, n_jobs: int=None):
        '''
        Provides a mask and probabilities to stars from the object

        Parameters
        -------
        n_jobs : int
            Number of threads on which the code will be ran, the default is the number 
            of cores on the machine -1

        Returns
        -------
        masks : dict
            Dictionary masking sources for which the probability to belong to the isochrone
            is lower than 68%. The keys correspond to the input distances (in kpc)
        
        probs : dict
            Dictionary providing the probability of a source to belong to the input 
            isochrone. The keys correspond to the input distances (in kpc)
        '''
        if has_joblib == True:
            if n_jobs is None: n_jobs = max(1, joblib.cpu_count() - 1) #nb cores on machine 
            masks = {} 
            probs = {} 

            results = Parallel(n_jobs=n_jobs)(
            delayed(self._launcher_joblib)(
                (dist, self.isochrone_table, self.isochrone_colour_mag,
                self.input_stars_colour_mag, self.paths_to_uncertainties, 
                self.mask_size, self.cmd_correlation)
            )
            for dist in self.distances)
            
            for mask, prob, dist in results:
                masks[str(dist)] = mask
                probs[str(dist)] = prob

            return masks, probs

        else: 
            print("The 'joblib' pacakge is not installed, please consider doing so, else use 'masks_and_probs_multiprocessing' function")