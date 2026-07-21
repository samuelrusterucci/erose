import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

class CoordinateTransformer():
    @staticmethod
    def gal_to_icrs(xcoord: np.ndarray, ycoord: np.ndarray):
        '''
        Transforms the galactic (l, b) coordinates to the ICRS (ra, dec) using astropy
        
        Parameters
        ----------
        xcoord, ycoord : np.ndarrays
            Input numpy arrays containing the (l, b) coordinates

        Returns
        -------
        icrs_coord.ra.deg, icrs_coord.dec.deg : np.ndarrays
            Output numpy arrays containing the (ra, dec) coordinates
        '''
        gal_coord  = SkyCoord(l=xcoord*u.deg, b=ycoord*u.deg, frame='galactic')
        icrs_coord = gal_coord.icrs

        return icrs_coord.ra.deg, icrs_coord.dec.deg

    @staticmethod
    def icrs_to_gal(xcoord, ycoord):
        '''
        Transforms the ICRS (ra, dec) coordinates to the galacitc (l, b) using astropy
        
        Parameters
        ----------
        xcoord, ycoord : np.ndarrays
            Input numpy arrays containing the (ra, dec) coordinates

        Returns
        -------
        gal_coord.l.deg, gal_coord.b.deg : np.ndarrays
            Output numpy arrays containing the (l, b) coordinates
        '''
        icrs_coord = SkyCoord(ra=xcoord*u.deg, dec=ycoord*u.deg, frame='icrs')
        gal_coord  = icrs_coord.galactic

        return gal_coord.l.deg, gal_coord.b.deg

    @staticmethod
    def sphere_to_plane(phi: np.ndarray, theta: np.ndarray, 
                        phi_c: float, theta_c: float): 
        '''
        Project spherical coordinates onto a plane following a gnomonic projection
        More details: https://mathworld.wolfram.com/GnomonicProjection.html
        
        Parameters
        ----------
        phi, theta : np.ndarrays
            Input spherical coordinates (given in degrees)

        phi_c, theta_c : floats
            Central coordinates around which the projection should be made

        Returns
        -------
        xi, eta : np.ndarrays
            Output numpy arrays containing the (xi, eta) coordinates in degrees
        '''
        phi_rad   = np.deg2rad(phi) 
        theta_rad = np.deg2rad(theta)

        c_xdiff = np.cos(phi_rad - np.deg2rad(phi_c)) 
        s_xdiff = np.sin(phi_rad - np.deg2rad(phi_c))
        c_y     = np.cos(theta_rad)                  
        s_y     = np.sin(theta_rad)
        c_yc    = np.cos(np.deg2rad(theta_c))         
        s_yc    = np.sin(np.deg2rad(theta_c))

        denominator = s_yc*s_y + c_yc*c_y*c_xdiff
        mask_denom  = abs(denominator) < 0.0000001; denominator[mask_denom] = 0.0000001

        plane_x = (c_y*s_xdiff)/denominator       
        plane_y = (c_yc*s_y - s_yc*c_y*c_xdiff)/denominator

        return np.rad2deg(plane_x), np.rad2deg(plane_y)

    @staticmethod
    def plane_to_sphere(xi: np.ndarray, eta: np.ndarray, 
                        phi_c: float, theta_c: float): 
        '''
        Re-projects onto a sphere coordinates that have been projected onto a plane 
        following a gnomonic projection scheme
        More details: https://mathworld.wolfram.com/GnomonicProjection.html
        
        Parameters
        ----------
        xi, eta : np.ndarrays
            Input gnomonic coordinates in degrees

        phi_c, theta_c : floats
            Central spherical coordinates around which the projection should be made

        Returns
        -------
        np.rad2deg(sphere_x), np.rad2deg(sphere_y) : np.ndarrays
            Contains the re-projection onto a sphere given in degrees
        '''
        X_rad = np.deg2rad(xi)
        Y_rad = np.deg2rad(eta)

        rho  = np.sqrt(X_rad**2 + Y_rad**2)   
        c    = np.arctan(rho)
        c_c  = np.cos(c)                      
        s_c  = np.sin(c)
        c_yc = np.cos(np.deg2rad(theta_c))         
        s_yc = np.sin(np.deg2rad(theta_c))

        sphere_x = np.deg2rad(phi_c) + np.arctan2(X_rad*s_c, rho*c_yc*c_c - Y_rad*s_yc*s_c)
        sphere_y = np.arcsin(np.clip(c_c*s_yc + (Y_rad*s_c*c_yc)/rho, -1, 1))

        return np.rad2deg(sphere_x), np.rad2deg(sphere_y)

    @staticmethod
    def right_ascension_recentering(phi: np.ndarray, phi_c: float):
        '''
        Re-centres spherical data on 0 around the phi axis, this allows not having to 
        care about potential boundary effects when then projection or making selections
        
        Parameters
        ----------
        phi : np.ndarray
            Input array in spherical coordinates (e.g. right ascension), in degrees

        phi_c : floats
            Central point around which the later selection will be made, in degrees

        Returns
        -------
        np.rad2deg(x) : np.ndarray
            Returns the input 'phi' array but re-centred on 'phi_c'
        '''
        x  = np.deg2rad(phi)  
        xc = np.deg2rad(phi_c)

        x -= xc   ;   x = x%(2*np.pi)  ;  x = np.where(x < np.pi, x, x - 2*np.pi)

        return np.rad2deg(x)

    @staticmethod
    def spheretoplane(lon: np.ndarray,    lat: np.ndarray,
                  central_lon: float, central_lat: float): 
        '''
        Transforms spherical coordinates into a plane through Gnomonic Projection. 
        For details: https://mathworld.wolfram.com/GnomonicProjection.html

        Note: similar to the 'sphere_to_plane' method but does the longitude recentering
              automatically
        
        Parameters
        ----------
        lon, lat : np.ndarray
            Longitude and latitude in degrees

        central_lon, central_lat : float
            Central point around which the projection should be made

        Returns
        -------
        xi, eta : np.ndarray
            Arrays being the plane projections of the inputs
        '''
        X_rad = np.deg2rad(lon) ; Y_rad = np.deg2rad(lat)

        X_rad -= np.deg2rad(central_lon) ; X_rad = X_rad%(2*np.pi) 
        X_rad = np.where(X_rad < np.pi, X_rad, X_rad - 2*np.pi) 

        c_xdiff = np.cos(X_rad)           ;  s_xdiff = np.sin(X_rad)
        c_y     = np.cos(Y_rad)           ;  s_y     = np.sin(Y_rad)
        c_yc    = np.cos(np.deg2rad(central_lat))  ;  s_yc    = np.sin(np.deg2rad(central_lat))

        denominator = s_yc*s_y + c_yc*c_y*c_xdiff
        mask_denom  = abs(denominator) < 0.0000001; denominator[mask_denom] = 0.0000001

        plane_x = (c_y*s_xdiff)/denominator       ;  plane_y = (c_yc*s_y - s_yc*c_y*c_xdiff)/denominator

        return np.rad2deg(plane_x), np.rad2deg(plane_y)
    
    @staticmethod
    def cone_search(ra_catalogue: np.ndarray, de_catalogue: np.ndarray,
                    ra_centre:         float, de_centre:         float, 
                    radius:            float, ang_dist:     bool=False):
        '''
        Makes a conesearch on an input catalogue.
        
        Parameters
        ----------
        ra_catalogue, de_catalogue : np.ndarray, np.ndarray
            Arrays of the ra and dec of the input catalogue (in deg)

        ra_centre, de_centre : float, float
            Centre of the conesearch (in deg)

        radius : float
            Radius of the conesearch (in arcmin)

        Returns
        -------
        final_mask : pandas.core.series.Series
            Mask of booleans selecting the stars within the conesearch

        ang_dist_array : np.ndarray
            Array of the distances to the centre of the conesearch
        '''
        x_temp = CoordinateTransformer.right_ascension_recentering(ra_catalogue, 
                                                                   ra_centre)
        y_temp = de_catalogue

        pre_mask = (x_temp**2 + (y_temp - de_centre)**2) < (5 * radius / 60)**2

        xi, eta = CoordinateTransformer.sphere_to_plane(x_temp[pre_mask], 
                                                        y_temp[pre_mask],
                                                        0, de_centre)

        final_mask = np.zeros_like(ra_catalogue, dtype=bool)
        final_mask[pre_mask] = (xi**2 + eta**2) < (radius / 60)**2

        if ang_dist == False:
            return final_mask
        
        else:
            ang_dist_array = np.sqrt(xi**2 + eta**2)[(xi**2 + eta**2) < (radius / 60)**2]
            return final_mask, ang_dist_array