import numpy as np
import matplotlib.pyplot as plt

class KernelGenerator():
    '''
    This class allows one to generate different kind of kernels, namely:
        - gaussian_kernel
        - annulus_kernel
    
    Once a kernel is generated, the user can easily check its shape using the 
    'Visualiser' method
    '''
    @staticmethod  
    def gaussian_kernel(sig_x: int, sig_y: int, angle: float=0):
        '''
        This method creates a kernel with the shape of a gaussian
        
        Parameters
        ----------
        sig_x, sig_y : int
            Standard deviations of the kernel to be generated (in pixels)
        
        angle : float
            The angle of the bivariate kernel with respect to the x-axis 
            (counterclockwise)

        Returns
        -------
        X, Y, Z : np.ndarrays 
            They are 2D numpy arrays storing the (xi, eta) positions and the kernel 
            "value". Their dimensions goes to (2*3sig by 2*3sig)
        '''
        if sig_x > sig_y:
            kernel_size = 3*sig_x
        elif sig_x < sig_y:
            kernel_size = 3*sig_y
        elif sig_x == sig_y:
            kernel_size = 3*sig_x

        Z = np.zeros((2*kernel_size + 1, 2*kernel_size + 1), dtype=int)
        X = np.zeros(np.shape(Z), dtype=int)
        Y = np.zeros(np.shape(Z), dtype=int)

        for i in range(2*kernel_size + 1):
            for j in range(2*kernel_size + 1):
                X[i][j] = i - 2*kernel_size//2
                Y[i][j] = j - 2*kernel_size//2

        angle_rad = np.radians(-angle)
        rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                    [np.sin(angle_rad),  np.cos(angle_rad)]])
        xy = np.vstack((X.flatten(), Y.flatten()))
        rotated_xy = np.dot(rotation_matrix, xy)
        rotated_x  = rotated_xy[0]   ;   rotated_y = rotated_xy[1]

        Z = (1/(2 * np.pi * sig_x * sig_y)) * np.exp(-0.5 * (((rotated_x) / sig_x)**2 + ((rotated_y) / sig_y)**2))
        Z = Z.reshape(np.shape(X))

        return Z, X, Y

    @staticmethod        
    def annulus_kernel(inner_radius:int, outer_radius:int):
        '''
        This method creates a kernel with the shape of an annulus
        
        Parameters
        ----------
        inner_radius, outer_radius : int
            Inner and outer radius of the annulus (given in units of pixels)

        Returns
        -------
        X, Y, Z : np.ndarrays 
            They are 2D numpy arrays storing the (xi, eta) positions and the kernel 
            values. The kernel values in the case of the annulus are either 0 or 1
        '''
        kernel_mid = 2*int((inner_radius)) + 1
        kernel_max = 2*int((outer_radius)) + 1
        middle_ind = kernel_max//2

        Z = np.zeros((kernel_max, kernel_max), dtype=int)
        X = np.zeros(np.shape(Z), dtype=int)
        Y = np.zeros(np.shape(Z), dtype=int)
        
        for i in range(kernel_max):
            for j in range(kernel_max):
                X[i][j] = i - middle_ind
                Y[i][j] = j - middle_ind

        for i in range(kernel_max):
            for j in range(kernel_max):
                if (X[i][j]**2 + Y[i][j]**2 >= (kernel_mid//2)**2
                  ) & (X[i][j]**2 + Y[i][j]**2 < (kernel_max//2)**2):
                    Z[i][j] = 1

        return Z, X, Y
                    
    @staticmethod
    def visualiser(Z: np.ndarray, X: np.ndarray, Y: np.ndarray, unit: str='deg'):
        '''
        This allows one to easily visualise the kernels from the class
        
        Parameters
        ----------
        Z, X, Y : np.ndarrays
            They are 2D numpy arrays storing the (xi, eta) positions and the kernel
            "value". They are the returns of the different xKernel methods

        unit : str
            Three different units for the pixels: "deg", "arcmin", "arcsec"

        Returns
        -------
        A matplotlib plot to visualise the kernel
        '''
        fig, ax = plt.subplots(figsize=(3,3), dpi=121)

        if unit == 'deg':
            pcm = ax.pcolormesh(X/60, Y/60, Z, cmap="Greys", rasterized=True)
            ax.set_xlabel(r'$\xi$ (deg)')   ;   ax.set_ylabel(r'$\eta$ (deg)')

        if unit == 'arcmin':
            pcm = ax.pcolormesh(X, Y, Z, cmap="Greys", rasterized=True)
            ax.set_xlabel(r'$\xi$ (arcmin)')   ;   ax.set_ylabel(r'$\eta$ (arcmin)')

        if unit == 'arcsec':
            pcm = ax.pcolormesh(X*60, Y*60, Z, cmap="Greys", rasterized=True)
            ax.set_xlabel(r'$\xi$ (arcsec)')   ;   ax.set_ylabel(r'$\eta$ (arcsec)')

        ax.grid(ls = (0, (5, 5)), dash_capstyle='round', c='black', 
                zorder=100, alpha=0.1)
        plt.show()