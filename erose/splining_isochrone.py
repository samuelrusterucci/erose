import numpy as np
import pandas as pd
import re

from scipy.interpolate import CubicSpline

def open_parsec(file_name: str):
    '''
    Simply opens a Parsec file by providing its name/path
    
    Parameters
    ----------
    file_name : str
        Path/name of the PARSEC file

    Returns
    -------
    parsec_file : pd.dataframe
        All points from the PARSEC file isochrone

    evolutionary_stage : tuple(np.ndarray)
        Masks of the different stages of the life of stars
        0=PMS (pre-main sequence)
        1=MS (main sequence)
        2=SGB (subgiant branch)
        3=RGB (red giant branch)
        (4,5,6)=different stages of CHEB (core-helium burning)
        7=EAGB (early asymptotic giant branch)
        8=TP-AGB (thermally pulsing AGB)
        9=post-AGB
    '''
    #Read all the lines in the PARSEC file
    with open(file_name) as f:
        lines = f.readlines()

    #Finds the line containing the column names
    condition = False 
    line_nb   = 0
    while (condition == False):
        x = re.search("# Zini", lines[line_nb]) 
        if (x is None) == True: line_nb+=1
        else: condition = True

    #Column names to open the file
    column_names = lines[line_nb].split()[1:]
    
    parsec_file = pd.read_csv(file_name, sep='\s+', comment='#', 
                              header=None, names=column_names)
    
    return parsec_file

def making_splines(input_magnitudes: np.ndarray, points_labels: np.ndarray=None,
                  labels_to_treat: list=None, nb_steps: int=10):
    '''
    Function linking input points according to cubic splines using the SciPy 
    CubicSpline class 
    
    Parameters
    ----------
    input_magnitudes : np.ndarray
        Array of the input magnitudes to be splined

    points_labels : np.ndarray
        The user can provide labels to the input points, such as 1 for main sequence or
        7 for early asymptotic giant branch. This is made so these star stages may be 
        treated independently

    labels_to_treat : list
        List of lists allowing to choose the different stages of the life of stars and 
        treat them independently when doing the spline fitting. In PARSEC the stages are
        labelled as follows:
            0=PMS (pre-main sequence)
            1=MS (main sequence)
            2=SGB (subgiant branch)
            3=RGB (red giant branch)
            (4,5,6)=different stages of CHEB (core-helium burning)
            7=EAGB (early asymptotic giant branch)
            8=TP-AGB (thermally pulsing AGB)
            9=post-AGB
        If one only wants to keep the MS and EAGB, but fitting them independently in the
        spline function, labels_to_treat = [ [1], [7] ]

    nb_steps : int
        Number of points fitted in between the input points

    Returns
    -------
    output_magnitude : np.ndarray
        Array of the splined magnitudes
    '''
    if points_labels is None:
        mask = {'single': np.ones(len(input_magnitudes), dtype=bool)}

    else:
        mask = {}
        for label_name in labels_to_treat:
            mask[str(label_name)] = (points_labels == label_name)

    mask_temp = np.zeros(len(input_magnitudes), dtype=bool)

    for _, (_, val) in enumerate(mask.items()):
        mask_temp += val

    output_magnitude = np.array([], dtype=np.float64)

    # Create a strictly increasing parameter
    input_param = np.linspace(
        0, 1, len(input_magnitudes[mask_temp])
    )

    #Creates the parameter for plotting the splines
    output_param = np.linspace(
        0, 1, len(input_magnitudes[mask_temp])*nb_steps
    )

    cs = CubicSpline(input_param, np.array(input_magnitudes[mask_temp]))
    output_magnitude = np.concatenate((output_magnitude, cs(output_param)))
    
    return output_magnitude

def open_and_spline_parsec(file_name: str, labels_to_treat: list=[[1,2,3], [4,7]],
                        nb_steps: int=10):
    '''
    Wraps the "open_parsec" and the "making_splines" functions in the special
    case of PARSEC isochrones
    
    Parameters
    ----------
    file_name : str
        Path/name of the input PARSEC file

    labels_to_treat : list
        List of lists allowing to choose the different stages of the life of stars and 
        treat them independently when doing the spline fitting. In PARSEC the stages are
        labelled as follows:
            0=PMS (pre-main sequence)
            1=MS (main sequence)
            2=SGB (subgiant branch)
            3=RGB (red giant branch)
            (4,5,6)=different stages of CHEB (core-helium burning)
            7=EAGB (early asymptotic giant branch)
            8=TP-AGB (thermally pulsing AGB)
            9=post-AGB
        If one only wants to keep the MS and EAGB, but fitting them independently in the
        spline function, labels_to_treat = [ [1], [7] ]

    nb_steps : int
        Number of points fitted in between the input points
        
    Returns
    -------
    final_table : pd.dataframe
        Same as the input PARSEC file but splined and only containing the magnitudes
    '''
    parsec_iso = open_parsec(file_name=file_name)

    cols_mag = [col for col in parsec_iso.columns if col.endswith("mag")]

    #Initialisation of the table
    final_table = pd.DataFrame()
        
    for labels in labels_to_treat:
        temp = pd.DataFrame()
        for column in cols_mag:    
            temp[column] = making_splines(
                np.array(parsec_iso[column]), 
                np.array(parsec_iso['label']), labels, nb_steps
            )    
               
        final_table = pd.concat((final_table, temp))

    return final_table.reset_index(drop=True)