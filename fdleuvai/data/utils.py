from iti.data.editor import LoadMapEditor, NormalizeRadiusEditor, RemoveOffLimbEditor, AIAPrepEditor
import numpy as np

def loadAIAMap(file_path, resolution=1024, remove_off_limb=False, off_limb_val=np.nan):
    """Load and preprocess AIA file to make them compatible to ITI.


    Parameters
    ----------
    file_path: path to the FTIS file.
    resolution: target resolution in pixels of 2.2 solar radii.
    remove_off_limb: set all off-limb pixels to NaN (optional).

    Returns
    -------
    the preprocessed SunPy Map
    """
    s_map, _ = LoadMapEditor().call(file_path)
    s_map = NormalizeRadiusEditor(resolution).call(s_map)
    s_map = AIAPrepEditor(calibration='auto').call(s_map)
    if remove_off_limb:
        s_map = RemoveOffLimbEditor(fill_value=off_limb_val).call(s_map)
    return s_map


def loadAIAStack(file_paths, resolution=1024, remove_off_limb=False, off_limb_val=np.nan, remove_nans=False):
    """Load a stack of AIA files, preprocess them at a specfied resolution, and stackt hem.


    Parameters
    ----------
    file_paths: list of files to stack.
    resolution: target resolution in pixels of 2.2 solar radii.
    remove_off_limb: set all off-limb pixels to NaN (optional).

    Returns
    -------
    numpy array with AIA stack
    """

    aia_stack = np.asarray([np.expand_dims(loadAIAMap(aia_file, resolution=resolution, remove_off_limb=remove_off_limb, off_limb_val=off_limb_val).data, axis=0) for aia_file in file_paths], dtype = np.float32)

    if remove_nans:
        aia_stack[np.isnan(aia_stack)] = 0

    return aia_stack