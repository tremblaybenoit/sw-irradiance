from iti.data.editor import LoadMapEditor, NormalizeRadiusEditor, RemoveOffLimbEditor, AIAPrepEditor
import numpy as np

def loadAIAMap(file_path, resolution=1024, remove_off_limb=False):
    """Load and preprocess AIA file.


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
        s_map = RemoveOffLimbEditor(fill_value=np.nan).call(s_map)
    return s_map