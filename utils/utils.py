import numpy as np
from PIL import Image

def photometric_error(img1: np.ndarray, img2: np.ndarray) -> float:
    assert img1.shape == img2.shape
    img1 = img1[:, :, :3].astype('float')
    img2 = img2[:, :, :3].astype('float')
    error = (img1 - img2 )**2
    error = error.sum()/(img1.shape[0]*img1.shape[1])
    return error


def load_final_image(path:str):
    try:
        final_img = Image.open(path).convert("RGB")
    except FileNotFoundError:
        print(f"========OUTPUT FILE {path} NOT FOUND======")
        return None
    final_img:np.ndarray = np.array(final_img)
    return final_img
