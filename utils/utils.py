import numpy as np
from PIL import Image

def photometric_error(img1: np.ndarray, img2: np.ndarray) -> float:
    img1 = img1[:, :, :3]
    img2 = img2[:, :, :3]
    error = (img1 - img2 + 0.0)**2
    error = np.sum(error)/float(img1.shape[0]*img1.shape[1])
    return error


def load_final_image(path:str):
    try:
        final_img = Image.open(path).convert("RGB")
    except FileNotFoundError:
        print(f"========OUTPUT FILE {path} NOT FOUND======")
        return None
    final_img:np.ndarray = np.array(final_img)
    return final_img
