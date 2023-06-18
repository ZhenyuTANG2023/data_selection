import os
import numpy as np
from typing import Tuple, List


class ObsDataManager:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def _get_feat_file_name(epoch:int, folder:str):
        return os.path.join(folder, f"epoch_{epoch:0>3d}_feat.npy")

    @staticmethod
    def _get_feat_info_file_name(epoch:int, folder:str):
        return os.path.join(folder, f"epoch_{epoch:0>3d}_feat_info.npy")

    @staticmethod
    def _get_y_file_name(epoch:int, folder:str):
        return os.path.join(folder, f"epoch_{epoch:0>3d}_y.npy")
    
    @staticmethod
    def get_feat(epoch:int, folder:str) -> np.ndarray:
        return np.load(ObsDataManager._get_feat_file_name(epoch, folder))
    
    @staticmethod
    def get_feat_info(epoch:int, folder:str) -> Tuple[List[str], np.ndarray]:
        raw_feat_info:np.ndarray = np.load(ObsDataManager._get_feat_info_file_name(epoch, folder), allow_pickle=True)
        feat_name:List[str] = raw_feat_info[:, 0].tolist()
        feat_nelement:np.ndarray = raw_feat_info[:, 1].astype(int)
        return feat_name, feat_nelement
    
    @staticmethod
    def get_y(epoch:int, folder:str) -> np.ndarray:
        return np.load(ObsDataManager._get_y_file_name(epoch, folder))
    
    @staticmethod
    def save_feat(epoch:int, folder:str, feat:np.ndarray) -> None:
        os.makedirs(folder, exist_ok=True)
        np.save(ObsDataManager._get_feat_file_name(epoch, folder), feat)
    
    @staticmethod
    def save_feat_info(epoch:int, folder:str, feat_info:List[Tuple[str, int]]) -> None:
        os.makedirs(folder, exist_ok=True)
        np.save(ObsDataManager._get_feat_info_file_name(epoch, folder), feat_info, allow_pickle=True)
    
    @staticmethod
    def save_y(epoch:int, folder:str, y:np.ndarray) -> None:
        os.makedirs(folder, exist_ok=True)
        np.save(ObsDataManager._get_y_file_name(epoch, folder), y)