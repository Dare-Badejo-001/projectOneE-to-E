import os 
import sys
import logging  

import pandas as pd
import numpy as np

from src.exception import CustomException
import dill

def save_object(file_path, obj ):
    '''
    This function saves the object to a file using dill.
    '''
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e, sys) 
