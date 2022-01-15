import pandas as pd
import eikon as ek
import numpy as np
import data_analysis_support
from tqdm.notebook import tqdm

# Set Eikon App Key
ek.set_app_key('DEFAULT_CODE_BOOK_APP_KEY')

# Function parameters
# -------------------
# Eikon Index RIC
index_ric = '.SPX' 
# Corporate yeaild to be used as discounting yeild
corporate_yeild = 0.06
# Discount rate 
discount_rate = 0.20



# Create download class
data_analysis = data_analysis_support.DataAnalysis(index_ric = index_ric, 
                                                   corporate_yeild = corporate_yeild, 
                                                   discount_rate = discount_rate, 
                                                   intermidate_path = 'current_data')
# Start analysis
data_analysis.run_analysis()
