from src.data import Data
import pandas as pd
from pandas._testing import assert_frame_equal
import numpy as np

def test_preprocess():
    df = pd.DataFrame(data={"Date": ["2020", "2021", "2022", "2013"], "new cases": [1, 200, -1, np.nan,]})
    data = Data(df, 1, 1)
    data.preprocess()
    df_expected = pd.DataFrame(data={"Date": ["2020", "2021",], "new cases": [1, 200,]})
    assert_frame_equal (data.df, df_expected, check_dtype=False)
    