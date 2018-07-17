#!/usr/bin/env python
# encoding:utf-8

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR, DynamicVAR
mdata = sm.datasets.macrodata.load_pandas().data
dates = mdata[["year", "quarter"]].astype(int).astype(str)

quarterly = dates["year"] +"Q" + dates["quarter"]

from statsmodels.tsa.base.datetools import dates_from_str
quarterly = dates_from_str(quarterly)
mdata = mdata[["realgdp", "realcons", "realinv"]]
mdata.index = pd.DatetimeIndex(quarterly)
data = np.log(mdata).diff().dropna()
model = VAR(data)

results = model.fit(2)
print(results.summary())
