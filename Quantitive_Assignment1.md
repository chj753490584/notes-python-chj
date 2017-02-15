#Assignment 1
pls produce exactly the same chart as in the csi slide in Lecture 1 and the rollingSig.csv given to you,
i.e. the each month volatility of the csi 300.


```python
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm, lognorm, gaussian_kde, t, laplace

#read data
csi = pd.read_csv('F:\CUHK\\700HK.csv', index_col=0)
dy = csi.Tencent
dy.index = pd.to_datetime(dy.index)
dyRet = (dy / dy.shift()).map(lambda x: np.log(x))

#create an empty dictionary
v = {}
y = np.array(csi.Tencent)
yGRet = y[1:]/y[:-1]
yRet = np.log(yGRet)
mu = np.average(yRet)
sig = np.std(yRet)

plt.subplot(221)
plt.hist(yRet, normed=True, bins=100)
distance = np.linspace(min(yRet),max(yRet))
plt.hold(True)
plt.plot(distance, norm.pdf(distance,mu,sig))
plt.xlabel('log return')
plt.ylabel('density')
plt.legend(loc="upper right")

plt.subplot(222)
plt.plot(distance, norm.pdf(distance,mu,sig), label='Normal density')
kernel = gaussian_kde(yRet)
plt.plot(distance,kernel(distance), label='Kernel density')
plt.legend(loc="upper right")

plt.subplot(223)
yNRet = (yRet-mu)/sig #standardization
distanceN = np.linspace(min(yNRet), max(yNRet))
plt.plot(distanceN, norm.pdf(distanceN,0,1), label='Normal density')
plt.plot(distanceN, t.pdf(distanceN, df=4), label='t-dist, df=4')
kernel = gaussian_kde(yNRet)
plt.plot(distanceN,kernel(distanceN), label='Kernel density')
plt.legend(loc="upper right")

plt.subplot(224)
plt.plot(distanceN, norm.pdf(distanceN,0,1), label='Normal density')
plt.plot(distanceN, t.pdf(distanceN, df=3), label='t-dist, df=3')
plt.plot(distanceN, laplace.pdf(distanceN), label='laplace-dist')
kernel = gaussian_kde(yNRet)
plt.plot(distanceN,kernel(distanceN), label='Kernel density')
plt.legend(loc="upper right")


year = np.unique(dy.index.year)
month = np.unique(dy.index.month)

for yi in year:
    for mi in month:
        try:
            temp = dyRet.ix[(dyRet.index.year == yi) & (dyRet.index.month == mi)]
            if len(temp)==0:
                break
            else:
                v[yi *100+ mi] = []
                vol = np.std(temp)
            v[yi *100+ mi].append(vol)

        except:
            continue

pd.DataFrame(v).to_csv('F:\CUHK\\rollingSig.csv')
```

