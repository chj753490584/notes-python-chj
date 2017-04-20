import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt

data_0912=pd.read_csv('C:\Users\chj75\Documents\python\project\\final_project\csv\\0912price.csv',encoding='gbk',index_col=0)
data_1217=pd.read_csv('C:\Users\chj75\Documents\python\project\\final_project\csv\\1217_tp_delet.csv',index_col=0)
gfactor=pd.read_csv('C:\Users\chj75\Documents\python\project\\final_project\csv\\gfactor.csv',index_col=0)
vfactor=pd.read_csv('C:\Users\chj75\Documents\python\project\\final_project\csv\\vfactor.csv',index_col=0)
R=pd.read_csv('C:\Users\chj75\Documents\python\project\\final_project\csv\\ExpectedStockPrice.csv',index_col=0)
CSI=pd.read_csv('C:\Users\chj75\Documents\python\project\\final_project\csv\\CSI300.csv',index_col=0)


st_code=[]

#处理ST数据
for col in data_0912:
    if col.find('ST')!=-1:
        st_code.append(data_0912[col][0])
        del data_0912[col]

gfactor=gfactor.drop(st_code)
vfactor=vfactor.drop(st_code)
#ST数据处理完毕

#计算R
temp={}
for i in R.index:
    temp[i]=R.loc[i].mean()
for ii in temp.keys():
    if math.isnan(temp[ii]):
        temp.pop(ii)
R=pd.Series(temp)
for i in st_code:
    try:
        R=R.drop(i)
    except:
        continue
R_index=R.index

#整理出index
index_data0912=[]
for col in data_0912:
    index_data0912.append(data_0912[col][0])
index_data1217=list(data_1217.columns)

#融合去除st_code,所有data_0912,有R数据的股票的，一起的index code
#final index
final_index0912=np.array(list(set(gfactor.index) & set(R_index) & set(index_data1217)))



#factor排序,排序是升序
gfactor_sortindex={}#排序出来的字典 of index
for gfactor_col in gfactor.columns:
    x=gfactor[gfactor_col]
    x=x.sort_values()
    gfactor_sortindex[gfactor_col]=x.index

vfactor_sortindex={}
for vfactor_col in vfactor.columns:
    x=vfactor[vfactor_col]
    x=x.sort_values()
    vfactor_sortindex[vfactor_col]=x.index
#factor 排序完成

#计算09-12年所有股票的收益率
data_0912.index=pd.to_datetime(data_0912.index,format='%Y/%m/%d')
Ret_0912=np.array(data_0912)[1:]
m=np.zeros(data_0912.shape[0]-1)
m[0]=-1
m[-1]=1
temp=[]
for x in Ret_0912:
    for y in x:
        y=float(y)
        temp.append(y)
temp=np.array(temp).reshape(-1,data_0912.shape[1])
temp_1=temp[0]
temp_2=[]
temp_3=[]
Ret_0912=np.dot(m,temp)
for x in temp_1:
    x=1/x
    temp_2.append(x)
for i in range(data_0912.shape[1]):
    x=temp_2[i]*Ret_0912[i]
    temp_3.append(x)
Ret_0912=pd.Series(temp_3,index=index_data0912)

#计算12-17年所有股票的收益率
data_1217.index=pd.to_datetime(data_1217.index,format='%Y/%m/%d')
Ret_1217=np.array(data_1217)[1:]
m=np.zeros(data_1217.shape[0]-1)
m[0]=-1
m[-1]=1
temp=[]
for x in Ret_1217:
    for y in x:
        y=float(y)
        temp.append(y)
temp=np.array(temp).reshape(-1,data_1217.shape[1])
temp_1=temp[0]
temp_2=[]
temp_3=[]
Ret_1217=np.dot(m,temp)
for x in temp_1:
    x=1/x
    temp_2.append(x)
for i in range(data_1217.shape[1]):
    x=temp_2[i]*Ret_1217[i]
    temp_3.append(x)
Ret_1217=pd.Series(temp_3,index=index_data1217)

#计算factor对应的收益率,排序是升序，所以后面的是收益好的
Ret_factor={}

for x in gfactor.columns:
    Ret_factor[x]=-float(np.dot(Ret_0912[gfactor_sortindex[x][0:303]],np.ones(303).reshape(-1,1)))+\
                  float(np.dot(Ret_0912[gfactor_sortindex[x][1514-303:1515]],np.ones(303).reshape(-1,1)))
for x in vfactor.columns:
    Ret_factor[x]=-float(np.dot(Ret_0912[vfactor_sortindex[x][0:303]],np.ones(303).reshape(-1,1)))+\
                  float(np.dot(Ret_0912[vfactor_sortindex[x][1514-303:1515]],np.ones(303).reshape(-1,1)))
efficient_factor=[]
for i in range(6,10):
    efficient_factor.append(pd.Series(Ret_factor).sort_values().index[i])

#按照efficient factor对股票进行打分
factor_price=pd.concat([gfactor, vfactor], axis=1)

factor_price_efficient=factor_price[efficient_factor].fillna(0)
grade={}
for i in factor_price_efficient.index:
    grade[i]=float(np.dot(np.array(factor_price_efficient.loc[i]),np.ones(4).reshape(-1,1)))


#建立portfolio,并调整为MSR可用格式
stock_150=np.array(pd.Series(grade).sort_values(ascending=False).index[0:80])
stock_150=list(set(stock_150) & set(final_index0912))#在这里将R存在的数据和已经做的数据进行合并
data_code0912=data_0912[1:]
data_code0912.columns=index_data0912
stock_150price=pd.DataFrame(data_code0912[stock_150])

#计算股票池的有效边界，即权重tangWGT
y_stock=np.array(stock_150price)
n=y_stock.shape[1]

temp=[]#数据中有unicode，都转变为float数据
for ii in y_stock:
    for i in ii:
        temp.append(float(i))
y_stock=np.array(temp).reshape(-1,len(stock_150))
temp=[]#数据中有unicode，都转变为float数据
for ii in np.array(stock_150price):
    for i in ii:
        temp.append(float(i))
stock_150price=np.array(temp).reshape(-1,len(stock_150))

yRet=np.log(y_stock[1:]/y_stock[:-1])
omega = np.cov(yRet, rowvar=False)
invOmega = np.linalg.inv(omega)

#设定Rf和最终的R
R=R[stock_150]
Rf=3.7/100
I = np.ones(len(stock_150)).reshape(-1, 1)
tangWgt = np.dot(invOmega, R - Rf) / np.dot(I.transpose(), np.dot(invOmega, R - Rf))
#porfolio 已经建完


#计算收益率
stock_150price_1217=pd.DataFrame(data_1217[stock_150])
portfolio_price=float(np.dot(tangWgt,np.array(stock_150price[-2:-1]).reshape(-1,1)))
portfolio_price17=float(np.dot(tangWgt,np.array(data_1217[stock_150][-2:-1]).reshape(-1,1)))
tranfee_buy=portfolio_price*30/10000
tranfee_sell=portfolio_price17*30/10000
portfolio_yield_cumulative=(portfolio_price17-portfolio_price-tranfee_sell-tranfee_buy)/portfolio_price
portfolio_yield_annual=(1 + portfolio_yield_cumulative) ** ( 1.0 / 4.3 ) - 1


#计算CSI收益率
CSI_price=np.array(CSI)
CSI_number=100000000/CSI_price[0]
CSI_portfolio=CSI_number * CSI_price

#画图
portfolio=np.dot(tangWgt,stock_150price_1217.T)
portfolio_number=100000000/portfolio[0]
portfolio=portfolio_number * portfolio
plt.plot(data_1217.index[0:len(portfolio)],list(portfolio),linewidth=3,label=portfolio,color='cornflowerblue')
plt.plot(data_1217.index[0:len(portfolio)],CSI_portfolio,linewidth=3,label=CSI,color='grey')
plt.show()

pd.DataFrame(portfolio).to_csv('C:\Users\chj75\Documents\python\project\\final_project\csv\\portfolio.csv')

#计算参数
CSI_std=np.std(CSI_portfolio) ** (1.0 / 4.3)
CSI_Maxdrawdown=(2.304-1.309)/2.304
CSI_yield_cumulative=(CSI_price[-1]-CSI_price[0])/CSI_price[0]
CSI_yield_annual=(1 + CSI_yield_cumulative) ** ( 1.0 / 4.3 ) - 1
CSI_CR= CSI_yield_annual / CSI_Maxdrawdown

portfolio_std=np.std(portfolio) ** (1.0 / 4.3)
portfolio_Maxdrawdown=(2.961-1.719)/2.961
portfolio_CR=portfolio_yield_annual/portfolio_Maxdrawdown
