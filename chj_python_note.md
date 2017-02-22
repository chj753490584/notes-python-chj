#目录

[一、基本知识](#一基本知识)

[1.1 list、元组 和字典](#11-list元组-和字典)
  
[1.2 python 中的循环语句及函数](#12-python-中的循环语句及函数)
  
[1、 定义函数](#1-定义函数)
    
[2、 if语句](#2-if语句)
    
[3、 while语句](#3-while语句)
    
[4、 for 语句](#4-for-语句)
    
[5、列表推导式：轻量级循环](#5列表推导式轻量级循环)

[1.3 python 中的类](#13-python-中的类)
  
[二、pandas知识点](#二pandas知识点)

[2.1 Series](#21-series)

[1、 Series数据的访问](#1-series数据的访问)

#chj-python学习笔记
##一、基本知识
###1.1 list、元组 和字典
- 列表的基本形式比如：
  [1,3,6,10]或者[‘yes’,’no’,’OK’]
- 元组的基本形式比如：
(1,3,6,10)或者(‘yes’,’no’,’OK’)
- 字典： d={7:'seven',8:'eight',9:'nine'}
- 列表和元祖主要的区别是列表可以修改，元组不能。
- 取某一列，a[2]，或者c[0:3],或者c[1:],c[:-1]，即通过索引访问，**但是字典不可以通过字典访问，只能通过值和键的映射访问**
- list的重新赋值
```
b=list('hello')
b
['h', 'e', 'l', 'l', 'o']
b[2:4]=list('yy')
b
['h', 'e', 'y', 'y', 'o']
```
- list 的函数使用：列表.函数（）
```Python
a.insert(2,'t')#在第3个索引总插入t这个元素
['h', 'e', 't', 'l', 'l', 'o']
a.append('q')#给列表的最后添加元素m
['h', 'e', 't', 'l', 'l', 'o', 'q']
a.index('e')#返回a列表中，元素m第一次出现的索引位置
1
```
- 字典的操作
dict(参数1=值1,参数2=值2, …)={参数1:值1, 参数2=值2, …}
![Alt text](./1486652583230.png)

----
###1.2 python 中的循环语句及函数

####**1、 定义函数**

- def 函数名（参数）：输入代码

```python

def square(x):return x*x

 

square(9)

81

```

####**2、 if语句**

- **注意Python是用缩进来标识出哪一段属于本循环**

- 对于多条件，注意的是elseif要写成elif，标准格式为：

```python

if 条件1:

执行语句1

elif 条件2:

执行语句2

else:

 执行语句3

```

####**3、 while语句**

```python

a=3

while a<10:

    a=a+1

    print a

    if a==8: break

4

5

6

7

8

```

####**4、 for 语句**

可以遍历一个序列/字典等。

```python

a=[1,2,3,4,5]

for i in a:

    print i

```

####**5、列表推导式：轻量级循环**

列表推导式，是利用其它列表来创建一个新列表的方法，工作方式类似于for循环，格式为：

**[输出值 for 条件]**

当满足条件时，输出一个值，最终形成一个列表：

```python

[x*x for x in range(10)]

[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

 

[x*x for x in range(10) if x%3==0]

[0, 9, 36, 81]

```
----

###1.3 python 中的类

- 类可以用来定义静态的属性，和动态的方法

```python

class boy:

    gender='male'#定义属性

    interest='girl'#定义属性

    def say(self):#定义方法

        return 'i am a boy'

 

peter=boy()

peter.gender

'male'

peter.interest

'girl'

peter.say()

'i am a boy'

```
----

##二、pandas知识点
###2.1 Series
####**1、 Series数据的访问**
- 访问第几个（只有值，不包括索引）
```python
s = Series(np.random.randn(10),index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
s[0]
1.4328106520571824
```
或者
```python
s['a']
1.4328106520571824
```
- 访问某一个（包括索引和值）
```python
s[[-1]]
j   -0.837685
dtype: float64
```
或者
```python
s[['j']]
j   -0.837685
dtype: float64
```
- 访问某几个（包括索引和值）
```python
s[:2]
a    1.432811
b    0.120681
dtype: float64
```
或者
```python
s[[2,0,4]]
c    0.578146
a    1.432811
e    1.327594
dtype: float64
```
- 取某几个值
```python
s[s > 0.5]
a    1.432811
c    0.578146
e    1.327594
g    1.850783
dtype: float64
```
- 检验某一个索引是否在
```python
'e' in s
True
```
----

###2.2 DataFrame
**1、定义**
DataFrame是将数个Series按列合并而成的二维数据结构**类似于二维表格**，每一列单独取出来是一个Series，这和SQL数据库中取出的数据是很类似的。所以，按列对一个DataFrame进行处理更为方便，用户在编程时注意培养按列构建数据的思维。DataFrame的优势在于可以方便地处理不同类型的列，因此，就不要考虑如何对一个全是浮点数的DataFrame求逆之类的问题了，处理这种问题还是把数据存成NumPy的matrix类型比较便利一些。

**2、创建DataFrame**
方括号 [ ] ：创建list的时候用的，或者取某一个或几列的值，或者创建index的时候用到
[1,2,3]：列1,2,3
[[1,2,3]]：行1,2,3
圆括号 ( ) ：创建Series，DataFrame 或者元组的时候用的
大括号 { } ：只有创建字典的时候用到，或者用index置顶

- 第一种创建方法，**按照index和coloum创建其中的value，没有的填写nan**
```python
d = {'one': pd.Series([1., 2., 3.], index=['a', 'b', 'c']), 'two': Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])}
df = DataFrame(d)
print df

    one  two
a    1    1
b    2    2
c    3    3
d  NaN    4
```
- 第二种：方法是，**先创建其中的colums和值，在创建index，但是必须是一样的**
```python
d = {'one': [1., 2., 3., 4.], 'two': [4., 3., 2., 1.]}
df = DataFrame(d, index=['a', 'b', 'c', 'd'])
print df
   one  two
a    1    4
b    2    3
c    3    2
d    4    1
```
- 第三种：重新创建，**根据已有的DF中的Index和columns重新检索和创建**
```python
import pandas as pd
d = {'one':pd.Series([1., 2., 3.], index=['a', 'b', 'c']), 'two': pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])}
df = pd.DataFrame(d)
print df


   two three
r  NaN   NaN
d    4   NaN
a    1   NaN
```
因为r行和three列不存在，所以值为NaN

- 第四种：还有方法是，就是**使用concat函数基于Serie或者DataFrame创建一个DataFrame**
```python
a = Series(range(5))
b = Series(np.linspace(4, 20, 5))
df = pd.concat([a, b], axis=1)#axis=1表示按列进行合并，axis=0表示按行合并,并且，Series都处理成一列
print df

   0   1
0  0   4
1  1   8
2  2  12
3  3  16
4  4  20
```
```python
a = Series(range(5))
b = Series(np.linspace(4, 20, 5))
df = pd.concat([a, b], axis=0)#这里如果选axis=0的话，将得到一个10×1的DataFrame，因为series都是变成一列
print df
0     0.0
1     1.0
2     2.0
3     3.0
4     4.0
0     4.0
1     8.0
2    12.0
3    16.0
4    20.0
dtype: float64

```
另外有一个知识点就是，在DataFrame中如果用 [ ] 圈起来的一把当成一组数据，可以给一个index，在按照indexconcat。也可以不圈起来，一个个给index，在按照列concat
```python
df = DataFrame()
index = ['alpha', 'beta', 'gamma', 'delta', 'eta']
for i in range(5):
    a = DataFrame([np.linspace(i, 5*i, 5)], index=[index[i]])#给每一行一个index
    df = pd.concat([df, a], axis=0)
print df

       0  1   2   3   4
alpha  0  0   0   0   0
beta   1  2   3   4   5
gamma  2  4   6   8  10
delta  3  6   9  12  15
eta    4  8  12  16  20
```
上面的输出结果就是，一行一行的给数据，给index，在用concat组合，axis=0
```python
import numpy as np
import pandas as pd
df = pd.DataFrame()
for i in range(5):
    a = pd.Series(np.linspace(i, 5*i, 5), index = ['alpha', 'beta', 'gamma', 'delta', 'eta'])#给这一列的每一个元素，一个index
    df = pd.concat([df, a], axis=1)
print df#在按照index 合并

         0    0     0     0     0
alpha  0.0  1.0   2.0   3.0   4.0
beta   0.0  2.0   4.0   6.0   8.0
delta  0.0  4.0   8.0  12.0  16.0
eta    0.0  5.0  10.0  15.0  20.0
gamma  0.0  3.0   6.0   9.0  12.0
```
这里的输出结果就是，一个一个的给数据，给同一个index，按照列组合，axis=1

**！！！上面两个程序非常不同的原因就是，[np.linespace(1,5,5)]代表的是行，np.linespace(1,5,5)代表的是列！！！**

**3、查看其中的值及数据访问**
- 查看所有的表格值
```Python
print df.values
[[  1.   1.]
 [  2.   2.]
 [  3.   3.]
 [ nan   4.]]
```
- 访问某一列
DataFrame是以列进行操作的，所以一般是先取一列，再从这个Series中取一个元素。可以用datafrae.column_name选取列*（只能取一列）*，也可以使用dataframe [ ] 操作选取列*（可以取多列）* **（若DataFrame没有列名，[]可以使用非负整数，也就是“下标”选取列；若有列名，则必须使用列名选取）**
```python
print df[1]
print df['b']
print df.b#注意此处直接引用列名，即'.'和'['']'各选一个
print df[['a', 'd']]#这里要千万注意，取了多列，必须用两个方括号
print np.array(df['b'])[:-1]
```
- 访问某一行
```python
print df.iloc[1]
print df.loc['beta']
```
选取行还可以使用切片的方式或者是布尔类型的向量：
```python
print df[1:3]
       a  b  c  d   e
beta   1  2  3  4   5
gamma  2  4  6  8  10

bool_vec = [True, False, True, True, False]
       a  b  c   d   e
alpha  0  0  0   0   0
gamma  2  4  6   8  10
delta  3  6  9  12  15
```

- 访问某一个元素
```python
print df['b'][2]
print df['b']['gamma']
4.0
4.0
```
```python
print df.iat[2, 3]
print df.at['gamma', 'd']
8.0
8.0
```
```python
print df[['b', 'd']].iloc[[1, 3]]
print df.iloc[[1, 3]][['b', 'd']]
print df[['b', 'd']].loc[['beta', 'delta']]
print df.loc[['beta', 'delta']][['b', 'd']]
```
- 更改DataFrame的列名
```python
df.columns=['a', 'b', 'c', 'd', 'e']
print df['b']
```
###2.3 DataFrame进阶属性

- dataframe.head()和dataframe.tail()可以查看数据的头五行和尾五行，若需要改变行数，可在括号内指定

```python

print "Head of this DataFrame:"

print df.head()

print "Tail of this DataFrame:"

print df.tail(3)

```

- dataframe.describe()提供了DataFrame中纯数值数据的统计信息，不包括其他类型值的数据，只包括纯数据类型的数据。

**1、数据排序**

对数据的排序将便利我们观察数据，DataFrame提供了两种形式的排序。
- 第一种是按行列排序，即按照索引（行名）或者列名进行排序，可调用dataframe.sort_index，指定axis=0表示按索引（行名）排序，axis=1表示按列名排序，并可指定升序或者降序：

```python

print df.sort_index(axis=1, ascending=False).head()

```
```python
df2 = df.sort(columns=['secID', 'tradeDate'], ascending=[True, False])
```
- 第二种排序是按值排序，可指定列名和排序方式，默认的是升序排序：

```python

print "Order by column value, ascending:"

print df.sort(columns='tradeDate').head()

print "Order by multiple columns value:"

df = df.sort(columns=['tradeDate', 'secID'], ascending=[False, True])

print df.head()

```

**2、获取数据**

- 一种方法是之前提到过的用布尔型向量获取

- 一种方法是过滤筛选数据

```python

print df[df.closePrice > df.closePrice.mean()].head(

```

即收盘价在均值以上的股票数据

- **isin ( ) 函数**可方便地过滤DataFrame中的数据：

```python

print df[df['secID'].isin(['601628.XSHG', '000001.XSHE', '600030.XSHG'])].head()

 

```

**3、数据操作**

Series和DataFrame的类函数提供了一些函数，如mean()、sum()等，指定0按列进行，指定1按行进行：

```python

df = raw_data[['secID', 'tradeDate', 'secShortName', 'openPrice', 'highestPrice', 'lowestPrice', 'closePrice', 'turnoverVol']]

print df.mean(0)

 

openPrice       1.517095e+01

highestPrice    1.563400e+01

lowestPrice     1.486545e+01

closePrice      1.524275e+01

turnoverVol     2.384811e+08

dtype: float64

```

在panda中，Series可以调用**map函数**来对每个元素应用一个函数，DataFrame可以调用**apply函数**对每一列（行）应用一个函数，**applymap**对每个元素应用一个函数。这里面的函数可以是用户自定义的一个lambda函数，也可以是已有的其他函数。下例展示了将收盘价调整到[0, 1]区间：

```python

print df[['closePrice']].apply(lambda x: (x - x.min()) / (x.max() - x.min())).head()#此处用df[['closePrice']]而不是df['closePrice']，就是因为dataframe才有每一列的最大最小值，df['closePrice']作为一个series，其中的每一个元素都是float的数字，没有最大最小值

   closePrice

0    0.331673

1    0.323705

2    0.313745

3    0.296481

4    0.300465

```

**4、合并**

- 之前知识点有concat知识点

- 这里介绍按照某几列合并的merge函数

```python

dat1 = df[['secID', 'tradeDate', 'closePrice']]

dat2 = df[['secID', 'tradeDate', 'turnoverVol']]

dat = dat1.merge(dat2, on=['secID', 'tradeDate'])

print "The first DataFrame:"

print dat1.head()

print "The second DataFrame:"

print dat2.head()

print "Merged DataFrame:"

print dat.head()

 

The first DataFrame:

         secID   tradeDate  closePrice

0  000001.XSHE  2015-01-05       16.02

1  000001.XSHE  2015-01-06       15.78

2  000001.XSHE  2015-01-07       15.48

3  000001.XSHE  2015-01-08       14.96

4  000001.XSHE  2015-01-09       15.08

The second DataFrame:

         secID   tradeDate  turnoverVol

0  000001.XSHE  2015-01-05    286043643

1  000001.XSHE  2015-01-06    216642140

2  000001.XSHE  2015-01-07    170012067

3  000001.XSHE  2015-01-08    140771421

4  000001.XSHE  2015-01-09    250850023

Merged DataFrame:

         secID   tradeDate  closePrice  turnoverVol

0  000001.XSHE  2015-01-05       16.02    286043643

1  000001.XSHE  2015-01-06       15.78    216642140

2  000001.XSHE  2015-01-07       15.48    170012067

3  000001.XSHE  2015-01-08       14.96    140771421

4  000001.XSHE  2015-01-09       15.08    250850023

```

**5、数据分组处理计算**

- groupby函数可以按照某一列进行分组处理

```python

df_grp = df.groupby('secID')

grp_mean = df_grp.mean()

print grp_mean

 

openPrice    highestPrice   lowestPrice   closePrice  turnoverVol  secID                                                                     

000001.XSHE    14.6550       14.9840      14.4330     14.6650    154710615

000002.XSHE    13.3815       13.7530      13.0575     13.4100    277459431

000568.XSHE    19.7220       20.1015      19.4990     19.7935     29199107

000625.XSHE    19.4915       20.2275      19.1040     19.7170     42633332

000768.XSHE    22.4345       23.4625      21.8830     22.6905     92781199

600028.XSHG     6.6060        6.7885       6.4715      6.6240    531966632

600030.XSHG    31.1505       32.0825      30.4950     31.2325    611544509

601111.XSHG     8.4320        8.6520       8.2330      8.4505    104143358

601390.XSHG     8.4060        8.6625       8.2005      8.4100    362831455

601998.XSHG     7.4305        7.6260       7.2780      7.4345    177541066

```

**6、取最新或者最老的数据（即第一个或最后一个数据）**

- **drop_duplicates**可以实现这个功能，首先对数据按日期排序和对数据进行ID排序，首先ID排序，将同一个股票的放在一块儿，再按时间降序排列，取每个股票的第一个数据（drop_duplicates默认取同一个股票的第一个数据）：

```python

df2 = df.sort(columns=['secID', 'tradeDate'], ascending=[True, False])#排序

print df2.drop_duplicates(subset='secID')#看secID中的重复数据，并只取第一个

```

- 若想要保留最老的数据，可以在降序排列后取最后一个记录，通过指定**take_last=True**（默认值为False，取第一条记录）可以实现

```python

print df2.drop_duplicates(subset='secID', take_last=True)

```

**7、数据可视化**

set_index('tradeDate')['closePrice']表示将DataFrame的'tradeDate'这一列作为索引，将'closePrice'这一列作为Series的值，返回一个Series对象，随后调用plot函数绘图，更多的参数可以在matplotlib的文档中查看。

```python

dat = df[df['secID'] == '600028.XSHG'].set_index('tradeDate')['closePrice']

dat.plot(title="Close Price of SINOPEC (600028) during Jan, 2015")

```
---

##三、numpy知识点
###3.1 数组
**1、 数组的创建**
- NumPy中的基本对象是同类型的多维数组（homogeneous multidimensional array），这和C++中的数组是一致的，例如字符型和数值型就不可共存于同一个数组中。
```python
a = np.arange(20)
print a
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
```
- 简单创建
```python
raw = [0,1,2,3,4]
a = np.array(raw)
a
array([0, 1, 2, 3, 4])

raw = [[0,1,2,3,4], [5,6,7,8,9]]
b = np.array(raw)
b
array([[0, 1, 2, 3, 4],
       [5, 6, 7, 8, 9]])
```
- 通过函数"reshape"，我们可以重新构造一下这个数组，例如，我们可以构造一个4x5的二维数组，其中"reshape"的参数表示各维度的大小，且按各维顺序排列（两维时就是按行排列，这和R中按列是不同的）：
```python
a = a.reshape(4, 5)
print a

[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]
 [15 16 17 18 19]]
 ```
 - 可以调用array的函数进一步查看a的相关属性："ndim"查看维度；"shape"查看各维度的大小；"size"查看全部的元素个数，等于各维度大小的乘积；
 - 一些特殊的数组有特别定制的命令生成，如4x5的全零矩阵：
```python
d = (4, 5)
np.zeros(d)

array([[ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.]])
```
- 默认生成的类型是浮点型，可以通过指定类型改为整型：`np.ones(d, dtype=int)`
- [0, 1)区间的随机数数组：
```python
print np.random.rand(5)
print np.random.rand(2,4)
[[ 0.17571282  0.98510461  0.94864387  0.50078988]
 [ 0.09457965  0.70251658  0.07134875  0.43780173]]
```
**2、 数组的操作**
- 开根号求指数
```python
print a
print np.exp(a)
print np.sqrt(a)
print np.square(a)
print np.power(a, 3)
```
- 二维数组的最大最小值,计算全部元素的和、按行求和、按列求和
```python
a = np.arange(20).reshape(4,5)
print "a:"
print a
print "sum of all elements in a: " + str(a.sum())
print "maximum element in a: " + str(a.max())
print "minimum element in a: " + str(a.min())
print "maximum element in each row of a: " + str(a.max(axis=1))
print "minimum element in each column of a: " + str(a.min(axis=0))
```
- **矩阵和二位数组**：除了数组，NumPy同时提供了矩阵对象（matrix）。矩阵对象和数组的主要有两点差别：一是矩阵是二维的，而数组的可以是任意正整数维；二是矩阵的'x'操作符进行的是矩阵乘法，乘号左侧的矩阵列和乘号右侧的矩阵行要相等，而在数组中'x'操作符进行的是每一元素的对应相乘，乘号两侧的数组每一维大小需要一致。数组可以通过asmatrix或者mat转换为矩阵，或者直接生成也可以：
```python
a = np.arange(20).reshape(4, 5)
a = np.asmatrix(a)

b = np.matrix('1.0 2.0; 3.0 4.0')
```
- range函数还可以通过arange(起始，终止，步长)的方式调用生成等差数列，注意含头不含尾。
```python
b = np.arange(2, 45, 3).reshape(5, 3)
b = np.mat(b)
print b
```
- arange指定的是步长，如果想指定生成的一维数组的长度怎么办？好办，"linspace"就可以做到：
```python
np.linspace(0, 2, 9)

array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ,  1.25,  1.5 ,  1.75,  2.  ])
```
**3、 数组的元素访问**
- 数组和矩阵元素的访问可通过下标进行，以下均以二维数组（或矩阵）为例：
```python
a = np.array([[3.2, 1.5], [2.5, 4]])
print a[0][1]
print a[0, 1]

1.5
1.5
```
- 可以通过下标访问来修改数组元素的值：
```python
b = a
a[0][1] = 2.0
a:
[[ 3.2  2. ]
 [ 2.5  4. ]]
b:
[[ 3.2  2. ]
 [ 2.5  4. ]]
```
现在问题来了，明明改的是a[0][1]，怎么连b[0][1]也跟着变了？这个陷阱在Python编程中很容易碰上，其原因在于**Python不是真正将a复制一份给b，而是将b指到了a对应数据的内存地址上。**想要真正的复制一份a给b，可以使用copy：
```python
a = np.array([[3.2, 1.5], [2.5, 4]])
b = a.copy()
a[0][1] = 2.0
a:
[[ 3.2  2. ]
 [ 2.5  4. ]]
b:
[[ 3.2  1.5]
 [ 2.5  4. ]]
 ```
 
 - 利用':'可以访问到某一维的全部数据，例如取矩阵中的指定列：
```python
 a = np.arange(20).reshape(4, 5)
print "a:"
print a
print "the 2nd and 4th column of a:"
print a[:,[1,3]]
```

- 取出满足某些条件的元素,例子是将第一列大于5的元素（10和15）对应的第三列元素（12和17）取出来：
```python
a[:, 2][a[:, 0] > 5]
array([12, 17])
```
可见这里`a[:, 2][a[:, 0] > 5]`就是`a[:,取列][:,取行]`

- 使用where函数查找特定值在数组中的位置：
```python
a:
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]
 [15 16 17 18 19]]
loc = numpy.where(a==11)
print loc
print a[loc[0][0], loc[1][0]]

(array([2]), array([1]))
11
```
**4、 数组的操作**
- 矩阵转置`a = np.transpose(a)`
- 矩阵求逆
```python
import numpy.linalg as nlg
a = np.random.rand(2,2)
a = np.mat(a)#转换为矩阵
ia = nlg.inv(a)#逆
print "inverse of a:"
print ia
```
- 求特征值和特征向量
```python
a = np.random.rand(3,3)
eig_value, eig_vector = nlg.eig(a)
print "eigen value:"
print eig_value
print "eigen vector:"
print eig_vector

eigen value:
[ 1.35760609  0.43205379 -0.53470662]
eigen vector:
[[-0.76595379 -0.88231952 -0.07390831]
 [-0.55170557  0.21659887 -0.74213622]
 [-0.33005418  0.41784829  0.66616169]]
 ```
 
- 用`column_stack`按列拼接向量成一个矩阵
```python
a = np.array((1,2,3))
b = np.array((2,3,4))
print np.column_stack((a,b))

[[1 2]
 [2 3]
 [3 4]]
```
- 将结果拼接成一个矩阵是十分有用的，可以通过`vstack`和`hstack`完成：
```python
a = np.random.rand(2,2)
b = np.random.rand(2,2)
c = np.hstack([a,b])
d = np.vstack([a,b])
print "horizontal stacking a and b:"
print c
print "vertical stacking a and b:"
print d

horizontal stacking a and b:
[[ 0.6738195   0.4944045   0.28058267  0.0967197 ]
 [ 0.25702675  0.15422012  0.55191041  0.04694485]]
vertical stacking a and b:
[[ 0.6738195   0.4944045 ]
 [ 0.25702675  0.15422012]
 [ 0.28058267  0.0967197 ]
 [ 0.55191041  0.04694485]]
```

**5、缺省值**
- NumPy提供nan作为缺失值的记录，通过isnan判定。
```python
a = np.random.rand(2,2)
a[0, 1] = np.nan
print np.isnan(a)

[[False  True]
 [False False]]
```
- nan_to_num可用来将nan替换成0
```python
print np.nan_to_num(a)

[[ 0.58144238  0.]
 [ 0.26789784 0.48664306]]
```
