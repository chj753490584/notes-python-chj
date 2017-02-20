[一、基本知识](#一基本知识)

  [1.1 list、元组 和字典](#11-list元组-和字典)
  
  [1.2 python 中的循环语句及函数](#12-python-中的循环语句及函数)
  
    [1、 定义函数](#1-定义函数)
    [2、 if语句](#2-if语句)
    [3、 while语句](#3-while语句)
    
  [1.3 python 中的类](#13-python 中的类)
[二、pandas知识点](#二pandas知识点)

 [2.1 Series](#21-Series)
 
 [2.2 DataFrame](22-DataFrame)
 
 [2.3 DataFrame进阶属性](#23-DataFrame进阶属性)
 
 []()


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

**4、 for 语句**

可以遍历一个序列/字典等。

```python

a=[1,2,3,4,5]

for i in a:

    print i

```

**5、列表推导式：轻量级循环**

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
**1、 Series数据的访问**
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
----

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

在panda中，Series可以调用**map函数**来对每个元素应用一个函数，DataFrame可以调用**apply函数**对每一列（行）应用一个函数，**applymap**对每个元素应用一个函数。这里面的
