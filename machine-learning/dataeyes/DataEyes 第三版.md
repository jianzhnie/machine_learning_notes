DataEyes 第三版

### step1: 新建数据集

Titanic 数据路径：`/public_datasets/titanic/Titanic.csv`

### step2: 新建实验

- 1. 数据源

- 2. SQL 语句

  - step name:  sample

```sql
select *
from titanic
LIMIT 100
```
- 3.  Python 语句

  - step name: trans

```python
# 代码模板
class MyTransformer(AbstractTransformer):
    def transform(self, input_map, spark_session, step_name):
        res = {}
        df = input_map['sample'] # 计算得到一个dataframe 
        df1 = df.select(df['Name'], df['Age'], df['Sex'] , df['Survived'])
        res[step_name] = df1
        return res
```

- 4. SQL 语句

  - step name:  sql

```sql
select count(*) as num, Sex
from trans
group by Sex
```

- 5.  python 语句

  - step name: str2index

```python
# 代码模板
from pyspark.ml.feature import StringIndexer
class Str2Index(AbstractTransformer):
    def transform(self, input_map, spark_session, step_name):
        res = {}
        df = input_map['sample'] # 计算得到一个dataframe 
        indexer = StringIndexer(inputCol="Embarked", outputCol="categoryIndex")
        indexed = indexer.fit(df).transform(df)
        res[step_name] = indexed
        return res
```

- 6. SQL 语句

  - step name: imputer

```sql
SELECT *
    FROM str2index
    WHERE Sex IS Not NULL and Age IS NOT NULL AND Embarked IS NOT NULL
```

- 7. 特征工程 OneHot transformer

  - step name: onehot
    - inputcols: 
      - categoryIndex
    - outputcosl:
      - str2onehot

- 8. data_spit  
- 9. 分类模型

  - step name:  rf
    - inputcols: 
      - str2onehot
      - Age
      - Sex

- 10. AutoML

  - step name:  rf
    - ​	inputcols: 
      - str2onehot
      - Age
      - Sex