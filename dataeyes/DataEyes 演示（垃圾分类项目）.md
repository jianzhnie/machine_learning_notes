# DataEyes 演示（垃圾分类项目）

## 1. 添加数据源
- Boundaries_mongo
- TestRecordDesc
- MaterialMapping
- Measurements
## 2. SQL 处理数据
- collect_boundaries
```sql
SELECT cast(ObjID as Long), Direction, Length, Samplings, num_arr_to_vec(Samplings) as hello from boundaries
```

- collect_desc

```sql
SELECT FileName, ObjID, MappingIDCorrect as Material From TestRecordDesc
```

- collect_label

```sql
Select ID as  Material, Category FROM MaterialMapping
```

- collect_absorbance

```sql
select cast(objectId as Long), replace(filename, "/home/hadoop/datasets/garbage_classification/TestRecords/", "") as filename, wavelet_db4(absorbance) as wavelet_arr from Measurements
where absorbance is not NULL
```

```sql
select objectId, replace(filename, '.csv','') as filename, wavelet_arr from  collect_absorbance
```

## 3. 数据合并

(1). 合并 `collect_boundaries`  and  `collect_desc`

```sql
SELECT collect_desc.ObjID, collect_desc.Filename, collect_desc.Material, collect_boundaries.Direction, collect_boundaries.Length, collect_boundaries.Samplings
FROM collect_desc, collect_boundaries 
where collect_desc.ObjID = collect_boundaries.ObjID
```

(2). 合并 `merge1`  and  `collect_label`

```sql
SELECT merge1.ObjID, merge1.Filename, merge1.Material, merge1.Direction, merge1.Length, merge1.Samplings, collect_label.Category
FROM merge1, collect_label 
where collect_label.Material = merge1.Material
```

(3). 合并 `merge2` and `collect_absorbance1`

```sql
SELECT merge2.ObjID, merge2.Filename, merge2.Material, merge2.Direction, merge2.Length, merge2.Samplings, merge2.Category, num_arr_to_vec(collect_absorbance1.wavelet_arr) as wavelet_arr
FROM merge2, collect_absorbance1
where collect_absorbance1.filename = merge2.Filename
```

## 4. Label encoder

- step_name:  LabelEncoder

- transform_function:  StringIndexer
- input_coloum: Category
- output_coloum: Category_index

## 5. StandardScaler

- step_name:  StandardScaler

- transform_function:  FeatureStandardScaler
- input_coloum: wavelet_arr
- output_coloum: wavelet_std

## 6. data_split

## 7. Model

- step_name:  clf

- transform_function:  simpleOnevsRest

- input_coloum:  wavelete_std

  



