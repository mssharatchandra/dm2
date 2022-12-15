import pandas as pd 
from mlxtend.preprocessing import TransactionEncoder 
from mlxtend.frequent_patterns import apriori, association_rules 
dataset = [['Milk','Onion', 'Bread', 'Kidney Beans','Eggs','Yoghurt'], 
 ['Fish','Onion','Bread','Kidney Beans','Eggs','Yoghurt'], 
 ['Milk', 'Apples', 'Kidney Beans’, ‘Eggs'], 
 ['Milk', 'Sugar', 'Tea Leaves', 'Kidney Beans', 'Yoghurt'], 
 ['Tea Leaves','Onion','Kidney Beans', 'Ice cream', 'Eggs'], 
 
] 
tr = TransactionEncoder() 
tr_arr = tr.fit(dataset).transform(dataset) 
df = pd.DataFrame(tr_arr, columns=tr.columns_) 
from mlxtend.frequent_patterns import apriori 
frequent_itemsets = apriori(df, min_support = 0.6, use_colnames = True) 
frequent_itemsets
