import pandas as pd

topics = ['V%i'%i for i in range(1, 51)]
topicprops = pd.read_csv('data/unprocessed/topicproportions.csv').drop('Unnamed: 0', axis = 1, errors = 'ignore')
topiclabels = pd.read_csv('data/unprocessed/topiclabels.csv').drop('Unnamed: 0', axis = 1, errors = 'ignore')
docs = pd.read_csv('data/unprocessed/document_sample.csv').drop(['Unnamed: 0', 'text'], axis = 1, errors = 'ignore')

docs['date'] = pd.to_datetime(docs.date, format ='mixed')
docs['date'] = docs.date.apply(lambda date: date.strftime("%Y-%m-%d"))

topiclabels['topic'] = topics

retainedtopics = topiclabels[~topiclabels.remove].topic.values
drop_docs = topicprops[retainedtopics].sum(1) < 0.5

print(f"Dropping {drop_docs.sum()} documents")
topicprops = topicprops[~drop_docs]
docs = docs[~drop_docs]

topicprops['org'] = docs.org
topicprops['year'] = docs.year

topicprops.to_parquet('data/processed/topic_proportions.parquet')
docs.to_parquet('data/processed/docs.parquet')
topiclabels.to_parquet('data/processed/topic_labels.parquet')

