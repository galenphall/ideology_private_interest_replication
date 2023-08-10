import pandas as pd
import os
# This should execute from the root of the repository
os.chdir('..')

combined_employees = pd.read_csv('data/processed/combined_employees.csv')

person_ids = pd.read_csv('data/processed/person_ids.csv')

# add the person_id to the combined_employees
combined_employees['person_id'] = combined_employees.name.map(person_ids.set_index('name').person_id.to_dict())

eins = pd.read_csv('data/processed/eins.csv')

docs = pd.read_csv('data/processed/docs.csv')

topic_labels = pd.read_csv('data/processed/topic_labels.csv')

topic_proportions = pd.read_csv('data/processed/topic_proportions.csv')

metadata = pd.read_csv('data/processed/metadata.csv')



