import pandas as pd
import os


# This should execute from the root of the repository

os.chdir('..')
print(os.getcwd())

combined_employees = pd.read_csv('data/processed/merged_employees.csv')

# create yearly bipartite graphs
yearly_bipartite = {}
for year in combined_employees.year.unique():
    year = int(year)
    yearly_bipartite[year] = combined_employees.loc[combined_employees.year == year, ['person_id', 'ein']].drop_duplicates()

# create the yearly interlock graphs
yearly_interlock = {}
for year in combined_employees.year.unique():
    year = int(year)
    yearly_interlock[year] = pd.merge(yearly_bipartite[year], yearly_bipartite[year], on='person_id').drop_duplicates()
    yearly_interlock[year] = yearly_interlock[year].loc[yearly_interlock[year].ein_x != yearly_interlock[year].ein_y]
    yearly_interlock[year] = yearly_interlock[year].groupby(['ein_x', 'ein_y']).size().reset_index()
    yearly_interlock[year].columns = ['ein1', 'ein2', 'weight']

# create the all-time bipartite graph, including non-concurrent connections
alltime_bipartite = combined_employees[['person_id', 'ein']].drop_duplicates()

# create the all-time interlock graph, including non-concurrent connections
alltime_interlock = pd.merge(alltime_bipartite, alltime_bipartite, on='person_id').drop_duplicates()
alltime_interlock = alltime_interlock.loc[alltime_interlock.ein_x != alltime_interlock.ein_y]
alltime_interlock = alltime_interlock.groupby(['ein_x', 'ein_y']).size().reset_index()
alltime_interlock.columns = ['ein1', 'ein2', 'weight']

# save the graphs
if not os.path.exists('data/processed/yearly_bipartite'):
    os.makedirs('data/processed/yearly_bipartite')
if not os.path.exists('data/processed/yearly_interlock'):
    os.makedirs('data/processed/yearly_interlock')
for year in combined_employees.year.unique():
    year = int(year)
    yearly_bipartite[year].to_csv(f'data/processed/yearly_bipartite/{year}.csv', index=False)
    yearly_interlock[year].to_csv(f'data/processed/yearly_interlock/{year}.csv', index=False)

alltime_bipartite.to_csv('data/processed/alltime_bipartite.csv', index=False)
alltime_interlock.to_csv('data/processed/alltime_interlock.csv', index=False)




