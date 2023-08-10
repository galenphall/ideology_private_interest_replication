import os
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

from matplotlib import pyplot as plt
from scipy.spatial.distance import jensenshannon
from scripts.utils.utils import int_to_ein


def social_distance_matrix(G, weight='weight'):
    """
    Compute the social distance between all pairs of nodes in the graph G.
    """
    # Set the distance as the inverse of the number of interlocks
    for u, v in G.edges:
        G[u][v]['distance'] = 1 / G[u][v][weight]

    # Compute the shortest path length between all pairs of nodes
    shortest_path_lengths = dict(nx.shortest_path_length(G, weight='distance'))

    # Convert the dictionary to a pd.DataFrame matrix
    shortest_path_lengths = pd.DataFrame(shortest_path_lengths).reindex(index=G.nodes, columns=G.nodes)

    return shortest_path_lengths


def calculate_partial_r_squared(predictand, full_model_formula, data, predictors):
    # Fit the full model
    model_full = smf.ols(full_model_formula, data=data).fit()

    # Initialize a dictionary to hold the partial R-squared values
    partial_r_squared_values = {}

    # Loop over the predictors
    for predictor in predictors:
        # Create a copy of the predictors list and remove the current predictor
        predictors_without_current = predictors.copy()
        predictors_without_current.remove(predictor)

        # Create the formula for the model without the current predictor
        model_without_current_formula = predictand + ' ~ ' + ' + '.join(predictors_without_current)

        # Fit the model without the current predictor
        model_without_current = smf.ols(model_without_current_formula, data=data).fit()

        # Calculate the partial R-squared and add it to the dictionary
        partial_r_squared_values[predictor] = model_full.rsquared - model_without_current.rsquared

    return partial_r_squared_values


# Change directory
os.chdir('..')

# Load data
alltime_bipartite = pd.read_csv('data/processed/alltime_bipartite.csv')
alltime_interlock = pd.read_csv('data/processed/alltime_interlock.csv')

# Create graph from data
alltime_interlock_graph = nx.from_pandas_edgelist(alltime_interlock, source='ein1', target='ein2', edge_attr='weight')

# Compute the social distance matrix
alltime_social_distance = social_distance_matrix(alltime_interlock_graph)

# Calculate eigenvector centrality
eigenvector_centrality = nx.eigenvector_centrality(alltime_interlock_graph, weight='weight')

# Load document data
topic_proportions = pd.read_parquet('data/processed/topic_proportions.parquet')
topics = pd.read_parquet('data/processed/topic_labels.parquet')[['topic', 'label']]

# Load eins
eins = pd.read_csv('data/processed/eins.csv')

# Load grant data
grants = pd.read_csv('data/processed/grants.csv')

# Keep grants with eins
grants.columns = grants.columns.str.lower().str.replace(' ', '_')
grants = grants[~grants.grantmaker_ein.isna()]
grants['grantmaker_ein'] = grants.grantmaker_ein.astype(int).apply(int_to_ein)
grants['recipient_ein'] = grants.recipient_ein.astype(int).apply(int_to_ein)

# Create the grant network
grant_network = nx.from_pandas_edgelist(grants, source='grantmaker_ein', target='recipient_ein', edge_attr='grant_2020_usd')

# Create the topic similarity network
# First, get average topic proportions for each ein
topic_proportions['ein'] = topic_proportions['org'].map(eins.set_index('org').ein.to_dict())
topic_proportions = topic_proportions[topic_proportions.ein.notna()]
topic_proportions['ein'] = topic_proportions['ein'].apply(int_to_ein)
ein_topic_proportions = topic_proportions.groupby('ein')[topics.topic].mean()

# Then, compute the jensen-shannon distance between each pair of eins
topic_distance = ein_topic_proportions.T.corr(method=jensenshannon)
np.fill_diagonal(topic_distance.values, np.nan)

# Prepare data for plotting
data = grants.groupby(['grantmaker_ein', 'recipient_ein']).grant_2020_usd.sum().reset_index()
data['interlock_tuple'] = data.apply(lambda row: tuple(sorted([row.grantmaker_ein, row.recipient_ein])), axis=1)

# Prepare social distance data for merging
alltime_social_distance = alltime_social_distance.stack().to_frame('social_distance').reset_index()
alltime_social_distance.columns = ['ein1', 'ein2', 'social_distance']
alltime_social_distance['interlock_tuple'] = alltime_social_distance.apply(lambda row: tuple(sorted([row.ein1, row.ein2])), axis=1)

# Merge social distance data
data = data.merge(alltime_social_distance[['interlock_tuple', 'social_distance']], on='interlock_tuple', how='outer')

# Prepare topic distance data for merging
topic_distance.index.name = 'ein1'
topic_distance = topic_distance.stack().to_frame('topic_distance').reset_index()
topic_distance.columns = ['ein1', 'ein2', 'topic_distance']
topic_distance['interlock_tuple'] = topic_distance.apply(lambda row: tuple(sorted([row.ein1, row.ein2])), axis=1)

# Merge topic distance data
data = data.merge(topic_distance[['interlock_tuple', 'topic_distance']], on='interlock_tuple', how='outer')

# Log transform grant data
data['log_grants'] = np.log10(data.grant_2020_usd)

# Plot data
fig, axes = plt.subplots(1, 2, figsize=(8, 3))

sns.regplot(
    data=data[data.social_distance.notna()],
    x='social_distance',
    y='log_grants',
    ax=axes[0],
    scatter_kws={'alpha': 0.1, 'color': 'orange'}
)

sns.regplot(
    data=data[data.topic_distance.notna() & data.social_distance.notna()],
    x='social_distance',
    y='topic_distance',
    ax=axes[1],
    scatter_kws={'alpha': 0.05, 'color': 'orange'}
)

axes[0].set_xlabel('Social distance')
axes[0].set_ylabel('Log grant size')
axes[1].set_xlabel('Social distance')
axes[1].set_ylabel('Topic distance')

fig.tight_layout()
fig.savefig('figures/interlock_social_distance.png', dpi=300)
plt.show()

# Load metadata
metadata = pd.read_csv('data/processed/metadata.csv')
ein_to_dc = (metadata.set_index('ein').state == 'DC').to_dict()
ein_to_ruling_year = metadata.set_index('ein').ruling_year.to_dict()

# Prepare data for model
data['average_received'] = data.groupby('grantmaker_ein').grant_2020_usd.transform('mean')
data['average_given'] = data.groupby('recipient_ein').grant_2020_usd.transform('mean')
data['log_average_received'] = np.log10(data.groupby('grantmaker_ein').grant_2020_usd.transform('mean'))
data['log_average_given'] = np.log10(data.groupby('recipient_ein').grant_2020_usd.transform('mean'))
data['dc'] = data.recipient_ein.map(ein_to_dc)
data['received_x_given'] = data.average_received * data.average_given
data['recipient_eigenvector_centrality'] = data.recipient_ein.map(eigenvector_centrality)
data['ruling_year'] = data.recipient_ein.map(ein_to_ruling_year)

# Calculate partial R-squared values
predictors = ['social_distance', 'log_average_received', 'log_average_given', 'recipient_eigenvector_centrality', 'dc', 'ruling_year']
full_model_formula = 'log_grants ~ ' + ' + '.join(predictors)
data = data[data.social_distance.notna() & data.log_grants.notna()]

partial_r_squared_values = calculate_partial_r_squared('log_grants', full_model_formula, data, predictors)
print(partial_r_squared_values)

from sklearn.utils import resample

def bootstrap_partial_r_squared(data, predictand, predictors, full_model_formula, num_bootstraps=1000):
    bootstrap_partial_r_squared_values = []

    for _ in range(num_bootstraps):
        # Create a bootstrap sample
        bootstrap_sample = resample(data)

        # Calculate the partial R-squared for the bootstrap sample
        bootstrap_partial_r_squared = calculate_partial_r_squared(predictand, full_model_formula, bootstrap_sample, predictors)

        # Store the bootstrap partial R-squared
        bootstrap_partial_r_squared_values.append(bootstrap_partial_r_squared)

    return pd.DataFrame(bootstrap_partial_r_squared_values)

# Calculate bootstrap partial R-squared values
if not os.path.exists('data/calculated/bootstrap_partial_r_squared_values.csv'):
    bootstrap_partial_r_squared_values = bootstrap_partial_r_squared(data, 'log_grants', predictors, full_model_formula)

    # Save the bootstrap partial R-squared values
    bootstrap_partial_r_squared_values.to_csv('data/calculated/bootstrap_partial_r_squared_values.csv', index=False)

# Load the bootstrap partial R-squared values
bootstrap_partial_r_squared_values = pd.read_csv('data/calculated/bootstrap_partial_r_squared_values.csv')

# Calculate the 95% confidence interval for the partial R-squared
lower_bound = bootstrap_partial_r_squared_values.quantile(0.025)
upper_bound = bootstrap_partial_r_squared_values.quantile(0.975)

# Plot the partial R-squared values
fig, ax = plt.subplots(figsize=(4, 3))

sns.kdeplot(
    data=bootstrap_partial_r_squared_values,
    ax=ax,
    fill=True,
    color='orange'
)

ax.axvline(partial_r_squared_values['social_distance'], color='black', linestyle='--')
ax.axvline(lower_bound['social_distance'], color='black', linestyle=':')
ax.axvline(upper_bound['social_distance'], color='black', linestyle=':')
ax.set_xlabel('Partial R-squared')
ax.set_ylabel('Density')

fig.tight_layout()
fig.savefig('figures/interlock_social_distance_partial_r_squared.png', dpi=300)
plt.show()


