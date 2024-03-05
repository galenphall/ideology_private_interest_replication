# Figure 5: Correlates of climate focus and substantive topics among community 1

climate_document_proportions = pd.read_csv('data/processed/climateDocumentProportions.csv').set_index('org')
org_level_1_to_2 = datasets.metadata.set_index('org_level_1').org_level_2.to_dict()
climate_document_proportions = climate_document_proportions.groupby(org_level_1_to_2).mean().mean(axis=1)

centralities = nx.eigenvector_centrality_numpy(interlocks.interlock_graph, weight='weight')
centralities = pd.Series(centralities).sort_values(ascending=False)
centralities = centralities[centralities.index.isin(climate_document_proportions.index)]
centralities = centralities.replace(np.nan, 0)

distance_to_com_2 = {}
distance_to_com_3 = {}
com2 = interlocks.partition[interlocks.partition == 2].index
com3 = interlocks.partition[interlocks.partition == 3].index
for org in centralities.index:
    distance_to_com_3[org] = np.mean([interlocks.social_distance.loc[org, org2] for org2 in com3])
    distance_to_com_2[org] = np.mean([interlocks.social_distance.loc[org, org2] for org2 in com2])

distance_to_com_3 = pd.Series(distance_to_com_3)
distance_to_com_2 = pd.Series(distance_to_com_2)

def dropna_pearsonr(x, y):
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    intersection = x.index.intersection(y.index)
    x = x[intersection]
    y = y[intersection]
    return scipy.stats.pearsonr(x, y)

correlations = {}
correlations['Eigenvector Centrality'] = dropna_pearsonr(centralities, climate_document_proportions)
correlations['Distance to Community 2'] = dropna_pearsonr(distance_to_com_2, climate_document_proportions)
correlations['Distance to Community 3'] = dropna_pearsonr(distance_to_com_3, climate_document_proportions)

# Plot the correlations as a bar chart
fig, ax = plt.subplots(figsize=(5, 4))
for i, (k, v) in enumerate(correlations.items()):
    x = i
    y = v.correlation
    yerr = v.confidence_interval(0.95)
    yerr = [[y - yerr[0]], [yerr[1] - y]]
    ax.bar(x, y, yerr=yerr, color='C0', edgecolor='black', linewidth=1, capsize=5)

ax.set_xticks(range(len(correlations)))
ax.set_xticklabels(correlations.keys())
ax.set_ylabel('Correlation with Climate Focus')
ax.set_ylim(-0.5, 0.5)
ax.axhline(0, color='black', linewidth=1, linestyle='--')
plt.tight_layout()
plt.show()

community_mean_topic_proportions = topic.topic_proportions.merge(
        interlocks.partition, left_on='org', right_index=True).groupby('module_id')[topic.topics_to_retain].mean()
community_mean_topic_proportions = community_mean_topic_proportions.div(community_mean_topic_proportions.sum(1), 0)
com_2_3_mean = community_mean_topic_proportions.loc[[2, 3]].mean()

def similarity_to_trade_docs(x):
    """
    Use cosine similarity to estimate how similar an organization's documents are to the average produced
    by organizations in communities 2 and 3.
    """
    assert isinstance(x, pd.Series)
    x = x.div(x.sum())
    return 1 - scipy.spatial.distance.cosine(x, com_2_3_mean)

topicprops = topic.topic_proportions.loc[
    topic.topic_proportions.org.isin(interlocks.partition[interlocks.partition == 1].index),
    topic.topics_to_retain
]
similarities = topicprops.apply(similarity_to_trade_docs, 1)
similarities = similarities.reset_index(drop=False).rename(columns={0: 'similarity'})
similarities['org'] = topic.topic_proportions.loc[
    topic.topic_proportions.org.isin(interlocks.partition[interlocks.partition == 1].index),
    'org'
]
similarities['technical_doc'] = similarities.similarity > 0.5

number_technical = similarities.groupby('org').technical_doc.sum()

# Plot the number of technical documents per organization
fig, ax = plt.subplots(figsize=(5, 4))
ax.hist(number_technical, color='C0', edgecolor='black', linewidth=1)
ax.set_xlabel('Number of Technical Documents')
ax.set_ylabel('Number of Organizations')
plt.tight_layout()
plt.show()

correlations = {}
correlations['Eigenvector Centrality'] = dropna_pearsonr(centralities, number_technical)
correlations['Distance to Community 2'] = dropna_pearsonr(distance_to_com_2, number_technical)
correlations['Distance to Community 3'] = dropna_pearsonr(distance_to_com_3, number_technical)

# Plot the correlations as a bar chart
fig, ax = plt.subplots(figsize=(5, 4))
for i, (k, v) in enumerate(correlations.items()):
    x = i
    y = v.correlation
    yerr = v.confidence_interval(0.95)
    yerr = [[y - yerr[0]], [yerr[1] - y]]
    ax.bar(x, y, yerr=yerr, color='C0', edgecolor='black', linewidth=1, capsize=5)

ax.set_xticks(range(len(correlations)))
ax.set_xticklabels(correlations.keys())
ax.set_ylabel('Correlation with Number of Technical Documents')
ax.set_ylim(-0.5, 0.5)
ax.axhline(0, color='black', linewidth=1, linestyle='--')
plt.tight_layout()
plt.show()