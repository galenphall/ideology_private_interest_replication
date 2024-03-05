import copy
import os
from pathlib import Path

# use mathteX font
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy
import seaborn as sns

# import adjustText
matplotlib.use('TkAgg')
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

currpath = Path(__file__)

# if we are not running from the "cccm-structure" directory, then we need to
# change the path to the data
if not currpath.name == 'cccm-structure':
    os.chdir(currpath.parent.parent)

from scripts import topic, interlocks, datasets

shapes = ['v', 's', 'o', '^', 'D', 'P', 'X', 'd', 'p', 'h', '8']
combined_org_types = {
    'Foundation': 'Foundation',
    '501(c)(3)': 'Think Tank/Advocacy Org',
    '501(c)(4)': 'Think Tank/Advocacy Org',
    '501(c)(5)': 'Trade Association',
    '501(c)(6)': 'Trade Association',
}
org_type_shapes = {org_type: shape for org_type, shape in zip(combined_org_types.values(), shapes)}


def figure_1(ax):
    G = interlocks.interlock_graph
    # Create position in two steps
    G_gc = G.subgraph(max(nx.connected_components(G), key=len))
    G_zerodegree = G.subgraph([node for node in G.nodes if G.degree[node] == 0])
    G_remaining = G.subgraph(set(G.nodes) - set(G_gc.nodes) - set(G_zerodegree.nodes))

    pos = nx.spring_layout(G_gc, iterations=1000, scale=1, seed=0)
    pos.update(nx.circular_layout(G_zerodegree, scale=1.2))
    pos.update(nx.spring_layout(G_remaining, iterations=1000, scale=1.2, seed=0))

    # Draw each node type separately (to allow for different shapes)
    for org_type in set(combined_org_types.values()):

        nodes = [node for node in G.nodes if combined_org_types[G.nodes[node]['type']] == org_type]
        node_colors = []
        for node in nodes:
            if node in interlocks.partition.index:
                module = interlocks.partition.loc[node]
                if module == -1:
                    node_colors.append('lightgray')
                else:
                    node_colors.append("C" + str(module - 1))
            else:
                node_colors.append('lightgray ')

        patches = nx.draw_networkx_nodes(G, pos,
                                         nodelist=nodes,
                                         node_size=150,
                                         node_color=node_colors,
                                         node_shape=org_type_shapes[org_type],
                                         ax=ax)

        patches.set_edgecolor('black')
        patches.set_label(org_type + "  ")

    # Draw edges
    weights = [G[u][v]['weight'] for u, v in G.edges]
    edges = nx.draw_networkx_edges(G, pos, width=weights, ax=ax, alpha=0.5)

    # Make legend
    handles, labels = ax.get_legend_handles_labels()
    new_handles = []
    new_labels = []
    for h in handles:
        # Copy the pathcollection
        new_handle = copy.copy(h)
        new_handle.set_facecolor('lightgray')
        new_handles.append(new_handle)
    ax.legend(new_handles, labels, loc='upper left', bbox_to_anchor=(0, 1.05),
              ncol=3, frameon=False, fontsize=16, handletextpad=0.1, columnspacing=0.1)

    ax.set_aspect('equal', adjustable='box')
    # turn off axis
    ax.axis('off')

    return ax


# Figure 2: typical documents from each interlock community
def figure_2(axes):
    community_mean_topic_proportions = topic.topic_proportions.merge(
        interlocks.partition, left_on='org', right_index=True).groupby('module_id')[topic.topics_to_retain].mean()
    community_mean_topic_proportions = community_mean_topic_proportions.div(community_mean_topic_proportions.sum(1), 0)

    for i in range(3):
        module = i + 1
        ax1 = axes[i]
        topic_prop = community_mean_topic_proportions.loc[module]
        topic_prop = topic_prop.sort_values(ascending=False).head(10)
        topic_prop.plot.barh(ax=ax1, color='C' + str(i))
        ax1.set_title(f'Community {module} (N = {len(interlocks.partition[interlocks.partition == module])})',
                      fontsize=16)
        ax1.set_xlabel('Topic Proportion', fontsize=16)
        ax1.set_ylabel('Topic', fontsize=16)
        ax1.set_yticklabels([topic.topic_to_label[t] for t in topic_prop.index], fontsize=16)
        ax1.invert_yaxis()

        members = interlocks.partition[interlocks.partition == module].index
        degrees = {u: interlocks.interlock_graph.degree[u] for u in members}
        top_members = pd.Series(degrees).sort_values(ascending=False).head(5)

        ax1.text(0.65, 0.6, 'Top members by degree', transform=ax1.transAxes, fontsize=16)
        for j, (name, degree) in enumerate(top_members.iteritems()):
            ax1.text(0.65, 0.5 - j * 0.1, name.title().replace("'S", "'s") + " (%i)" % degree, transform=ax1.transAxes,
                     fontsize=14)

        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)

    return axes


# Figure 3: connections between interlocks and grants, and interlocks and topics
def figure_3(axes):
    grant_edgelist = datasets.grants.groupby(['GRANTMAKER', 'RECIPIENT']).GRANT_2020_USD.sum().reset_index()
    grant_edgelist.columns = ['GRANTMAKER', 'RECIPIENT', 'weight']
    grant_network = nx.from_pandas_edgelist(grant_edgelist, source='GRANTMAKER', target='RECIPIENT',
                                            edge_attr='weight', create_using=nx.Graph())
    topic_network = nx.Graph(topic.A_topics)
    social_distance = interlocks.social_distance

    for i, (network, ax) in enumerate(zip([grant_network, topic_network], axes)):

        pandas_adj = nx.to_pandas_adjacency(network, weight='weight')
        pandas_adj = pandas_adj.reindex(
            index=social_distance.index,
            columns=social_distance.index,
        )

        if i == 0:
            pandas_adj = np.log10(pandas_adj)
            pandas_adj = pandas_adj.replace(-np.inf, np.nan)
            ax.set_title('Grant Network', fontsize=14)
        else:
            ax.set_title('Topic Network', fontsize=14)

        plotdata = pandas_adj.stack().reset_index()
        plotdata.columns = ['u', 'v', 'weight']
        plotdata = plotdata[plotdata.weight.notnull()]
        plotdata['social_distance'] = plotdata.apply(lambda row: social_distance.loc[row.u, row.v], 1)
        plotdata = plotdata[plotdata.social_distance.notnull()]
        # Drop duplicates
        plotdata = plotdata[plotdata.u < plotdata.v]

        x = plotdata.social_distance
        y = plotdata.weight

        linfit = scipy.stats.linregress(x, y)

        ax.scatter(x, y, s=10, alpha=1, color='C0', edgecolor='none')
        linx = np.linspace(0, 6, 100)

        # QAP for null model: randomize the social distance matrix
        # by permuting the rows and columns. This preserves the
        # marginal distributions of the social distance matrix
        # but destroys any structure in the relationships between
        # the rows and columns. Note that the same permutation must be
        # applied to both rows and columns.
        qap_r = []
        for _ in range(1000):
            perm = np.random.permutation(len(social_distance))
            permuted_social_distance = social_distance.iloc[perm, perm]
            permuted_plotdata = plotdata.copy()
            permuted_plotdata['social_distance'] = permuted_plotdata.apply(
                lambda row: permuted_social_distance.loc[row.u, row.v], 1)
            permuted_plotdata = permuted_plotdata[permuted_plotdata.social_distance.notnull()]
            permuted_plotdata = permuted_plotdata[permuted_plotdata.u < permuted_plotdata.v]
            x = permuted_plotdata.social_distance
            y = permuted_plotdata.weight
            linfit_qap = scipy.stats.linregress(x, y)
            qap_r.append(linfit_qap.rvalue)

        qap_p_value = np.mean(np.array(qap_r) > linfit.rvalue)

        ax.plot(linx, linfit.slope * linx + linfit.intercept, color='black', linestyle='-', linewidth=2,
                label=f'$R^2$ = {linfit.rvalue ** 2:.2f}' + '\n $p_{OLS} < 0.001$' + '\n $p_{QAP}$ = %.3f' % qap_p_value)

        ax.set_xlabel('Social Distance', fontsize=14)
        if i == 0:
            ax.set_ylabel('Grant Amount (2020 USD)', fontsize=14)
            ax.set_yticklabels(['$10^{}$'.format(int(tick)) for tick in ax.get_xticks()], fontsize=14)
        else:
            ax.set_ylabel('Topic Similarity', fontsize=14)

        ax.legend(frameon=False, fontsize=14)



# Plot the relationship between organizational centrality and climate prevalence
def figure_4(ax):
    df = pd.read_csv("data/processed/alltime_grants_interlocks.csv")

    sns.stripplot(x="interlock", y="value", data=df, color='C0', s=3, zorder=-100, ax=ax, alpha=0.5)

    xlims = ax.get_xlim()
    for cond in (True, False):
        subdf = df[cond == df.interlock]
        value = int(subdf[(subdf.value > 1)].value.mean())
        # ax.hlines(value, int(cond), -0.5, transform=ax.transData, color='k', linewidth=0.5, linestyle='dashed')
        # ax.plot(int(cond), subdf[(subdf.value > 1)].value.mean(), marker='o', mfc='white', mec='k')
        ax.annotate(xy=[cond, 1],
                    text=f"avg. donated:\n${value:,}\n$P$ (grant) = {(subdf.value > 0).mean() * 100:.1f}%",
                    ha='center')

    sns.stripplot(x="interlock", y="value", data=df[(~df.interlock) & (df.value == 0)], color='C0', s=5, ax=ax)

    ax.set_ylabel(r"Amount donated, all time (2020 USD)"), ax.set_xlabel("Interlock, all time")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"No\n$N$ = {(~df.interlock).sum()}", f"Yes\n$N$ = {(df.interlock).sum()}"])

    ax.set_yscale('symlog')
    ax.set_ylim(-1, max(ax.get_ylim()) * 5), ax.set_xlim(xlims)

    return ax


# # For figures 1 and 2, combine them in axes like so:
# # |     |   2a   |
# # |  1  |   2b   |
# # |     |   2c   |
#
# fig = plt.figure(figsize=(17, 10))
# gs = fig.add_gridspec(3, 2, width_ratios=[1.5, 1], height_ratios=[1, 1, 1])
# ax1 = fig.add_subplot(gs[:, 0])
# axes = [fig.add_subplot(gs[i, 1]) for i in range(3)]
# # Make axes share x axis
# for i in range(1, 3):
#     axes[i].sharex(axes[0])
#
# figure_1(ax1)
# figure_2(axes)
#
# # Add subplot labels
# ax1.text(0, 1.05, 'A', transform=ax1.transAxes, fontsize=24, fontweight='bold')
# axes[0].text(-0.1, 1.05, 'B', transform=axes[0].transAxes, fontsize=24, fontweight='bold')
# axes[1].text(-0.1, 1.05, 'C', transform=axes[1].transAxes, fontsize=24, fontweight='bold')
# axes[2].text(-0.1, 1.05, 'D', transform=axes[2].transAxes, fontsize=24, fontweight='bold')
#
# plt.tight_layout()
# plt.savefig('figures/interlock_network.pdf', bbox_inches='tight')
# plt.show()
#
# For figures 3a, 3b, 4, combine them in axes like so:
# |  3a  |     |
# |------|  4  |
# |  3b  |     |

# fig = plt.figure(figsize=(10, 8))
# gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], height_ratios=[1, 1])
#
# axes = [fig.add_subplot(gs[i, 0]) for i in range(2)]
#
# ax4 = fig.add_subplot(gs[:, 1])
#
# figure_3(axes)
# figure_4(ax4)

# fig.savefig('figures/interlocks_grants_topics.pdf', bbox_inches='tight')
# fig.savefig('figures/interlocks_grants_topics.png', bbox_inches='tight', dpi=300)
# plt.show()

# Add subplot labels
# axes[0].text(-0.1, 1.05, 'A', transform=axes[0].transAxes, fontsize=24, fontweight='bold')
# axes[1].text(-0.1, 1.05, 'B', transform=axes[1].transAxes, fontsize=24, fontweight='bold')
# ax4.text(-0.1, 1.05, 'C', transform=ax4.transAxes, fontsize=24, fontweight='bold')
#
# plt.tight_layout()
# plt.savefig('figures/social_financial_semantic_ties.pdf', bbox_inches='tight')
# plt.savefig('figures/social_financial_semantic_ties.png', bbox_inches='tight', dpi=300)
# plt.show()


# # Figure 5: data collection and summary statistics
# # first row: topic modeling
# fig, ax = plt.subplots(1, 1, figsize=(10, 5))
#
# matplotlib.rcParams['font.family'] = 'sans-serif'
# matplotlib.rcParams['font.size'] = 16
#
# # Plot most common topics with top words
# # as a horizontal bar chart showing topic proportion in corpus
# # with top words labeling bars at their ends
# n_top_topics = 10
# retained_topics = topic.topics_to_retain
# topic_props = topic.topic_proportions[retained_topics].sum(0)
# topic_props = topic_props / topic_props.sum()
# topic_props = topic_props.sort_values(ascending=False)[:n_top_topics]
# topic_props = topic_props.sort_values(ascending=True)
# topic_props.plot.barh(ax=ax, color='darkgrey', zorder=-100, edgecolor='k', linewidth=1.5)
# # Rescale x axis to make room for labels
# curr_xmax = ax.get_xlim()[1]
# ax.set_xlim(0, curr_xmax * 1.5)
#
# labels = topic.topic_labels.set_index('topic').reindex(topic_props.index).iloc[:n_top_topics]
# for i, (t, row) in enumerate(labels.iterrows()):
#     topwords = ', '.join(row[['V1', 'V2', 'V3', 'V4', 'V5']].values)
#     ax.text(topic_props[t] * 1.05, i, topwords, ha='left', va='center', fontsize=12)
#
# ax.set_yticklabels(labels.label.values, fontsize=16)
#
# ax.set_xlabel('Topic proportion in corpus', fontsize=16)
# ax.set_ylabel('Topic', fontsize=16)
# ax.set_title('Most common topics in corpus', fontsize=20)
#
# # Remove all but left spine
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
#
# fig.savefig('figures/topic_modeling_summary.pdf', bbox_inches='tight')
# fig.savefig('figures/topic_modeling_summary.png', bbox_inches='tight', dpi=300)
# fig.show()
#
# # Figure 6: Social and Topic distance distributions between modules
# # |  6a  |
# # |  6b  |
#
# # Calculate social distance distributions
# matplotlib.rcParams['mathtext.fontset'] = 'cm'
# matplotlib.rcParams['font.family'] = 'STIXGeneral'
# fig, ax = plt.subplots(1,1, figsize=(10, 3.5))
#
# module_1_members = interlocks.partition[interlocks.partition == 1].index
# for module in [1, 2, 3]:
#     members = interlocks.partition[interlocks.partition == module].index
#     # Get social distance distribution for module
#     social_distance = interlocks.social_distance.loc[module_1_members, members]
#
#     # Plot histogram
#     sns.distplot(social_distance, ax=ax, label='Module {}'.format(module), kde=False, hist_kws={'alpha': 0.5},
#                  bins = np.arange(0, 6.1, 0.25), norm_hist=True)
#
# ax.legend(fontsize=16, loc='upper right', frameon=False)
# ax.set_xlabel('Social distance', fontsize=16)
# ax.set_ylabel('Density', fontsize=16)
# ax.set_title('Social distance distributions to members of module 1', fontsize=20)
#
# plt.tight_layout()
# fig.savefig('figures/social_distance_distributions.pdf', bbox_inches='tight')

