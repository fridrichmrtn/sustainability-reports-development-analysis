
def get_ngram_freq(strings, ngram_range=(1,1), max_features=5000):
    from sklearn.feature_extraction.text import CountVectorizer
    import pandas as pd
    import numpy as np

    gramvec = CountVectorizer(ngram_range=ngram_range,
        stop_words="english", max_features=max_features)
    gram_counts = gramvec.fit_transform(strings)
    gram_counts = pd.DataFrame(gram_counts.todense(),
        columns=gramvec.get_feature_names_out())
    gram_counts = gram_counts.sum().reset_index()
    gram_counts.columns = ["ngram", "frequency"]
    gram_counts["n"] = gram_counts.ngram.apply(lambda x: len(x.split(" ")))
    return gram_counts.loc[:,["n","ngram", "frequency"]]

def plot_ngrams(counts, label=""):
    import numpy as np
    import matplotlib.pyplot as plt

    c, r = 2, int(np.ceil(counts.n.max()/2))
    f, axs = plt.subplots(r,c,figsize=(7.5*c, 5.75*r))
    for ax, n in zip(axs.flatten(), counts.n.unique()):
        counts[counts.n==n].sort_values("frequency").tail(15).\
            plot(y="frequency",x="ngram", kind="barh", ax=ax, legend="false",
            logx=True);
        ax.set_title(str(n)+"-gram");
        ax.set_ylabel("");
        ax.set_xlabel("frequency");
        ax.get_legend().remove();
    f.tight_layout();
    f.suptitle(label);

def plot_upos(upos, filter=["NOUN", "VERB","ADJ"]):
    import numpy as np
    import matplotlib.pyplot as plt

    c, r = 3, int(np.ceil(len(filter)/3))
    f, axs = plt.subplots(r,c,figsize=(8*c, 6*r))
    for ax, up in zip(axs.flatten(), filter):
        upos[upos.pos==up].lemma.value_counts().sort_values().tail(15).\
            plot(kind="barh", ax=ax, legend="false");
        ax.set_title(up);
        ax.set_xlabel("frequency");
        ax.get_legend().remove();
    f.tight_layout();    

def get_coocurence(strings):
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.feature_extraction.text import TfidfVectorizer #, CountVectorizer    
    import numpy as np
    from itertools import combinations
    import pandas as pd

    coovec = TfidfVectorizer(stop_words="english",
        ngram_range=(1,1), max_features=1000)
    coo_w = coovec.fit_transform(strings)
    tokens = coovec.get_feature_names_out()
    coo_w = coo_w.T.dot(coo_w)
    coo_w = np.triu(coo_w.todense(), k=1)
    edges = list(combinations(range(coo_w.shape[0]),2))
    ind0, ind1 = [e[0] for e in edges], [e[1] for e in edges]
    freq = coo_w[ind0,ind1]
    coocurence = pd.DataFrame(columns=["from", "to", "weight"])
    coocurence["from"], coocurence["to"] = [tokens[i]for i in ind0], [tokens[i]for i in ind1]
    coocurence["weight"] = freq
    coocurence["weight"] = MinMaxScaler().fit_transform(coocurence[["weight"]])
    return coocurence

def plot_coocurence(coocurence, label=""):
    import networkx as nx
    import matplotlib.pyplot as plt

    net = nx.convert_matrix.from_pandas_edgelist(coocurence.sort_values("weight").tail(1000),
        source="from", target="to", edge_attr="weight")
    f, ax = plt.subplots(1,1, figsize=(20, 20))
    pos = nx.kamada_kawai_layout(net)
    nx.draw_networkx_labels(net, pos, font_size=10,
        font_family="sans-serif", alpha=1, ax=ax);
    nx.draw_networkx_edges(net, width=[net[u][v]['weight']*10 for u,v in net.edges()],
        pos=pos, alpha=.05);
    ax.set_title(label);

def get_tfidf_lda(strings):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation

    tfidfvec = TfidfVectorizer(max_features=5000)
    tfidf_counts = tfidfvec.fit_transform(strings)
    lda = LatentDirichletAllocation(n_components=15, max_iter=10,
        learning_method='online', learning_offset=50., random_state=0)
    lda.fit(tfidf_counts)
    return (lda, tfidfvec)            

def plot_top_words(model, feature_names, n_top_words, title):
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 5, figsize=(20, 15))
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]
        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7);
        ax.set_title(f'Topic {topic_idx +1}',
            fontdict={'fontsize': 12});
        ax.invert_yaxis();
        ax.tick_params(axis='both', which='major', labelsize=10);
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False);
        fig.suptitle(title, fontsize=14);
    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3);
    plt.show();      