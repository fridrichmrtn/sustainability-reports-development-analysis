def get_tools():
    import requests
    import glob
    file_set = set(glob.glob("*"))
    if "lid.176.bin" not in file_set:
        ftm_url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
        r = requests.get(ftm_url, allow_redirects=True)
        open("lid.176.bin", "wb").write(r.content)
 
def purge_chars(text):
    import re
    
    text = text.lower()
    text = re.sub(r"<.*?>|</.*?>","",text)
    text = re.sub(r"(s?)(f|ht)tp(s?)://\S+\b","",text)
    text = re.sub(r"^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}$","", text) #email
    text = re.sub(r"\\-","", text)
    text = re.sub("[^a-z '.,?!:]"," ",text)
    text = re.sub(r"\b(\w+\s*)\1{1,}", " ",text) #dupli "\\1"
    return re.sub(r" +"," ",text)

def deduplicate(docs, thre=.97):
    import pandas as pd
    from itertools import product

    def get_similarity(row):
        import spacy
        #spacy.require_gpu()
        nlp =  spacy.load("en_core_web_lg")
        nlp.max_length = 1600000
        d1, d2 = nlp(row["ft"]), nlp(row["tt"]) 
        return d1.similarity(d2)

    def concat_docs(df):
        rd = {}
        rd["index"] = np.min(df.index)
        rd["OutputChars"] = " ".join(df["OutputChars"])
        return pd.Series(rd, index=rd.keys())        

    agp = docs.Participant.value_counts().sort_values()
    di = docs.index[[p in set(agp.index[agp>1]) for p in docs.Participant]]
    di_map = pd.DataFrame(list(product(di, di)), columns=["f", "t"])
    di_map = di_map[di_map.f>di_map.t]
    di_map["fn"] = docs.loc[di_map.f].Participant.values
    di_map["tn"] = docs.loc[di_map.t.values].Participant.values
    di_map["ft"] = docs.loc[di_map.f.values].OutputChars.values
    di_map["tt"] = docs.loc[di_map.t.values].OutputChars.values
    di_map =  di_map[di_map.fn==di_map.tn].reset_index(drop=True)
    di_map["similarity"] = di_map.apply(get_similarity, axis=1)
    fi = [i for i in docs.index if i not in set(di)] + \
        [r[1]["f"] for r in di_map.iterrows() if r[1]["similarity"]>=thre]
    deduplicated = docs.loc[list(set(fi))]
    sf = di_map.similarity<thre
    coni = pd.concat([di_map[sf].f, di_map[sf].t]).reset_index(drop=True).values
    if len(coni)>0:
        dg = docs.loc[coni].groupby("Participant", as_index=False).apply(concat_docs).set_index("index")
    else:
        dg = pd.DataFrame(columns=["Participant", "OutputChars"])
    return deduplicated.append(
        dg.merge(docs.loc[:,[c for c in docs.columns if c not in set(dg.columns)]],
        left_index=True, right_index=True))

def get_text_properties(row, model):
    import re
    import pandas as pd

    string = row.OutputChars
    row["NoChars"] = len(string)
    row["NoWords"] = len(re.split("\w+",string))
    row["NoSentences"] = len(re.split("[.!?]", string))
    lest = model.predict(string, k=1)
    row["EstimatedLanguage"] = lest[0][0].split("_")[-1]
    row["EstimatedLanguageConfidence"] = lest[1][0]
    return row

def get_ngram_freq(strings, ngram_range=(1,1), max_features=5000):
    from sklearn.feature_extraction.text import CountVectorizer
    import pandas as pd

    gramvec = CountVectorizer(ngram_range=ngram_range,
        stop_words="english", max_features=max_features)
    gram_counts = gramvec.fit_transform(strings)
    gram_counts = pd.DataFrame(gram_counts.todense(),
        columns=gramvec.get_feature_names())
    gram_counts = gram_counts.sum().reset_index()
    gram_counts.columns = ["ngram", "frequency"]
    gram_counts["n"] = gram_counts.ngram.apply(lambda x: len(x.split(" ")))
    return gram_counts.loc[:,["n","ngram", "frequency"]]

def plot_ngrams(counts):
    import numpy as np
    import matplotlib.pyplot as plt

    c, r = 2, np.ceil(counts.n.max()/2)
    f, axs = plt.subplots(r,c,figsize=(6.5*c, 5*r))
    for ax, n in zip(axs, counts.n.unique()):
        counts[counts.n==n].sort_values("frequency").tail(15).\
            plot(y="frequency",x="ngram", kind="barh", ax=ax, legend="false",
            logx=True);
        ax.set_title(str(n)+"-gram");
        ax.set_ylabel("");
        ax.set_xlabel("frequency");
        ax.get_legend().remove();
    f.tight_layout()

def get_upos(text, i):
    import spacy
    import pandas as pd

    #spacy.require_gpu()
    nlp =  spacy.load("en_core_web_lg")
    nlp.max_length = 1600000
    parsed = nlp(text)
    ls = [(i, t.text, t.lemma_, t.pos_, t.tag_, t.dep_,
    t.shape_, t.is_alpha, t.is_stop) for t in parsed]
    return pd.DataFrame(ls,
        columns=["doc_id","text", "lemma", "pos", "tag",
            "dep", "shape", "is_alpha", "is_stopword"])

def plot_upos(upos, filter=["NOUN", "VERB","ADJ"]):
    import numpy as np
    import matplotlib.pyplot as plt

    c, r = 3, np.ceil(len(filter)/3)
    f, axs = plt.subplots(r,c,figsize=(6.5*c,5 *r))
    for ax, up in zip(axs, filter):
        upos[upos.pos==up].lemma.value_counts().sort_values().tail(15).\
            plot(kind="barh", ax=ax, legend="false");
        ax.set_title(up);
        ax.set_xlabel("frequency");
        ax.get_legend().remove();
    f.tight_layout();

def reconstruct_upos(upos, docs):
    import numpy as np
    import pandas as pd

    upos = upos.loc[[d in set(["NOUN", "ADJ", "VERB"])for d in upos.pos],:] # heuristic
    upos = upos.loc[np.logical_not(upos.is_stopword),:]
    upos = upos.loc[[(len(d)>2) & (len(d)<19) for d in upos.lemma],:]
    lemma_stats = upos.groupby("lemma", as_index=False).agg({"doc_id":["count", "nunique"]})
    pf = (lemma_stats[("doc_id","count")]>500)&(lemma_stats[("doc_id","nunique")]>250)
    stopword_set = set([])
    lemma_set = set(lemma_stats.loc[pf,"lemma"].values).difference(stopword_set)
    upos = upos.loc[upos.lemma.isin(lemma_set),:]
    reconstructed = upos.groupby("doc_id").apply(lambda x:" ".join(x["lemma"]))
    return docs.merge(pd.DataFrame(reconstructed,
        columns=["ReconstructedChars"]), how="inner", left_index=True, right_index=True)

def get_coocurence(strings):
    from sklearn.feature_extraction.text import TfidfVectorizer #, CountVectorizer    
    import numpy as np
    from itertools import combinations
    import pandas as pd

    coovec = TfidfVectorizer(stop_words="english",
        ngram_range=(1,1), max_features=1000)
    coo_w = coovec.fit_transform(strings)
    tokens = coovec.get_feature_names()
    coo_w = coo_w.T.dot(coo_w)
    coo_w = np.triu(coo_w.todense(), k=1)
    edges = list(combinations(range(coo_w.shape[0]),2))
    ind0, ind1 = [e[0] for e in edges], [e[1] for e in edges]
    freq = coo_w[ind0,ind1]
    coocurence = pd.DataFrame(columns=["from", "to", "weight"])
    coocurence["from"], coocurence["to"] = [tokens[i]for i in ind0], [tokens[i]for i in ind1]
    coocurence["weight"] = freq    
    return coocurence

def plot_coocurence(coocurence):
    import networkx as nx
    import matplotlib.pyplot as plt

    net = nx.convert_matrix.from_pandas_edgelist(coocurence.sort_values("weight").tail(100),
        source="from", target="to", edge_attr="weight")
    f, ax = plt.subplots(1,1, figsize=(15, 10))
    pos = nx.spring_layout(net, seed=1, iterations=100)
    nx.draw_networkx_labels(net, pos, font_size=12,
        font_family="sans-serif", alpha=1, ax=ax);
    nx.draw_networkx_edges(net, width=[net[u][v]['weight']*.5 for u,v in net.edges()],
        pos=pos, alpha=.1);

def get_tfidf_lda(strings):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation

    tfidfvec = TfidfVectorizer(min_df=50, max_features=5000)
    tfidf_counts = tfidfvec.fit_transform(strings)
    lda = LatentDirichletAllocation(n_components=15, max_iter=10,
        learning_method='online', learning_offset=50., random_state=0)
    lda.fit(tfidf_counts)
    return (lda, tfidfvec)            

def plot_top_words(model, feature_names, n_top_words, title):
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 5, figsize=(15, 10))
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

if __name__=="__main__":
    import numpy as np
    import pandas as pd
    import fasttext as fs
    from joblib import Parallel, delayed
    import datetime

    st = datetime.datetime.now()
    # docs
    get_tools()
    docs = pd.read_pickle("../../data/texts.pkl")
    #docs = docs.iloc[:100,:]
    docs = docs[docs.OutputText.notnull()]
    docs["OutputChars"] = docs.apply(lambda x: purge_chars(x.OutputText), axis=1)
    #text filter
    fsm = fs.load_model("lid.176.bin")
    docs = docs.apply(lambda x: get_text_properties(x, fsm), axis=1)
    docs = docs[(docs.EstimatedLanguage=="en")&(docs.EstimatedLanguageConfidence>0.8)&\
       (docs.NoChars>=1000)]
    docs = deduplicate(docs)        
    #upos
    upos_ls = Parallel(n_jobs=4)(delayed(get_upos)(docs.OutputChars.loc[i], i)\
       for i in docs.index)
    upos = pd.concat(upos_ls)    
    docs = reconstruct_upos(upos, docs)
    docs.to_csv("../../data/texts-processed.csv")
    td = (datetime.datetime.now()-st).seconds/60
    print("Text processing finished in {:.2f} mins".format(td))