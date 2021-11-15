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

def reconstruct_upos(upos, docs):
    import numpy as np
    import pandas as pd
    import re

    upos = upos.loc[[d in set(["NOUN", "ADJ", "VERB"])for d in upos.pos],:] # heuristic
    upos = upos.loc[np.logical_not(upos.is_stopword),:]
    upos = upos.loc[[(len(d)>2) & (len(d)<19) for d in upos.lemma],:]
    lemma_stats = upos.groupby("lemma", as_index=False).agg({"doc_id":["count", "nunique"]})
    pf = (lemma_stats[("doc_id","count")]>500)&(lemma_stats[("doc_id","nunique")]>250)
    stopword_set = set([])
    lemma_set = set(lemma_stats.loc[pf,"lemma"].values).difference(stopword_set)
    upos = upos.loc[upos.lemma.isin(lemma_set),:]
    reconstructed = pd.DataFrame(upos.groupby("doc_id").apply(lambda x:" ".join(x["lemma"])),
        columns=["ReconstructedChars"])
    reconstructed["ReconstructedChars"] = reconstructed["ReconstructedChars"].apply(\
        lambda x: re.sub(r'\b(\w+\s*)\1{1,}', '\\1', x))  
    return docs.merge(reconstructed,
        how="inner", left_index=True, right_index=True)      

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
    docs.to_csv("../../data/texts.csv")
    td = (datetime.datetime.now()-st).seconds/60
    print("Text processing finished in {:.2f} mins".format(td))