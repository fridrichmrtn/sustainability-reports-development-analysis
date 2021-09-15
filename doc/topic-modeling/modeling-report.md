\#zerowaste subset - topic modeling report
================

> Martin Fridrich 09/2021

This document aims to preprocess the CSR reports, examine properties of
the resulting dataset, find a viable topic model, and present imprinted
perspectives. The study is structured as follows:

1 [Housekeepin’](#housekeepin)  
2 [Data processing](#data-processing)  
  2.1 [Character & document-level
processing](#character-document-level-processing)  
  2.2 [Token-level processing](#token-level-processing)  
  2.3 [Execution](#execution)  
3 [Exploratory data analysis](#exploratory-data-analysis)  
4 [Topic modeling](#topic-modeling)  
  4.1 [Hyperparameter sweep](#hyperparameter-sweep)  
  4.2 [Topic labels](#topic-labels)  
  4.3 [Covariates](#covariates)  
  4.4 [Correlation map](#correlation-map)

## Housekeepin’

In the opening section, we load required libs, import raw CSV files &
union them into the resulting data.frame. In addition, we sanitize the
column names and present the overall structure of the dataset.

``` r
data_dir = "..//..//data//"
csv_to_load = paste0(data_dir, list.files(data_dir, pattern="docs-reconstructed.csv"))
raw_docs = data.table::fread(csv_to_load, data.table=F, na.strings="", nThread=4)
raw_docs = raw_docs %>% janitor::clean_names() %>% as.data.frame()
raw_docs = raw_docs[,c("v1", "participant", "sector", "country", "economy_type", "output_text")]
names(raw_docs) = c("doc_id", "participant", "sector", "country", "economy_type", "text")
raw_docs$raw_text = raw_docs$text
#raw_docs = raw_docs[1:200,]
as_tibble(head(raw_docs))
```

## Data processing

In this section, we outline and implement covariate, character, document
& token level transformation steps. Covariates transformation include
datetime conversion & state extraction. Character level processing
transforms the tweets to lower case, removes XML tags, removes links,
unusual characters, and collapses multiple spaces. Document-level
processing is conservative; we keep only tweets written in English and
originated in the US/UK. Consequently, we annotate the tokens with
`udpipe` and retain only `VERB`, `ADJ` & `NOUN` observed in at least ten
separate tweets.

### Character & document-level processing

``` r
# character-level
purge_chars = function(char_vec){
  # to lower
  char_vec = tolower(char_vec)
  # remove hardcoded chars
  char_vec = gsub("(&amp;|&gt;)"," ",char_vec)
  # remove tags
  char_vec = gsub("<.*?>|</.*?>","",char_vec)
  # remove links
  char_vec = gsub("(s?)(f|ht)tp(s?)://\\S+\\b", "", char_vec)
  # use only a-z chars, spaces and punct
  char_vec = gsub("[^a-z#@_ ']"," ",char_vec)
  # collapse multiple spaces
  char_vec = gsub("\\s+"," ",char_vec)
  return(char_vec)}

# docu-level
purge_docs = function(doc_df){
  doc_df = doc_df[nchar(doc_df$text)>=1000,]
  # doc_df = doc_df[doc_df$lang=="en" &
  #     doc_df$state %in% c("United States", "United Kingdom"),]
  # doc_df = doc_df[!grepl("rt @",doc_df$text),]
  # doc_df = arrange(doc_df, date) %>%
  #     distinct(text, .keep_all=T)
  return(doc_df)}
```

### Token-level processing

``` r
# get udpipe model
get_udpipe_model = function(lang="english-ewt", dir="..//..//data//"){
  require(udpipe)
  udp_files = list.files(dir, pattern=paste0("^",lang), full.names=TRUE)
  if (length(udp_files)==0){
    udp_model = udpipe_download_model(language=lang,
      model_dir=dir)
    upd_path = udp_model$file_model}else{
    file_df = file.info(udp_files)
    file_df = file_df[base::order(file_df$ctime,decreasing = T),]
    upd_path = rownames(file_df)[1]}
  return(udpipe_load_model(file = upd_path))}

# annotate texts
get_annotated_texts = function(txt_vec, id, model=get_udpipe_model()){
  require(udpipe)
  names(txt_vec) = id
  annotated_df= udpipe(txt_vec, model, parallel.cores=30)
  annotated_df = as.data.frame(annotated_df, detailed=T)
  return(annotated_df[,c("doc_id", "token_id", "sentence_id",
    "token", "lemma", "upos", "xpos", "dep_rel")])}

# purge lemmas
purge_lemma = function(annotated_df){
  require(dplyr)
  lag_df = data.frame(doc_id = annotated_df$doc_id,
    token_id = as.character(as.numeric(annotated_df$token_id)+1),
    lag = ifelse(annotated_df$lemma %in% c("@", "#"), annotated_df$lemma, NA)) %>%
    filter(!is.na(lag))
  
  annotated_df = annotated_df %>%
    #left_join(lag_df, by=c("doc_id", "token_id")) %>%
    #mutate(lemma = ifelse(!is.na(lag), paste0(lag, lemma), lemma)) %>%
    filter(upos %in% c("NOUN", "VERB", "ADJ") & !grepl("([a-zA-Z0-9])\\1\\1+",lemma) &
        nchar(lemma)>2 & nchar(lemma)<19)
  
  frequent_lemmas = annotated_df %>%
    group_by(lemma) %>%
    summarise(n_doc=n_distinct(doc_id), n=n()) %>%
    filter(n>500 & n_doc>250)
  
  annotated_df = annotated_df %>% # alternatively use inner_join
    filter(lemma %in% frequent_lemmas$lemma)
  return(annotated_df)}

# purge annotations
purge_annot = function(doc_df){
  require(bit64)
  annotated_df = get_annotated_texts(doc_df$text, doc_df$doc_id) %>%
    purge_lemma()

  annotated_df = annotated_df %>%
    group_by(doc_id) %>%
    mutate(text = paste0(lemma, collapse=' ')) %>%
    select(doc_id, annot_text=text) %>% ungroup() %>% unique() %>%
    mutate(doc_id = as.integer64(doc_id))
    
  doc_df = doc_df %>%
    inner_join(annotated_df, by='doc_id') %>%
    select(doc_id, participant, text=annot_text,
    raw_text, economy_type)
  return(doc_df)}
```

### Execution

The next code chunk applies the outlined transformation to the raw data.
Pieces of character & token level processing are done in a parallel
manner.

``` r
st = Sys.time()
print("Entering processing stage...")

docs = raw_docs

cl = makeCluster(20)
  docs$text = parSapply(cl, docs$text, purge_chars)
stopCluster(cl)

docs = purge_docs(docs)
docs = docs[,c("doc_id", "participant", "economy_type", "raw_text", "text")]
docs = purge_annot(docs)

print(paste0("The procedure finished in ", format(Sys.time()-st, digits=2) ,"."))
```

Let’s construct the objects expected by the downstream STM model.

``` r
processed = textProcessor(documents=docs$text,
  metadata=docs, lowercase=F, removestopwords=F,
  removenumbers=F, removepunctuation=F, stem=F)

doc = processed$documents
voc = processed$vocab
meta = processed$meta  
```

## Exploratory data analysis

Within this section, we extract & examine selected properties of the
processed tweets.

``` r
# tweets per year
barplot(table(meta$economy_type),
     main="no of participants",
     xlab="economy type",
     #ylim=c(0,45000),
     #breaks=11,
     cex.main=0.8, cex.axis=0.8, cex.lab=0.8, cex.names=0.8)
```

<img src="img/modeling-report/econ_histogram-1.png" style="display: block; margin: auto;" />

In the plot above, we can see a steady incline until 2019, a decline
since. The behavior is aligned with what we see in the original data
set.

``` r
# tokens & chars per tweet
n_chars = nchar(meta$text)
n_tokens = stringr::str_count(meta$text, "\\w+")

par(mfrow=c(1,2))
hist(n_chars,
     main="no of characters per tweet",
     ylab="frequency",
     xlab="character count",
     cex.main=0.8,
     cex.axis=0.8,
     cex.lab=0.8)
hist(n_tokens,
     main="no of tokens per tweet",
     ylab="frequency",
     xlab="token count",
     cex.main=0.8,
     cex.axis=0.8,
     cex.lab=0.8)
```

<img src="img/modeling-report/char_tok_histograms-1.png" style="display: block; margin: auto;" />

``` r
# the most frequent tokens
freq_df = data.frame(doc) %>% data.table::transpose() %>%
  select(token=V1,count=V2) %>% group_by(token) %>%
  summarize(count=sum(count)) %>% arrange(count) %>%
  mutate(token=voc[token])

par(mfrow=c(1,2))
hist(x=log10(freq_df$count), main="token frequency",
  ylab="frequency", xlab="log10 count", breaks=25,
  cex.lab=0.8, cex.main=0.8, cex.axis=0.8)
with(tail(freq_df, 20),
  barplot(height = log10(count),
  names.arg=token, main="20 most frequent tokens", xlab="log10 count",
  horiz = T, las=2, cex.names=0.8, cex.lab=0.8, cex.main=0.8, cex.axis=0.8))
```

<img src="img/modeling-report/tok_bars-1.png" style="display: block; margin: auto;" />

## Topic modeling

This section deals with several steps; we define the structural topic
model and specific covariate formula, propose & implement
straightforward optimization to estimate a satisfying number of topics.
In addition, we present tools to aid human comprehension of the model.

### Hyperparameter sweep

In the next code chunk, we employ multiobjective grid-search
optimization to find viable candidate models. The evaluation metrics
involve exclusivity and semantic coherence.

``` r
evaluate_topics = function(k, doc, voc, meta){
  require(stm, quietly=T)
  fit = stm(documents=doc, vocab=voc, data=meta,
    prevalence=~economy_type, verbose=F, K=k)
  sc = mean(semanticCoherence(fit, doc))
  frex = mean(exclusivity(fit))
  return(c(k, sc, frex))}

print("Starting hyperparam sweep for number of topics...")
st =  Sys.time()
cl = makeCluster(20)
sweep_df = parSapply(cl, 5:50,
  FUN=evaluate_topics, doc=doc, voc=voc, meta=meta)
stopCluster(cl)
print(paste0("The procedure finished in ", format(Sys.time()-st, digits=2) ,"."))

colnames(sweep_df) = NULL; rownames(sweep_df) = c("k", "semcoh", "frex");
sweep_df = as.data.frame(t(sweep_df))
```

Consequently, L2 distance from the utopia point is estimated, and five
models are selected (see in red).

``` r
# plotting & selection

# min-max lin scale
scale_linear = function(x){
  x_range = range(x)
  scale_linear = (x-x_range[1])/(x_range[2]-x_range[1])
  return(scale_linear)}

# scale metrics
sweep_df[,c("semcoh_scaled", "frex_scaled")] = apply(sweep_df[,c('semcoh','frex')],
  2, scale_linear)
# distance from utopia
sweep_df$dist = ((1-sweep_df$frex_scaled)^2+(1-sweep_df$semcoh_scaled)^2)^(1/2)
max_dist =  arrange(sweep_df, dist) %>% select(dist) %>% slice(5) %>% unlist()

par(mar=c(4,4,2,1))
plot(x=sweep_df$semcoh_scaled, y=sweep_df$frex_scaled, type='n',
  main="exclusivity & coherence trade-off",xlab='semantic coherence', ylab='exclusivity',
  cex.main=0.8, cex.axis=0.8, cex.lab=0.8)

for(r in 1:nrow(sweep_df)){
  text(x=sweep_df$semcoh_scaled[r], y=sweep_df$frex_scaled[r], label=sweep_df$k[r],
    cex=0.75, col = ifelse(sweep_df$dist[r]<=max_dist,"red","black"))}
```

<img src="img/modeling-report/sweep_tradeoff-1.png" style="display: block; margin: auto;" />

Let’s build the topic model! Note the prevalence formula - we estimate
topical prevalence on dummy variable `economy_type`.

``` r
stm_model = stm(documents=doc, vocab=voc, data=meta,
  prevalence=~economy_type, verbose=F, K=15)

toprint = sprintf("We fit a topic model with %i topics, %i documents and a %i word dictionary. \n In addition, the model's semantic coherence is %f and its exclusivity is %f. \n", 
  stm_model$settings$dim$K, stm_model$settings$dim$N, stm_model$settings$dim$V,
  mean(semanticCoherence(stm_model, doc)), mean(exclusivity(stm_model)))
cat(toprint)
```

    ## We fit a topic model with 15 topics, 2122 documents and a 2584 word dictionary. 
    ##  In addition, the model's semantic coherence is -10.141337 and its exclusivity is 9.215940.

### Topic labels

First, we try to describe the topics with relevant tokens & texts. In
the plot below, we can take a peek at topical prevalence & respective
frequent entities.

``` r
# topic prevalence & props
par(mar=c(4,1,2,1))
plot(stm_model, type='summary', labeltype='prob', main="top topics",
  xlab="expected topic proportions", cex.lab=0.8, cex.axis=0.8, text.cex=0.8,
  cex.main=0.8, n=5)
```

<img src="img/modeling-report/topic_prevalence-1.png" style="display: block; margin: auto;" />

There are, however, various techniques to identify exciting tokens
within a latent factor. Besides the observed probability, we include
`frex`, `lift`, and `score` indicators to get descriptive lemmas (see
the
[vignette](https://cran.r-project.org/web/packages/stm/vignettes/stmVignette.pdf)).

``` r
par(mfrow=c(4,1), mar=c(1,1,1,1))
plot(stm_model, type="labels", labeltype = "prob", main="proba",
  cex.main=1.3, text.cex=1.15, n=10)
plot(stm_model, type="labels", labeltype = "frex", main="frex",
  cex.main=1.3, text.cex=1.15, n=10)
plot(stm_model, type="labels", labeltype = "lift", main="lift",
  cex.main=1.3, text.cex=1.15, n=10)
plot(stm_model, type="labels", labeltype = "score", main="score",
  cex.main=1.3, text.cex=1.15, n=10)
```

<img src="img/modeling-report/topic_labeling-1.png" style="display: block; margin: auto;" />

Furthermore, latent factors are examined from the perspective of the
most representative documents. For each topic, we extract only tweets
with a prevalence of 25 % or higher. For each tweet, original raw text
is printed.

``` r
# top tweets per topic
ft = findThoughts(stm_model, texts=meta$text, topics=1:stm_model$settings$dim$K,
  thresh=0.25, meta=meta)
ft_df = lapply(names(ft$index),
  function(x) data.frame(topic = rep(x, length(ft$index[x])),
    index = ft$index[x][[1]])) %>% data.table::rbindlist()
ft_df[, c("doc_id", "raw_text", "text", "economy_type")] = meta[
  ft_df$index, c("doc_id", "raw_text", "text", "economy_type")]
ft_df[,c("doc_id", "raw_text", "text", "economy_type")] %>%
  mutate(raw_text=substr(raw_text,1,70)) %>% as_tibble()
```

    ## # A tibble: 45 × 4
    ##       doc_id raw_text                    text                       economy_type
    ##        <dbl> <chr>                       <chr>                      <chr>       
    ##  1 7.52e-321 "ANNUAL REPORT 2019 INTROD… annual report introductio… Advanced    
    ##  2 1.41e-321 "BL. me La ’ te Take. iy )… take ser cae owe would ce… Advanced    
    ##  3 1.72e-321 "i es ; {, _— 2 —— a 4 es,… deliver wae see sustainab… Advanced    
    ##  4 1.69e-321 "A ad ry Oe he i, A * eg \… poe have bee responsible … Advanced    
    ##  5 7.85e-322 "The J.J. Ugland Companies… company annual corporate … Advanced    
    ##  6 2.05e-321 "SYA @@] eA eee mee uel eu… performance data summary … Developing  
    ##  7 1.20e-320 "ZenS. TECHNOLOGIES Januar… technology january ant se… Developing  
    ##  8 1.09e-320 "Mapping of UNGC Principle… mapping ungc principles p… Developing  
    ##  9 1.08e-320 "[A] Pal SOLUTION Date 12.… pal solution date united … Advanced    
    ## 10 9.49e-321 "a —,... Eizigi a iil ase … ase see sae sustainabilit… Advanced    
    ## # … with 35 more rows

### Covariates

The structural topic models stand out to include external covariates to
explain the observed topical prevalence. Thus, in the following code
chunks, we present captured relationships.

``` r
ee = estimateEffect(1:stm_model$settings$dim$K ~ economy_type, stm_model,
  meta=meta, documents=doc, uncertainty='Local', nsims=100)
```

``` r
# states
plot(ee, model=stm_model, topics=1:stm_model$settings$dim$K, method="difference",
  covariate="economy_type", cov.value1="Advanced", cov.value2 = "Developing",
  xlim=c(-0.1,0.1), verbose.labels=F, main="diff in topical prevalence between adv & dev economies",
  labeltype="custom", custom.labels = paste0("T", 1:stm_model$settings$dim$K),
  xlab = "diff", cex.main=0.8, cex.axis=0.8, cex.lab=0.8)
```

<img src="img/modeling-report/econ_difference-1.png" style="display: block; margin: auto;" />

### Correlation map

The topic model allows for correlation between topics; positive
correlation suggests that both subjects are likely to be discussed
within one tweet. We construct a network with positive correlations
between factors, we ignore the asymptotic p-values.

``` r
corr_mat = Hmisc::rcorr(stm_model$theta)
edges = which(corr_mat$r>0 & corr_mat$r!=1 & corr_mat$P<.05/14, arr.ind = T)
edges_df = as.data.frame(edges)
edges_df$value = corr_mat$r[edges]
edges_df = edges_df[edges_df$row>edges_df$col,]
nodes_df = data.frame(name=1:stm_model$settings$dim$K,
  proportion=colMeans(stm_model$theta)) %>%
  filter(name %in% edges_df$row | name %in% edges_df$col)

tc_net = graph_from_data_frame(edges_df, vertices=nodes_df, directed=F) %>%
  as_tbl_graph(tc_net) %>%
  mutate(community=as.factor(group_fast_greedy()))

ggraph(tc_net, 'kk')+
  geom_edge_link(aes(width=value), alpha=0.25)+
  scale_edge_width(range = c(0.5, 2), breaks = c(0.1,0.2,0.3))+
  geom_node_point(aes(size=proportion, colour=community))+
  scale_size(range = c(0.5, 2), breaks = c(0.125, 0.15, 0.175))+
  geom_node_text(aes(label=paste0("T", name)), size=4, repel=T)+
  theme_graph(base_family = 'sans',
    base_size = 10, background='white',plot_margin = margin(15, 15, 15, 15))+
  theme(legend.title=element_text(size=8),
    legend.text=element_text(size=8), legend.justification=c(0,0),
      legend.box = "horizontal")+
    guides(color=F)+
    labs(edge_alpha='Pearson\'s correlation', edge_width='Pearson\'s correlation',
      size='Average topic prevalance')
```

    ## Warning: `guides(<scale> = FALSE)` is deprecated. Please use `guides(<scale> =
    ## "none")` instead.

<img src="img/modeling-report/corr_network-1.png" style="display: block; margin: auto;" />

> Martin Fridrich 09/2021
