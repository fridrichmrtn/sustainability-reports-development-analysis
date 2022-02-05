
# exporting model characteristics to aid human reader

# libs
library(stm)
library(magrittr)
library(igraph)
library(tidygraph)
library(ggraph)

# load data
# load
data_dir = "data//"
csv_to_load = paste0(data_dir, list.files(data_dir, pattern="texts.csv"))
docs = data.table::fread(csv_to_load, data.table=F, na.strings="", nThread=4)
colnames(docs)[1] = "doc_id"
colnames(docs) = janitor::make_clean_names(colnames(docs))

# prepare for stm objects
out = textProcessor(documents=docs$reconstructed_chars,
                    metadata=docs, lowercase=F, removestopwords=F,
                    removenumbers=F, removepunctuation=F, stem=F)

# loop over the topic models based on hp sweep
n_topics = 5:20
formula = as.formula("~ economy_type")

for (n in n_topics){
  
  stm_model = stm(documents=out$documents, vocab=out$vocab, data=out$meta,
    prevalence=formula, verbose=F, K=n, seed=n, max.em.its=750,
    init.type="Spectral")
  
  # init dir
  export_dir = paste0("data//model_exports//",
    stringi::stri_pad_left(n,width=nchar(max(n_topics)),pad="0"),"_topics//")
  dir.create(export_dir, recursive=T, showWarnings = F)
  
  # sink model info
  toprint = sprintf("We fit a topic model with %i topics, %i documents and a %i word dictionary.\nIn addition, the model's semantic coherence is %f and its exclusivity is %f. \n", 
    stm_model$settings$dim$K, stm_model$settings$dim$N, stm_model$settings$dim$V,
    mean(semanticCoherence(stm_model, out$documents)), mean(exclusivity(stm_model)))
  sink(paste0(export_dir,"_model_summary.txt"))
  cat(toprint)
  sink()
  
  # topic & labels
  png(filename=paste0(export_dir,"1_topic_prevalence.png"),
    width = 8, height = ceiling(n/2)+0.5, units="in", res=300)
  #par(mar=c(4,1,2,1))
  #plot(stm_model, type="summary", labeltype="frex", main="",
  #  xlab="expected topic proportions", cex.lab=0.8, cex.axis=0.8, text.cex=0.8,
  #  cex.main=0.8, n=5)
  
  topic_prevalence = data.frame(topic=paste0("Topic ", 1:n),
    prevalence=stm_model$theta%>%colMeans()) %>% arrange(prevalence)
  par(mar=c(4,6,1,1))
  barplot(topic_prevalence$prevalence, names.arg=topic_prevalence$topic,
    horiz=T, las=1, xlim=c(0,0.3), xlab="expected topic proportion")
  dev.off()
  
  # top tokens
  png(filename=paste0(export_dir,"2_topic_tokens.png"),
    width = 28, height = n, units="in", res=300)  
  par(mfrow=c(1,4), mar=c(1,1,1,1))
  plot(stm_model, type="labels", labeltype = "prob", main="proba",
    cex.main=1.3, text.cex=1.3, n=15)
  plot(stm_model, type="labels", labeltype = "frex", main="frex",
    cex.main=1.3, text.cex=1.3, n=15)
  plot(stm_model, type="labels", labeltype = "lift", main="lift",
    cex.main=1.3, text.cex=1.3, n=15)
  plot(stm_model, type="labels", labeltype = "score", main="score",
    cex.main=1.3, text.cex=1.3, n=15)
  dev.off()
  
  # top 3 docs
  ft = findThoughts(stm_model, texts=out$meta$reconstructed_chars,
    topics=1:stm_model$settings$dim$K, n=3, meta=out$meta)
  ft_df = lapply(names(ft$index),
    function(x) data.frame(topic = rep(x, length(ft$index[x])),
      index = ft$index[x][[1]])) %>% data.table::rbindlist() %>%
      as.data.frame()
  mirror_cols = c("doc_id", "participant", "reconstructed_chars", "output_chars", "economy_type")
  ft_df[, mirror_cols] = out$meta[ft_df$index, mirror_cols]
  ft_df[, c("doc_id", "participant", "economy_type", "reconstructed_chars")] %>%
    mutate(reconstructed_chars=substr(reconstructed_chars,1,500)) %>%
    write.csv(paste0(export_dir,"3_topic_documents.csv"))
    
  # covariates
  estimated_effects = estimateEffect(formula, stm_model,
    meta=out$meta, documents=out$documents, uncertainty="Global", nsims=250)
  
  png(filename=paste0(export_dir,"4_economy_diff.png"), units="in", width=8,
        height=ceiling(n/2)+0.5, res=300)
  #par(mar=c(1,1,1,1))
  plot(estimated_effects, model=stm_model, topics=1:n, method="difference",
    covariate="economy_type", cov.value1="Advanced", cov.value2 = "Developing",
    xlim=c(-0.15,0.15), verbose.labels=F, main="",
    labeltype="custom", custom.labels = paste0("Topic ", 1:n),
    xlab = "prevalence difference", cex.main=0.8, cex.axis=0.8, cex.lab=0.8)
  dev.off()
  
  sector_prevalence = stm_model$theta %>% as.data.frame() %>%
    mutate(sector=docs$sector) %>% group_by(sector) %>%
    dplyr::summarise(across(everything(), mean), n=n()) %>%
    arrange(desc(n))
  
  png(filename=paste0(export_dir,"5_sector_prevalence.png"), units="in", width=8,
      height=ceiling(n/2)+0.5, res=300)
  par(mfrow=c(2,3))
  for (i in 1:6){
    sector = sector_prevalence$sector[i]
    np = n+1
    #par(mar=c(4,6,4,1))
    barplot(as.matrix(sector_prevalence[i, np:2]),
            names.ar =paste0("Topic ", n:1), xlim=c(0,.45),
            col="gray", horiz=T, las=1, xlab="expected topical prevalence", main=sector)}
  dev.off()
  
  # correlation map
  corr_mat = Hmisc::rcorr(stm_model$theta)
  edges = which(corr_mat$r>0 & corr_mat$r!=1, arr.ind = T)
  
  if (length(edges)>0){
    edges_df = as.data.frame(edges)
    edges_df$value = corr_mat$r[edges]
    edges_df = edges_df[edges_df$row>edges_df$col,]
    nodes_df = data.frame(name=1:stm_model$settings$dim$K,
      proportion=colMeans(stm_model$theta)) %>%
      filter(name %in% edges_df$row | name %in% edges_df$col)
    tc_net = graph_from_data_frame(edges_df, vertices=nodes_df, directed=F) %>%
      as_tbl_graph(tc_net)
    gg = ggraph(tc_net, "kk")+
      geom_edge_link(aes(width=value), alpha=0.25)+
      scale_edge_width(range = c(0.5, 2), breaks = 1:6/10)+
      geom_node_point(aes(size=proportion))+
      scale_size(range = c(0.5, 2), breaks = 1:6/10)+
      geom_node_text(aes(label=paste0("T", name)), size=4, repel=T)+
      theme_graph(base_family = "sans",
                  background="white", plot_margin = margin(15, 15, 15, 15))+
      theme(legend.position="right",
            legend.title = element_text(size=10), legend.text=element_text(size=8))+
      labs(edge_alpha="Pearson\'s correlation",
           edge_width="Pearson\'s correlation", size="topic prevalance")
    png(filename=paste0(export_dir,"6_corr_network.png"),
        units="in", width = 10, height = 4, res=300)  
    plot(gg)
    
    dev.off()}
  
  # push down objects
  save(stm_model, estimated_effects, out,
       file=paste0(export_dir,"stm_artifacts.RData"))}