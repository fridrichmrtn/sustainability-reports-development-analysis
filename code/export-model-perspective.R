
# exporting model characteristics to aid human reader

# libs
library(stm)
library(magrittr)
library(igraph)
library(tidygraph)
library(ggraph)

# load data
load(file="data//sat.RData")

# loop over the topic models based on hp sweep
n_topics = c(5,6,7,8,10)

for (n in n_topics){
  
  stm_model = stm(documents=doc, vocab=voc, data=meta,
                  prevalence=~year+state+year*state, verbose=F, K=n)
  
  # init dir
  export_dir = paste0("data//model-perspective//",
    stringi::stri_pad_left(1,width=nchar(max(n_topics)),pad="0"),"-topics//")
  dir.create(export_dir, recursive=T, showWarnings = F)
  
  # sink model info
  toprint = sprintf("We fit a topic model with %i topics, %i documents and a %i word dictionary.\nIn addition, the model's semantic coherence is %f and its exclusivity is %f. \n", 
    stm_model$settings$dim$K, stm_model$settings$dim$N, stm_model$settings$dim$V,
    mean(semanticCoherence(stm_model, doc)), mean(exclusivity(stm_model)))
  
  sink(paste0(export_dir,"_summary.txt"))
  cat(toprint)
  sink()
  
  # topic & labels
  png(filename=paste0(export_dir,"1-topic-prevalence.png"),
    width = 8, height = ceiling(n/2)+0.5, units="in", res=96)
  par(mar=c(4,1,2,1))
  plot(stm_model, type="summary", labeltype="prob", main="top topics",
    xlab="expected topic proportions", cex.lab=0.8, cex.axis=0.8, text.cex=0.8,
    cex.main=0.8, n=5)
  dev.off()
  
  png(filename=paste0(export_dir,"2-topic-labeling.png"),
    width = 28, height = n, units="in", res=96)  
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
  
  # representative tweets where prevalence>.5 
  ft = findThoughts(stm_model, texts=meta$text,
    topics=1:n, thresh=0.5, meta=meta)
  ft_df = lapply(names(ft$index),
    function(x) data.frame(topic = rep(x, length(ft$index[x])),
      index = ft$index[x][[1]])) %>% data.table::rbindlist()
  ft_df[, c("doc_id", "raw_text", "text", "year", "state")] = meta[
    ft_df$index, c("doc_id", "raw_text", "text", "year", "state")]
  write.csv(ft_df[,c("topic", "year", "state", "raw_text", "text")],
    file=paste0(export_dir,"representative-tweets.csv"), row.names=F)
    
  # covariates
  ee = estimateEffect(1:n ~ state + year + year*state, stm_model,
    meta=meta, documents=doc, uncertainty="Local", nsims=100)
  
  # uk & us diffs
  png(filename=paste0(export_dir,"3-state-difference.png"),
    width = 9, height = 4, units="in", res=96)
  plot(ee, model=stm_model, topics=1:n, method="difference",
    covariate="state", cov.value1="United States", cov.value2 = "United Kingdom",
    verbose.labels=F, main="difference in topical prevalence between US & UK",
    labeltype="custom", custom.labels = paste0("T", 1:n),
    xlab = "diff", cex.main=0.8, cex.axis=0.8, cex.lab=0.8)
  dev.off()
    
  # trends
  png(filename=paste0(export_dir,"4-linear-trends.png"),
      width = 12, height = 4*ceiling(n/3), units="in", res=96)
  par(mfrow=c(ceiling(n/3),3), mar=c(2,4,4,2))
  for (i in 1:n){
    plot(ee, model=stm_model, topics=i, covariate="year", method="continuous",
         moderator="state", moderator.value = "United Kingdom", linecol="red",
         printlegend=F,  cex.axis=1.3, cex.lab=1.3, cex.main=1.3,
         main=paste0("T",i), ylim=c(-0.05,0.6), ylab="expected topic proportions", add=F)
    plot(ee, model=stm_model, topics=i, covariate="year", method="continuous",
         moderator="state", moderator.value = "United States", linecol="blue",
         printlegend=F, add=T)
    abline(a=0,b=0, lty="dashed")}
  legend("bottomright", c("United Kingdom", "United States"), lwd=1, col=c("red","blue"))
  dev.off()
  
  # correlation map
  corr_mat = Hmisc::rcorr(stm_model$theta)
  edges = which(corr_mat$r>0 & corr_mat$r!=1 & corr_mat$P<=0.05, arr.ind = T)
  
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
    
    png(filename=paste0(export_dir,"5-corr-network.png"),
        width = 10, height = 4, units="in", res=96)  
    plot(gg)
    dev.off()}
}