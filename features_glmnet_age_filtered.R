#%%
rm(list = ls())
setwd('/Users/michaelwu/Desktop/MSN_FC_SC')
library(reticulate)
np <- import('numpy')
library(glmnet)
library(readxl)
library(caret)
library(Matrix)
library(tidyverse)
library(xgboost)
library(rBayesianOptimization)
library(ggplot2)
library(doParallel)
registerDoParallel(cores = parallel::detectCores())
library(PRROC)
library(pROC)
library(RColorBrewer)
library(wordcloud)
library(gplots)
library(plotly)


#%%
# read demo
demo <- read_excel('demo20230519.xlsx')

# revise the names of the group column
demo$group <- as.character(demo$group)
demo$group[demo$group == 'PA_SUB'] <- 'OBDs'
demo$group[demo$group == 'HC'] <- 'HC'
demo$group[demo$group == 'PA_NO'] <- 'OBDns'
demo$group[demo$group == 'NO_SUB'] <- 'nOBDs'
demo$group[demo$group == 'BP'] <- 'BD'

# find the row indices of the demo$age > 360 with the demo$group == 'BP'
idx <- which(demo$age > 240 & demo$group == 'BD')

# remove the rows in demo
demo <- demo[-idx,]

# load the graph measures in csv
# local measures
clustering_fc <- read.csv('/Volumes/wjf-2/MSN_FC_SC/FC_SC_MatPlots/clustering_fc.csv', header = FALSE)
clustering_sc <- read.csv('/Volumes/wjf-2/MSN_FC_SC/FC_SC_MatPlots/clustering_sc_hemisphere.csv', header = FALSE)
local_efficiency_fc <- read.csv('/Volumes/wjf-2/MSN_FC_SC/FC_SC_MatPlots/local_efficiency_fc.csv', header = FALSE)
local_efficiency_sc <- read.csv('/Volumes/wjf-2/MSN_FC_SC/FC_SC_MatPlots/local_efficiency_sc_hemisphere.csv', header = FALSE)
degree_fc <- read.csv('/Volumes/wjf-2/MSN_FC_SC/FC_SC_MatPlots/degree_fc.csv', header = FALSE)
degree_sc <- read.csv('/Volumes/wjf-2/MSN_FC_SC/FC_SC_MatPlots/degree_sc_hemisphere.csv', header = FALSE)
betweenness_fc <- read.csv('/Volumes/wjf-2/MSN_FC_SC/FC_SC_MatPlots/betweenness_fc.csv', header = FALSE)
betweenness_sc <- read.csv('/Volumes/wjf-2/MSN_FC_SC/FC_SC_MatPlots/betweenness_sc_hemisphere.csv', header = FALSE)
# global measures
characteristic_path_length_fc <- read.csv('/Volumes/wjf-2/MSN_FC_SC/FC_SC_MatPlots/characteristic_path_length_fc.csv', header = FALSE)
characteristic_path_length_sc <- read.csv('/Volumes/wjf-2/MSN_FC_SC/FC_SC_MatPlots/characteristic_path_length_sc_hemisphere.csv', header = FALSE)
global_efficiency_fc <- read.csv('/Volumes/wjf-2/MSN_FC_SC/FC_SC_MatPlots/global_efficiency_fc.csv', header = FALSE)
global_efficiency_sc <- read.csv('/Volumes/wjf-2/MSN_FC_SC/FC_SC_MatPlots/global_efficiency_sc_hemisphere.csv', header = FALSE)
modularity_fc <- read.csv('/Volumes/wjf-2/MSN_FC_SC/FC_SC_MatPlots/modularity_fc.csv', header = FALSE)
modularity_sc <- read.csv('/Volumes/wjf-2/MSN_FC_SC/FC_SC_MatPlots/modularity_sc_hemisphere.csv', header = FALSE)
# small_worldness_fc <- read.csv('/media/hello/wjf-2/MSN_FC_SC/FC_SC_MatPlots/small_worldness_fc.csv', header = FALSE)
# small_worldness_sc <- read.csv('/media/hello/wjf-2/MSN_FC_SC/FC_SC_MatPlots/small_worldness_sc_hemisphere.csv', header = FALSE)

# stack the graph measures to form the third dimension
graph_measures <- cbind(clustering_fc, clustering_sc, local_efficiency_fc, local_efficiency_sc, degree_fc, degree_sc,
                        betweenness_fc, betweenness_sc, characteristic_path_length_fc, characteristic_path_length_sc,
                        global_efficiency_fc, global_efficiency_sc, modularity_fc, modularity_sc)
graph_measures <- graph_measures[-idx ,]
rm(list = c('clustering_fc', 'clustering_sc', 'local_efficiency_fc', 'local_efficiency_sc', 'degree_fc', 'degree_sc',
            'betweenness_fc', 'betweenness_sc', 'characteristic_path_length_fc', 'characteristic_path_length_sc',
            'global_efficiency_fc', 'global_efficiency_sc', 'modularity_fc', 'modularity_sc'))

# load msn metrics
data <- np$load('surface_metrics.npy')
# remove the 41, 285, 287 in second dimension of data
data <- data[, -c(41, 285, 287),]
# remove the idx in the second dimension of data
data <- data[,-idx ,]
struc_measures <- array(data, dim = c(309, 2800))
# convert the struc_measures to dataframe
struc_measures <- as.data.frame(struc_measures)
rm(data)

# load the roi labels
annot <- read.csv("/Volumes/wjf-2/MSN_FC_SC/lut_schaefer-400_mics.csv")
annot <- annot[-c(1,202), ]
# remove the string '7Networks_' in the label column of annot
annot$label <- gsub("7Networks_", "", annot$label)
annot$label <- gsub("LH", "L", annot$label)
annot$label <- gsub("RH", "R", annot$label)
annot$label <- gsub("DorsAttn", "DA", annot$label)
annot$label <- gsub("Default", "DF", annot$label)
annot$label <- gsub("SalVentAttn", "SVA", annot$label)
annot$label <- gsub("Limbic", "LB", annot$label)
annot$label <- gsub("Cont", "CON", annot$label)
annot$label <- gsub("SomMot", "SM", annot$label)
annot$label <- gsub("Vis", "VS", annot$label)
roi_labels <- annot$label

# structure labels
struc_labels <- c('Volume', 'Area', 'Thickness', 'Mean_Curvature', 'Gaussian_Curvature', 'FA', 'MD')

# Repeat struc_labels 400 times
struc_labels <- rep(struc_labels, each = 400)

# Combine the labels one by one
combined_labels <- paste(roi_labels, struc_labels, sep = "_")

# graph labels
graph_labels <- c('Clustering_FC', 'Clustering_SC', 'Local_Efficiency_FC', 'Local_Efficiency_SC', 'Degree_FC', 'Degree_SC',
                  'Betweenness_FC', 'Betweenness_SC')
graph_labels2 <-c('Characteristic_Path_Length_FC', 'Characteristic_Path_Length_SC',
                  'Global_Efficiency_FC', 'Global_Efficiency_SC', 'Modularity_FC', 'Modularity_SC')

# repeat the graph_labels 400 times for each element
graph_labels <- rep(graph_labels, each = 400)

# Combine the labels one by one
combined_labels2 <- paste(roi_labels, graph_labels, sep = "_")

# add the graph_labels2 to the combined_labels2
combined_labels2 <- c(combined_labels2, graph_labels2)

# add the combined labels to the dataframe
colnames(struc_measures) <- combined_labels
colnames(graph_measures) <- combined_labels2

# combine the two dataframes
combined_measures <- cbind(struc_measures, graph_measures)

# combine the demo with the combined_measures
features <- cbind(demo$group, demo$age, demo$volume, demo$gender, demo$heredity,demo$BPRS,
                           demo$GAS, demo$HAMA, demo$HAMD, demo$YMRS,demo$school,demo$PDD,demo$PSUI,
                           demo$FAMDD,demo$FAMSUI,demo$FAMSP,demo$FAMSUB, combined_measures)
rm(struc_measures,graph_measures)

# set the column names of first four columns
colnames(features)[1:17] <- c('Group', 'Age','Volume', 'Gender', 'Heredity', 'BPRS', 'GAS', 'HAMA', 'HAMD', 'YMRS',
                              'Education', 'PDD', 'PSUI', 'FAMDD', 'FAMSUI', 'FAMSP', 'FAMSUB')
# features <- features[rows,,]

# turn the combined_measures into model matrix
model_mat <- model.matrix(Group ~ ., data = features)
rm(features)
# remove the intercept column
model_mat <- model_mat[, -1]

#%% build the glmnet model ______________________________________________________________________
set.seed(3) # For reproducibility

# demo <- demo[rows,]
# convert the demo$group as factor
group <- as.factor(demo$group)

# split the data into training and testing sets
split_indices <- createDataPartition(group, p = 0.8, list = FALSE)
X_train <- model_mat[split_indices, ]
Y_train <- group[split_indices]
X_test <- model_mat[-split_indices, ]
Y_test <- group[-split_indices]

# Set up the trainControl object for cross-validation
train_control <- trainControl(
  method = "cv",
  number = 10,
  search = "grid",
  allowParallel = TRUE
)

# Fit the glmnet model using caret with automatic parameter search
model <- train(
  x = X_train,
  y = Y_train,
  family = "multinomial",
  type.multinomial = "ungrouped",
  metric = "Accuracy",
  method = "glmnet",
  trControl = train_control,
  tuneLength = 50,
  preProc = c("center", "scale")
)

# Print the best model
print(model$bestTune)

# Extract the results
results <- model$results

# Find the best combination of alpha and lambda
best_result <- results[which.max(results$Accuracy), ]

# Create a new column in results for marker colors based on Accuracy
results$color <- scales::col_numeric("YlOrRd", domain = results$Accuracy)(results$Accuracy)

# Create the plot
fig1 <- plot_ly(results, x = ~alpha, y = ~log(lambda), z = ~Accuracy, type = "scatter3d", mode = "markers",
               marker = list(size = 2, color = ~color, colorscale = "YlOrRd", opacity = 1, showscale = FALSE),
               hoverinfo = "text",
               text = ~paste('Alpha: ', alpha, '<br>log(Lambda): ', log(lambda), '<br>Accuracy: ', Accuracy)) %>%
  layout(scene = list(xaxis = list(title = "Alpha"),
                      yaxis = list(title = "log(Lambda)"),
                      zaxis = list(title = "Accuracy")),
         title = "3D scatter plot of glmnet tuning parameters",
         font = list(size = 14))

# Add a trace for the best combination of alpha and lambda
fig1 <- fig1 %>% 
  add_trace(x = best_result$alpha, y = log(best_result$lambda), z = best_result$Accuracy, 
            type = "scatter3d", mode = "markers",
            marker = list(symbol = "diamond", size = 6, color = "blue", opacity = 1),
            hoverinfo = "text",
            text = ~paste('Best Alpha: ', alpha, '<br>Best log(Lambda): ', log(lambda), 
                          '<br>Best Accuracy: ', Accuracy))

fig1 <- fig1 %>% 
  add_text(
    x = best_result$alpha,
    y = log(best_result$lambda),
    z = best_result$Accuracy,
    text = paste('Best Result: ', '<br>Alpha: ', round(best_result$alpha,3), '<br>Lambda: ',
                 round(best_result$lambda,3), '<br>Accuracy: ', round(best_result$Accuracy,3)),
    mode = "text",
    type = "scatter3d",
    textfont = list(color = "#000000", size = 14)
  )


# Make predictions on the test set
predictions <- predict(model, X_test)

# Calculate the accuracy of the predictions
accuracy1 <- mean(predictions == Y_test)
print(accuracy1)

# Initialize a color scheme
colors <- c("#FF4040","#005800","#FFBF00","#9A32CD","#CD5B45")

# Plotting ROC curve for each class
prediction_probabilities <- predict(model, X_test, type = "prob")

# Plotting ROC curve for each class
png("aged/ROC_plot_ALLfeatures.png", width = 2000, height = 1800, res = 300)
par(mar = c(5, 4, 4, 5) + 0.1) # Set margins
# Initialize an empty vector to store AUC values
auc_values <- vector()
for (i in seq_along(unique(Y_test))) {
  # Convert labels to binary: 'class' and 'other'
  binary_labels <- ifelse(Y_test == levels(Y_test)[i], levels(Y_test)[i], 'other')
  roc_obj <- roc(response = binary_labels,
                 predictor = prediction_probabilities[,i],
                 levels = c(levels(Y_test)[i], 'other'))
  auc_values <- c(auc_values, round(auc(roc_obj), 2)) # Store rounded AUC values
  if(i == 1){
    plot(roc_obj, col = colors[i], lwd = 2, main = "ROC Curves for Model Predictions", 
         xlab = "False Positive Rate", ylab = "True Positive Rate",
        cex.main = 1.7, cex.lab = 1.7, cex.axis = 1.7)
  }else{
    lines(roc_obj, col = colors[i], lwd = 2)
  }
}
# Create a legend text with class names and corresponding AUC values
legend_text <- paste0("AUC for ", levels(Y_test), ": ", auc_values)
legend(x = "bottomright", legend = legend_text, fill = colors, bty = "n", title = "Class & AUC",
       text.font = 1, cex = 1.5)
dev.off()


# Extract the coefficients of the best model
best_model_coefficients <- coef(model$finalModel, model$bestTune$lambda)

# Plot the coefficients
png("aged/allFeatures_plot.png", width = 2000, height = 1600, res = 300)
plot(model$finalModel, xvar = "lambda", label = TRUE, type.coef = "2norm")
dev.off()

# Extract BP from best_model_coefficients
labels <- rownames(best_model_coefficients[[1]])[-1]
bp_coefficients <- as.data.frame(as.matrix(best_model_coefficients[[1]]))
bp_coefficients <- as.data.frame(bp_coefficients[-1,])
rownames(bp_coefficients) <- labels
colnames(bp_coefficients) <- 'BD_coefficients'

hc_coeficients <- as.data.frame(as.matrix(best_model_coefficients[[2]]))
hc_coeficients <- as.data.frame(hc_coeficients[-1,])
rownames(hc_coeficients) <- labels
colnames(hc_coeficients) <- 'HC_coefficients'

nosub_coeficients <- as.data.frame(as.matrix(best_model_coefficients[[3]]))
nosub_coeficients <- as.data.frame(nosub_coeficients[-1,])
rownames(nosub_coeficients) <- labels
colnames(nosub_coeficients) <- 'nOBDs_coefficients'

pano_coeficients <- as.data.frame(as.matrix(best_model_coefficients[[4]]))
pano_coeficients <- as.data.frame(pano_coeficients[-1,])
rownames(pano_coeficients) <- labels
colnames(pano_coeficients) <- 'OBDns_coefficients'

pasub_coeficients <- as.data.frame(as.matrix(best_model_coefficients[[5]]))
pasub_coeficients <- as.data.frame(pasub_coeficients[-1,])
rownames(pasub_coeficients) <- labels
colnames(pasub_coeficients) <- 'OBDs_coefficients'

# sort the bp_coefficients
indices <- order(abs(bp_coefficients[,1]), decreasing = TRUE)
bp_coef_sort <- as.data.frame(bp_coefficients[indices, ])
rownames(bp_coef_sort) <- rownames(bp_coefficients)[indices]
colnames(bp_coef_sort) <- 'BD_coefficients'
write.csv(bp_coef_sort, file = 'aged/bp_coef_sort.csv')

# sort the hc_coefficients
indices <- order(abs(hc_coeficients[,1]), decreasing = TRUE)
hc_coef_sort <- as.data.frame(hc_coeficients[indices, ])
rownames(hc_coef_sort) <- rownames(hc_coeficients)[indices]
colnames(hc_coef_sort) <- 'HC_coefficients'
write.csv(hc_coef_sort, file = 'aged/hc_coef_sort.csv')

# sort the nosub_coefficients
indices <- order(abs(nosub_coeficients[,1]), decreasing = TRUE)
nosub_coef_sort <- as.data.frame(nosub_coeficients[indices, ])
rownames(nosub_coef_sort) <- rownames(nosub_coeficients)[indices]
colnames(nosub_coef_sort) <- 'nOBDs_coefficients'
write.csv(nosub_coef_sort, file = 'aged/nosub_coef_sort.csv')

# sort the pano_coefficients
indices <- order(abs(pano_coeficients[,1]), decreasing = TRUE)
pano_coef_sort <- as.data.frame(pano_coeficients[indices, ])
rownames(pano_coef_sort) <- rownames(pano_coeficients)[indices]
colnames(pano_coef_sort) <- 'OBDns_coefficients'
write.csv(pano_coef_sort, file = 'aged/pano_coef_sort.csv')

# sort the pasub_coefficients
indices <- order(abs(pasub_coeficients[,1]), decreasing = TRUE)
pasub_coef_sort <- as.data.frame(pasub_coeficients[indices, ])
rownames(pasub_coef_sort) <- rownames(pasub_coeficients)[indices]
colnames(pasub_coef_sort) <- 'OBDs_coefficients'
write.csv(pasub_coef_sort, file = 'aged/pasub_coef_sort.csv')

# Combine the coefficients into one dataframe
coefficients1 <- cbind(bp_coefficients, hc_coeficients, nosub_coeficients, pano_coeficients, pasub_coeficients)
colnames(coefficients1) <- c('BD', 'HC', 'nOBDs', 'OBDns', 'OBDs')
write.csv(coefficients1, file = 'aged/coefficients.csv')

# word cloud
png("aged/BP_ALLfeatures_plot.png", width = 2500, height = 2500, res = 300)
df <- subset(bp_coefficients, bp_coefficients[,1] != 0)
wordcloud(words = rownames(df), freq = abs(df$BD_coefficients),
          random.order=TRUE,
          colors=brewer.pal(8, "Dark2"))

dev.off()

png("aged/HC_ALLfeatures_plot.png", width = 1500, height = 1500, res = 300)
df <- subset(hc_coeficients, hc_coeficients[,1] != 0)
rownames(df)[rownames(df) == "HeredityNH"] <- "Heredity"
wordcloud(words = rownames(df), freq = abs(df$HC_coefficients),
          random.order=TRUE,
          colors = brewer.pal(8, "Dark2"))
dev.off()

png("aged/NOSUB_ALLfeatures_plot.png", width = 2000, height = 2000, res = 300)
df <- subset(nosub_coeficients, nosub_coeficients[,1] != 0)
rownames(df)[rownames(df) == "HeredityNH"] <- "Heredity"
wordcloud(words = rownames(df), freq = abs(df$nOBDs_coefficients),
          random.order=TRUE,
          colors = brewer.pal(8, "Dark2"))
dev.off()

png("aged/PANO_ALLfeatures_plot.png", width = 1500, height = 1500, res = 300)
df <- subset(pano_coeficients, pano_coeficients[,1] != 0)
rownames(df)[rownames(df) == "HeredityNH"] <- "Heredity"
wordcloud(words = rownames(df), freq = abs(df$OBDns_coefficients),
          random.order=TRUE,
          colors = brewer.pal(8, "Dark2"))
dev.off()

png("aged/PASUB_ALLfeatures_plot.png", width = 1000, height = 1000, res = 200)
df <- subset(pasub_coeficients, pasub_coeficients[,1] != 0)
rownames(df)[rownames(df) == "HeredityNH"] <- "Heredity"
wordcloud(words = rownames(df), freq = abs(df$OBDs_coefficients),
          random.order=TRUE,
          colors = brewer.pal(8, "Dark2"))
dev.off()

# confusion matrix
cm <- confusionMatrix(predictions, Y_test)
print(cm)

# visualize the confusion matrix
png("aged/confusion_matrix.png", width = 2000, height = 2000, res = 300)
color_map <- colorRampPalette(c("white", "navyblue"))(25)
heatmap.2(cm$table,
          col = color_map,
          margins = c(10, 10),
          Rowv = NA,
          Colv = NA,
          cex.main = 0.5,         # size of title
          notecol="magenta",      # change font color of cell labels to black
          density.info="none",  # turn off density plot inside color legend
          trace="none",         # turn off trace lines inside the heat map
          cexRow=1.5,             # size of row labels
          cexCol=1.5,             # size of column labels
          labRow = rownames(cm$table),  # labels for rows
          labCol = colnames(cm$table),  # labels for columns
          key = FALSE,           # show color key
          dendrogram = "none",  # don't draw a dendrogram
          cellnote = cm$table,
          notecex=1.5)  # add cell labels
# Add a subtitle
mtext(paste("Overall Accuracy: ", round(accuracy1, 2)), side = 3, line = -4.2, at = 0.5, cex = 1, font = 2)
mtext('Confusion Matrix', side = 3, line = -3, at = 0.5, cex = 1.5, font = 2)
dev.off()

# save out
cm_stats <- list(
  matrix = cm$table,
  overall = cm$overall,
  by_class = cm$byClass
)
output_file <- "aged/confusion_matrix_statistics.txt"
sink(output_file)
cat("\n\n\nConfusion Matrix:\n")
print(cm_stats$matrix)
cat("\n\n\nOverall Statistics:\n")
print(cm_stats$overall)
cat("\n\n\nStatistics by Class:\n")
print(cm_stats$by_class)
sink()


#%% ----------------------------------------------
# alternative way to do the glmnet
# combine the demo with the combined_measures
features2 <- cbind(demo$group, demo$age, demo$volume, demo$gender, demo$school,combined_measures)
# rm(combined_measures)

# set the column names of first four columns
colnames(features2)[1:5] <- c('group', 'Age','Volume', 'Gender','Education')

# turn the combined_measures into model matrix
model_mat <- model.matrix(group ~ ., data = features2)
rm(features2)
# remove the intercept column
model_mat <- model_mat[, -1]

# split the data into training and testing sets
X_train <- model_mat[split_indices, ]
Y_train <- group[split_indices]
X_test <- model_mat[-split_indices, ]
Y_test <- group[-split_indices]

# Set up the trainControl object for cross-validation
train_control2 <- trainControl(
  method = "cv",
  number = 10,
  search = "grid",
  allowParallel = TRUE,
)

# Fit the glmnet model using caret with automatic parameter search
model2 <- train(
  x = X_train,
  y = Y_train,
  family = "multinomial",
  type.multinomial = "ungrouped",
  metric = "Accuracy",
  method = "glmnet",
  trControl = train_control2,
  tuneLength = 50,
  preProc = c("center", "scale")
)

# Print the best model
print(model2$bestTune)

# Extract the results
results <- model2$results

# Find the best combination of alpha and lambda
best_result <- results[which.max(results$Accuracy), ]

# Create a new column in results for marker colors based on Accuracy
results$color <- scales::col_numeric("YlOrRd", domain = results$Accuracy)(results$Accuracy)

# Create the plot
fig <- plot_ly(results, x = ~alpha, y = ~log(lambda), z = ~Accuracy, type = "scatter3d", mode = "markers",
               marker = list(size = 2, color = ~color, colorscale = "YlOrRd", opacity = 1, showscale = FALSE),
               hoverinfo = "text",
               text = ~paste('Alpha: ', alpha, '<br>log(Lambda): ', log(lambda), '<br>Accuracy: ', Accuracy)) %>%
  layout(scene = list(xaxis = list(title = "Alpha", tickcolor = "black"),
                      yaxis = list(title = "log(Lambda)", tickcolor = "black"),
                      zaxis = list(title = "Accuracy", tickcolor = "black")),
         title = "3D scatter plot of glmnet tuning parameters",
         font = list(size = 14))

# Add a trace for the best combination of alpha and lambda
fig <- fig %>% 
  add_trace(x = best_result$alpha, y = log(best_result$lambda), z = best_result$Accuracy, 
            type = "scatter3d", mode = "markers",
            marker = list(symbol = "diamond", size = 6, color = "blue", opacity = 1),
            hoverinfo = "text",
            text = ~paste('Best Alpha: ', alpha, '<br>Best log(Lambda): ', log(lambda), 
                          '<br>Best Accuracy: ', Accuracy))

fig <- fig %>% 
  add_text(
    x = best_result$alpha,
    y = log(best_result$lambda),
    z = best_result$Accuracy,
    text = paste('Best Result: ', '<br>Alpha: ', round(best_result$alpha,3), '<br>Lambda: ',
                 round(best_result$lambda,3), '<br>Accuracy: ', round(best_result$Accuracy,3)),
    mode = "text",
    type = "scatter3d",
    textfont = list(color = "#000000", size = 14)
  )


# print the training accuracy
print(model2$results$Accuracy)

# Make predictions on the test set
predictions2 <- predict(model2, X_test)

# Calculate the accuracy of the predictions
accuracy2 <- mean(predictions2 == Y_test)
print(accuracy2)

# Extract the coefficients of the best model
best_model_coefficients <- coef(model2$finalModel, model2$bestTune$lambda)

# Plotting ROC curve for each class
prediction_probabilities <- predict(model2, X_test, type = "prob")
colors <- c("#FF4040","#005800","#FFBF00","#9A32CD","#CD5B45")
png("aged/ROC_plot_MRI_features.png", width = 2000, height = 1800, res = 300)
par(mar = c(5, 4, 4, 5) + 0.1) # Set margins
# Initialize an empty vector to store AUC values
auc_values <- vector()
for (i in seq_along(unique(Y_test))) {
  # Convert labels to binary: 'class' and 'other'
  binary_labels <- ifelse(Y_test == levels(Y_test)[i], levels(Y_test)[i], 'other')
  roc_obj <- roc(response = binary_labels,
                 predictor = prediction_probabilities[,i],
                 levels = c(levels(Y_test)[i], 'other'))
  auc_values <- c(auc_values, round(auc(roc_obj), 2)) # Store rounded AUC values
  if(i == 1){
    plot(roc_obj, col = colors[i], lwd = 2, main = "ROC Curves for Model Predictions",
         xlab = "False Positive Rate", ylab = "True Positive Rate",
        cex.main = 1.7, cex.lab = 1.7, cex.axis = 1.7)
  }else{
    lines(roc_obj, col = colors[i], lwd = 2)
  }
}
# Create a legend text with class names and corresponding AUC values
legend_text <- paste0("AUC for ", levels(Y_test), ": ", auc_values)
legend(x = "bottomright", legend = legend_text, fill = colors, bty = "n", title = "Class & AUC", text.font = 1,
       cex = 1.5)
dev.off()


# Plot the coefficients
png("aged/MRI_features_plot.png", width = 2000, height = 1800, res = 300)
plot(model2$finalModel, xvar = "lambda", label = TRUE, type.coef = "2norm")
dev.off()

# Extract BP from best_model_coefficients
labels <- rownames(best_model_coefficients[[1]])[-1]
bp_coefficients <- as.data.frame(as.matrix(best_model_coefficients[[1]]))
bp_coefficients <- as.data.frame(bp_coefficients[-1,])
rownames(bp_coefficients) <- labels
colnames(bp_coefficients) <- 'BD_coefficients'

hc_coeficients <- as.data.frame(as.matrix(best_model_coefficients[[2]]))
hc_coeficients <- as.data.frame(hc_coeficients[-1,])
rownames(hc_coeficients) <- labels
colnames(hc_coeficients) <- 'HC_coefficients'

nosub_coeficients <- as.data.frame(as.matrix(best_model_coefficients[[3]]))
nosub_coeficients <- as.data.frame(nosub_coeficients[-1,])
rownames(nosub_coeficients) <- labels
colnames(nosub_coeficients) <- 'nOBDs_coefficients'

pano_coeficients <- as.data.frame(as.matrix(best_model_coefficients[[4]]))
pano_coeficients <- as.data.frame(pano_coeficients[-1,])
rownames(pano_coeficients) <- labels
colnames(pano_coeficients) <- 'OBDns_coefficients'

pasub_coeficients <- as.data.frame(as.matrix(best_model_coefficients[[5]]))
pasub_coeficients <- as.data.frame(pasub_coeficients[-1,])
rownames(pasub_coeficients) <- labels
colnames(pasub_coeficients) <- 'OBDs_coefficients'

# word cloud
png("aged/BP_MRIfeatures_plot.png", width = 2000, height = 2000, res = 300)
df <- subset(bp_coefficients, bp_coefficients[,1] != 0)
wordcloud(words = rownames(df), freq = abs(df$BD_coefficients),
         random.order=TRUE,
          colors=brewer.pal(8, "Dark2"))
dev.off()

png("aged/HC_MRIfeatures_plot.png", width = 2000, height = 2000, res = 300)
df <- subset(hc_coeficients, hc_coeficients[,1] != 0)
wordcloud(words = rownames(df), freq = abs(df$HC_coefficients),
          random.order=TRUE,
          colors = brewer.pal(8, "Dark2"))
dev.off()

png("aged/NOSUB_MRIfeatures_plot.png", width = 2000, height = 2000, res = 300)
df <- subset(nosub_coeficients, nosub_coeficients[,1] != 0)
wordcloud(words = rownames(df), freq = abs(df$nOBDs_coefficients),
          random.order=TRUE,
          colors = brewer.pal(8, "Dark2"))
dev.off()

png("aged/PANO_MRIfeatures_plot.png", width = 2000, height = 2000, res = 300)
df <- subset(pano_coeficients, pano_coeficients[,1] != 0)
wordcloud(words = rownames(df), freq = abs(df$OBDns_coefficients),
          random.order=TRUE,
          colors = brewer.pal(8, "Dark2"))
dev.off()

png("aged/PASUB_MRIfeatures_plot.png", width = 2000, height = 2000, res = 300)
df <- subset(pasub_coeficients, pasub_coeficients[,1] != 0)
wordcloud(words = rownames(df), freq = abs(df$OBDs_coefficients),
          random.order=TRUE,
          colors = brewer.pal(8, "Dark2"))
dev.off()

# sort the bp_coefficients
indices <- order(abs(bp_coefficients[,1]), decreasing = TRUE)
bp_coef_sort <- as.data.frame(bp_coefficients[indices, ])
rownames(bp_coef_sort) <- rownames(bp_coefficients)[indices]
colnames(bp_coef_sort) <- 'BD_coefficients'
write.csv(bp_coef_sort, file = 'aged/bp_coef_sort2.csv')

# sort the hc_coefficients
indices <- order(abs(hc_coeficients[,1]), decreasing = TRUE)
hc_coef_sort <- as.data.frame(hc_coeficients[indices, ])
rownames(hc_coef_sort) <- rownames(hc_coeficients)[indices]
colnames(hc_coef_sort) <- 'HC_coefficients'
write.csv(hc_coef_sort, file = 'aged/hc_coef_sort2.csv')

# sort the nosub_coefficients
indices <- order(abs(nosub_coeficients[,1]), decreasing = TRUE)
nosub_coef_sort <- as.data.frame(nosub_coeficients[indices, ])
rownames(nosub_coef_sort) <- rownames(nosub_coeficients)[indices]
colnames(nosub_coef_sort) <- 'nOBDs_coefficients'
write.csv(nosub_coef_sort, file = 'aged/nosub_coef_sort2.csv')

# sort the pano_coefficients
indices <- order(abs(pano_coeficients[,1]), decreasing = TRUE)
pano_coef_sort <- as.data.frame(pano_coeficients[indices, ])
rownames(pano_coef_sort) <- rownames(pano_coeficients)[indices]
colnames(pano_coef_sort) <- 'OBDns_coefficients'
write.csv(pano_coef_sort, file = 'aged/pano_coef_sort2.csv')

# sort the pasub_coefficients
indices <- order(abs(pasub_coeficients[,1]), decreasing = TRUE)
pasub_coef_sort <- as.data.frame(pasub_coeficients[indices, ])
rownames(pasub_coef_sort) <- rownames(pasub_coeficients)[indices]
colnames(pasub_coef_sort) <- 'OBDs_coefficients'
write.csv(pasub_coef_sort, file = 'aged/pasub_coef_sort2.csv')

# Combine the coefficients into one data frame
coefficients2 <- cbind(bp_coefficients, hc_coeficients, nosub_coeficients, pano_coeficients, pasub_coeficients)
colnames(coefficients2) <- c('BP', 'HC', 'nOBDs', 'OBDns', 'OBDs')
write.csv(coefficients2, file = 'aged/coefficients_MRIfeatures.csv')

# confusion matrix
cm2 <- confusionMatrix(predictions2, Y_test)
print(cm2)

# visualize the confusion matrix
png("aged/confusion_matrix_MRI.png", width = 2000, height = 2000, res = 300)
color_map <- colorRampPalette(c("white", "navyblue"))(25)
heatmap.2(cm2$table,
          col = color_map,
          margins = c(10, 10),
          Rowv = NA,
          Colv = NA,
          notecol="magenta",      # change font color of cell labels to black
          density.info="none",  # turn off density plot inside color legend
          trace="none",         # turn off trace lines inside the heat map
          cexRow=1.5,             # size of row labels
          cexCol=1.5,             # size of column labels
          labRow = rownames(cm$table),  # labels for rows
          labCol = colnames(cm$table),  # labels for columns
          key = FALSE,           # show color key
          dendrogram = "none",  # don't draw a dendrogram
          cellnote = cm$table,
          notecex=1.5)  # add cell labels
# Add a subtitle
mtext(paste("Overall Accuracy: ", round(accuracy2, 2)), side = 3, line = -4.2, at = 0.5, cex = 1, font = 2)
mtext('Confusion Matrix', side = 3, line = -3, at = 0.5, cex = 1.5, font = 2)
dev.off()

# save out
cm_stats <- list(
  matrix = cm2$table,
  overall = cm2$overall,
  by_class = cm2$byClass)

output_file <- "aged/confusion_matrix_statistics_MRI.txt"
sink(output_file)
cat("\n\n\nConfusion Matrix:\n")
print(cm_stats$matrix)
cat("\n\n\nOverall Statistics:\n")
print(cm_stats$overall)
cat("\n\n\nStatistics by Class:\n")
print(cm_stats$by_class)
sink()



# 
# #%% cross validations
# # combine the demo with the combined_measures
# features <- cbind(demo$group, demo$age, demo$volume, demo$gender, demo$heredity,demo$BPRS,
#                   demo$GAS, demo$HAMA, demo$HAMD, demo$YMRS,demo$school,demo$PDD,demo$PSUI,
#                   demo$FAMDD,demo$FAMSUI,demo$FAMSP,demo$FAMSUB, combined_measures)
# 
# # set the column names of first four columns
# colnames(features)[1:17] <- c('Group', 'Age','Volume', 'Gender', 'Heredity', 'BPRS', 'GAS', 'HAMA', 'HAMD', 'YMRS',
#                               'Education', 'PDD', 'PSUI', 'FAMDD', 'FAMSUI', 'FAMSP', 'FAMSUB')
# # features <- features[rows,,]
# 
# # turn the combined_measures into model matrix
# model_mat <- model.matrix(Group ~ ., data = features)
# rm(features)
# # remove the intercept column
# model_mat <- model_mat[, -1]
# 
# ## cross validations
# # Pre-allocate a vector of length 1000
# acc_cross1 <- vector("numeric", 100)
# # set up a loop for 1000 times
# for (i in 1:100){
#   # split the data into training and testing sets
#   split_indices <- createDataPartition(group, p = 0.8, list = FALSE)
#   X_train <- model_mat[split_indices, ]
#   Y_train <- group[split_indices]
#   X_test <- model_mat[-split_indices, ]
#   Y_test <- group[-split_indices]
# 
#   # Set up the trainControl object for cross-validation
#   train_control <- trainControl(
#     method = "cv",
#     number = 10,
#     search = "grid",
#     allowParallel = TRUE
#   )
# 
#   # Fit the glmnet model using caret with automatic parameter search
#   model <- train(
#     x = X_train,
#     y = Y_train,
#     family = "multinomial",
#     type.multinomial = "ungrouped",
#     metric = "Accuracy",
#     method = "glmnet",
#     trControl = train_control,
#     tuneLength = 50,
#     preProc = c("center", "scale")
#   )
#   # Make predictions on the test set
#   predictions <- predict(model, X_test)
#   # Calculate the accuracy of the predictions
#   accuracy <- mean(predictions == Y_test)
#   print(paste("Iteration", i, "Accuracy", accuracy))
#   acc_cross1[i] <- accuracy
# }
# 
# # plot the acc_cross1
# png("aged/acc_cross1.png", width = 2000, height = 2000, res = 300)
# plot(acc_cross1, type = 'l', xlab = 'Number of iterations', ylab = 'Accuracy', main = 'Cross validation accuracy')
# dev.off()
# 
# #%% cross validations
# # combine the demo with the combined_measures
# features2 <- cbind(demo$group, demo$age, demo$volume, demo$gender, demo$school,combined_measures)
# # rm(combined_measures)
# 
# # set the column names of first four columns
# colnames(features2)[1:5] <- c('group', 'Age','Volume', 'Gender','Education')
# 
# # turn the combined_measures into model matrix
# model_mat <- model.matrix(group ~ ., data = features2)
# rm(features2)
# # remove the intercept column
# model_mat <- model_mat[, -1]
# 
# # Pre-allocate a vector of length 1000
# acc_cross2 <- vector("numeric", 100)
# 
# # set up a loop for 1000 times
# for (i in 1:100){
#   # split the data into training and testing sets
#   split_indices <- createDataPartition(group, p = 0.8, list = FALSE)
#   X_train <- model_mat[split_indices, ]
#   Y_train <- group[split_indices]
#   X_test <- model_mat[-split_indices, ]
#   Y_test <- group[-split_indices]
# 
#   # Set up the trainControl object for cross-validation
#   train_control <- trainControl(
#     method = "cv",
#     number = 10,
#     search = "grid",
#     allowParallel = TRUE
#   )
# 
#   # Fit the glmnet model using caret with automatic parameter search
#   model <- train(
#     x = X_train,
#     y = Y_train,
#     family = "multinomial",
#     type.multinomial = "ungrouped",
#     metric = "Accuracy",
#     method = "glmnet",
#     trControl = train_control,
#     tuneLength = 50,
#     preProc = c("center", "scale")
#   )
# 
#   # Make predictions on the test set
#   predictions <- predict(model, X_test)
#   # Calculate the accuracy of the predictions
#   accuracy <- mean(predictions == Y_test)
#   # print the iteration and accuracy
#   print(paste("Iteration", i, "Accuracy", accuracy))
#   acc_cross2[i] <- accuracy
# }
# 
# # plot the acc_cross2
# png("aged/acc_cross2.png", width = 2000, height = 2000, res = 300)
# plot(acc_cross2, type = 'l', xlab = 'Number of iterations', ylab = 'Accuracy', main = 'Cross validation accuracy')
# dev.off()
# 
# # plot the acc_cross1 and acc_cross2 together with different colors
# png("aged/acc_cross1_cross2.png", width = 2000, height = 2000, res = 300)
# plot(acc_cross1, type = 'p', col = "blue", xlab = 'Number of iterations', ylab = 'Accuracy', main = 'Cross validation accuracy',
#      lwd = 2, ylim = c(0.5, 0.92))
# lines(acc_cross2, type = "p", col = "green", lwd = 2)
# # Add dashed lines for mean
# mean_y1 <- mean(acc_cross1)
# mean_y2 <- mean(acc_cross2)
# abline(h = mean(acc_cross1), col = "red", lty = 2)
# abline(h = mean(acc_cross2), col = "red", lty = 2)
# # axis(2, at = c(mean_y1, mean_y2), labels = c(round(mean_y1, 2), round(mean_y2, 2)))
# legend("topright", legend = c("All features", "MRI features"), col = c("blue", "green"), lwd = 2, bty = "n")
# legend('bottomright', legend = "Mean accuracy", col = "red", lty = 2, bty = "n")
# title("Cross validation accuracy")
# dev.off()
