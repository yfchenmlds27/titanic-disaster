# Titanic survival prediction with logistic regression (R)
suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
})

data_dir   <- file.path("src", "data")
train_path <- file.path(data_dir, "train.csv")
test_path  <- file.path(data_dir, "test.csv")
out_path   <- file.path(data_dir, "predictions_r.csv")

section <- function(title) {
  cat("\n", strrep("=", 60), "\n", title, "\n", strrep("=", 60), "\n", sep = "")
}

common_titles <- c("Mr", "Mrs", "Miss", "Master")
title_levels  <- c(common_titles, "Rare")

extract_title <- function(name) {
  t <- trimws(sub(".*,\\s*([^.]+)\\..*", "\\1", name))
  t[t %in% c("Mlle", "Ms")] <- "Miss"
  t[t == "Mme"] <- "Mrs"
  t[!t %in% common_titles] <- "Rare"
  factor(t, levels = title_levels)
}

fit_stats <- function(df) {
  emb <- df$Embarked[!is.na(df$Embarked) & df$Embarked != ""]
  list(
    age_by_title = tapply(df$Age, df$Title, median, na.rm = TRUE),
    age_overall  = median(df$Age, na.rm = TRUE),
    fare         = median(df$Fare, na.rm = TRUE),
    embarked     = names(sort(table(emb), decreasing = TRUE))[1]
  )
}

transform_data <- function(df, stats) {
  n <- sum(is.na(df$Age))
  title_age <- stats$age_by_title[as.character(df$Title)]
  df$Age <- ifelse(is.na(df$Age), title_age, df$Age)
  df$Age[is.na(df$Age)] <- stats$age_overall
  cat(sprintf("- Filled %d missing Age values with the median age for each Title\n", n))

  n <- sum(is.na(df$Fare))
  df$Fare[is.na(df$Fare)] <- stats$fare
  cat(sprintf("- Filled %d missing Fare values with median fare (%.2f)\n", n, stats$fare))

  missing_emb <- is.na(df$Embarked) | df$Embarked == ""
  df$Embarked[missing_emb] <- stats$embarked
  cat(sprintf("- Filled %d missing Embarked values with mode ('%s')\n",
              sum(missing_emb), stats$embarked))

  df <- df %>%
    mutate(
      FamilySize = SibSp + Parch + 1,
      IsAlone    = as.integer(FamilySize == 1),
      Sex        = as.integer(Sex == "female"),
      LogFare    = log1p(Fare),
      Pclass     = factor(Pclass, levels = c(1, 2, 3)),
      Embarked   = factor(Embarked, levels = c("C", "Q", "S"))
    )
  cat("- Added FamilySize = SibSp + Parch + 1, and IsAlone = (FamilySize == 1)\n")
  cat("- Encoded Sex as 1 = female, 0 = male\n")
  cat("- Added LogFare = log(1 + Fare) to reduce skew\n")
  cat("- Converted Pclass, Embarked and Title to factors (glm one-hot encodes them)\n")
  cat("- Dropping PassengerId, Name, Ticket, Cabin (mostly missing) and raw Fare from the model\n")
  df
}

# SibSp and Parch are left out: FamilySize = SibSp + Parch + 1 already captures them
# (including all three would make the model rank-deficient)
model_formula <- Survived ~ Pclass + Sex + Age + FamilySize +
  IsAlone + LogFare + Embarked + Title

accuracy <- function(actual, predicted) mean(actual == predicted)

# ---------------------------------------------------------------------------
section("1. Loading training data")
for (p in c(train_path, test_path)) {
  if (!file.exists(p)) stop(paste(p, "not found. See the README for how to download the data."))
}
train <- read_csv(train_path, show_col_types = FALSE)
cat(sprintf("Loaded train.csv: %d rows, %d columns\n", nrow(train), ncol(train)))
print(head(train))

section("2. Exploring the data")
miss <- colSums(is.na(train) | train == "", na.rm = TRUE)
cat("Missing values per column:\n")
print(miss[miss > 0])
cat(sprintf("\nOverall survival rate: %.3f\n", mean(train$Survived)))
cat("\nSurvival rate by Sex:\n")
print(train %>% group_by(Sex) %>% summarise(SurvivalRate = round(mean(Survived), 3)))
cat("\nSurvival rate by Pclass:\n")
print(train %>% group_by(Pclass) %>% summarise(SurvivalRate = round(mean(Survived), 3)))

section("3. Adjusting the data")
train$Title <- extract_title(train$Name)
cat("- Extracted Title from Name (uncommon titles grouped as 'Rare'):\n")
print(table(train$Title))
stats <- fit_stats(train)
train <- transform_data(train, stats)

section("4. Training logistic regression")
# Kaggle's test.csv has no Survived labels, so hold out 20% of train.csv
# as a validation set to estimate out-of-sample accuracy.
set.seed(42)
val_idx <- sample(seq_len(nrow(train)), size = round(0.2 * nrow(train)))
tr  <- train[-val_idx, ]
val <- train[val_idx, ]
cat(sprintf("Split train.csv into %d training rows and %d validation rows\n", nrow(tr), nrow(val)))
model <- glm(model_formula, data = tr, family = binomial)
cat("Model fitted. Coefficients:\n")
print(round(coef(model), 3))

section("5. Accuracy on the training set")
train_acc <- accuracy(tr$Survived, as.integer(predict(model, tr, type = "response") > 0.5))
val_acc   <- accuracy(val$Survived, as.integer(predict(model, val, type = "response") > 0.5))
cat(sprintf("Training accuracy:            %.4f\n", train_acc))
cat(sprintf("Held-out validation accuracy: %.4f\n", val_acc))

section("6. Loading test.csv and predicting")
test <- read_csv(test_path, show_col_types = FALSE)
cat(sprintf("Loaded test.csv: %d rows, %d columns\n", nrow(test), ncol(test)))
test$Title <- extract_title(test$Name)
cat("Applying the same transformations (using training-set statistics):\n")
test <- transform_data(test, stats)

cat(sprintf("Refitting the model on all %d training rows before predicting test.csv\n", nrow(train)))
model <- glm(model_formula, data = train, family = binomial)
test_pred <- as.integer(predict(model, test, type = "response") > 0.5)
cat(sprintf("Predicted survival for %d passengers\n", length(test_pred)))
cat(sprintf("Predicted survival rate on test set: %.3f\n", mean(test_pred)))

section("7. Accuracy on the test set")
if ("Survived" %in% names(test)) {
  cat(sprintf("Test accuracy: %.4f\n", accuracy(test$Survived, test_pred)))
} else {
  cat("test.csv has no 'Survived' column (Kaggle withholds the labels),\n")
  cat("so true test accuracy cannot be computed locally.\n")
  cat(sprintf("Estimated test accuracy (held-out validation): %.4f\n", val_acc))
  cat("Submit the predictions file to Kaggle for the official test score.\n")
}

write_csv(data.frame(PassengerId = test$PassengerId, Survived = test_pred), out_path)
cat("\nSaved predictions to src/data/predictions_r.csv\nDone!\n")
