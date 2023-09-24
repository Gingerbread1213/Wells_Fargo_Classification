library(xgboost)
library(dplyr)
library(tidymodels)
library(tidyverse)
library(rpart)
library(xgboost)
library(glmnet)
library(LiblineaR)
library(kernlab)
library(kknn)

set.seed(666666)


test <- read.csv("test2.csv")
train <- read.csv("train2.csv")

cleaned_train <- train %>%
  mutate(action_taken = factor(action_taken))

train_folds <- vfold_cv(cleaned_train, v=2, strata = 'action_taken')

recipe <- recipe(action_taken ~ ., data = cleaned_train) %>%
  # step_novel(age_of_applicant_or_borrower) %>%
  
  step_rm(id) %>%
  
  step_rm(legal_entity_identifier_lei) %>%
  
  # unique(test$activity_year)  unique(train$activity_year) 2018
  step_rm(activity_year) %>%
  
  # unique(train$preapproval) unique(test$preapproval)  2
  step_rm(preapproval) %>%
  
  # contain NA value impossible to impute
  step_rm(state) %>%
  
  # impute nominal variable by most common values
  # step_impute_mode(ethnicity_of_applicant_or_borrower_1, ethnicity_of_co_applicant_or_co_borrower_1) %>%
  
  step_rm(ethnicity_of_applicant_or_borrower_2) %>%
  step_rm(ethnicity_of_applicant_or_borrower_3) %>%
  step_rm(ethnicity_of_applicant_or_borrower_4) %>%
  step_rm(ethnicity_of_applicant_or_borrower_5) %>%
  step_rm(ethnicity_of_co_applicant_or_co_borrower_2) %>%
  step_rm(ethnicity_of_co_applicant_or_co_borrower_3) %>%
  step_rm(ethnicity_of_co_applicant_or_co_borrower_4) %>%
  step_rm(ethnicity_of_co_applicant_or_co_borrower_5) %>%
  # step_impute_mode(race_of_applicant_or_borrower_1, race_of_co_applicant_or_co_borrower_1) %>%
  
  step_rm(race_of_applicant_or_borrower_2, race_of_applicant_or_borrower_3,
          race_of_applicant_or_borrower_4, race_of_applicant_or_borrower_5,
          race_of_co_applicant_or_co_borrower_2, race_of_co_applicant_or_co_borrower_3,
          race_of_co_applicant_or_co_borrower_4, race_of_co_applicant_or_co_borrower_5) %>%
  
  step_impute_mode(age_of_applicant_62, age_of_co_applicant_62) %>%
  
  step_impute_mean(income) %>%
  
  step_rm(total_points_and_fees, prepayment_penalty_term) %>%
  
  step_impute_mean(combined_loan_to_value_ratio, loan_term) %>%
  
  step_rm(introductory_rate_period) %>%
  
  step_impute_mean(property_value) %>%
  
  step_rm(multifamily_affordable_units) %>%
  
  # too many NA values
  step_rm(automated_underwriting_system_2, automated_underwriting_system_3,
          automated_underwriting_system_4, automated_underwriting_system_5) %>%
  
  step_dummy(all_factor_predictors(), one_hot = TRUE)


boost_tree_model<- boost_tree(
  trees = 732, 
  min_n = 12, 
  tree_depth = 12,
  learn_rate = 0.01531878, 
  loss_reduction = 8.004341e-04, 
  sample_size = 0.6441571	,
  stop_iter = 19
) %>%
  set_engine("xgboost") %>% 
  set_mode("classification")

boost_tree_workflow <- workflow() %>% 
  add_model(boost_tree_model) %>% 
  add_recipe(recipe, blueprint = hardhat::default_recipe_blueprint(allow_novel_levels = TRUE))

boost_fit <- boost_tree_workflow %>% 
  fit(cleaned_train)

boost_result <- boost_fit %>% 
  predict(test)

final <- test %>% 
  select(id) %>% 
  bind_cols(boost_result)

final <- final %>%
  select(id, action_taken = .pred_class)

write_csv(final, "boost_result.csv")