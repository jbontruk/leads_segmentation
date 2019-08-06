# Libraries and connection ----
library(dplyr)
library(DataExplorer)
library(RPostgreSQL)
library(xgboost)

# Get data ----
setwd('/Users/jaroslawbontruk/Downloads')
a <- read.csv2('deals-7920291-1282.csv', sep = ',', stringsAsFactors = F)
b <- read.csv2('deals-7920291-1283.csv', sep = ',', stringsAsFactors = F)

# Get additional data ----
setwd('/Users/jaroslawbontruk/GitHub/ML_model_development')
db_pass <- readRDS('../db_pass.RDS')
drv <- dbDriver("PostgreSQL")
db_prod <- dbConnect(drv, dbname = "payability_v2",
                     host = "payability-aurora-cluster.cluster-ro-cfsjjblwuyi4.us-east-1.rds.amazonaws.com", port = 5432,
                     user = "payability", password = db_pass)

a$supplier_key <- ifelse(nchar(a$Deal...SUPPLIER.ID) != 36, NA, a$Deal...SUPPLIER.ID)
b$supplier_key <- ifelse(nchar(b$Deal...SUPPLIER.ID) != 36, NA, b$Deal...SUPPLIER.ID)

a2 <- dbGetQuery(db_prod, paste0("select * from v_supplier_summary where supplier_key in ('",
                                 paste0(unique(na.omit(a$supplier_key)), collapse = "','"),
                                 "')"))
b2 <- dbGetQuery(db_prod, paste0("select * from v_supplier_summary where supplier_key in ('",
                                 paste0(unique(na.omit(b$supplier_key)), collapse = "','"),
                                 "')"))

cols_to_exclude <- c('supplier_site', 'b2b_seller', 'health_ok_days', 'owner_name', 'guarantor', 'supplier_covenants',
                     'fund_facility_limit', 'address', 'timezone', 'hide', 'company_name')
a2 <- a2 %>% select(-cols_to_exclude)
b2 <- b2 %>% select(-cols_to_exclude)

won_deals <- left_join(b, b2, by = "supplier_key")
not_won_deals <- left_join(a, a2, by = "supplier_key")

# Features selection ----
input_cols <- c('Deal...Done.activities',
                'Deal...Email.messages.count',
                'Deal...Inbound.Activity',
                'Deal...Marketing_Channel',
                'Deal...Marketing_Campaign',
                'Deal...Marketing_Medium',
                'Deal...Marketing_Source',
                'Deal...Marketplace',
                'Deal...Sales.History',
                'Deal...Total.activities',
                'Person...Activities.to.do',
                'Person...Closed.deals',
                #'Person...Done.activities',
                #'Person...Email.messages.count',
                'Person...Lost.deals',
                'Person...Open.deals',
                #'Person...Total.activities',
                'Person...Won.deals',
                'Organization...30.Day.Feedback.Count',
                'Organization...30.Day.Feedback.Rating',
                #'Organization...Activities.to.do',
                'Organization...Alexa.Rank',
                #'Organization...Closed.deals',
                #'Organization...Done.activities',
                #'Organization...Email.messages.count',
                'Organization...Employee.Count.Range',
                'Organization...Lifetime.Feedback.Count',
                'Organization...Lifetime.Feedback.Rating',
                #'Organization...Lost.deals',
                #'Organization...Open.deals',
                'Organization...People',
                'Organization...Product.Count',
                #'Organization...Total.activities',
                #'Organization...Won.deals',
                'general_info_updated',
                'legal_signed',
                'state_code',
                #'estimated_monthly_revenue',
                'length_time_selling',
                'time_selling',
                'marketplace_chosen',
                'referer',
                'self_fullfilled_ratio',
                'days_to_go_live',
                'eligibility_type',
                'amazon_lending',
                'last_uw_decision',
                'api_status')

# Remove columns with >95% missing data
cols_to_exclude2 <- c('last_uw_decision', 'self_fullfilled_ratio', 'Organization...Alexa.Rank', 'Organization...Product.Count',
                      'Organization...Lifetime.Feedback.Count', 'Organization...Lifetime.Feedback.Rating', 'Organization...30.Day.Feedback.Count',
                      'Organization...30.Day.Feedback.Rating', 'days_to_go_live', 'amazon_lending')
input_cols <- setdiff(input_cols, cols_to_exclude2)

# Training dataset preparation ----
training <- won_deals %>%
  filter(Deal...Value != 0) %>%
  mutate(Deal...Value = as.numeric(Deal...Value),
         value_class = ifelse(Deal...Value < 10000, 0,
                              ifelse(Deal...Value < 25000, 1,
                                     ifelse(Deal...Value < 50000, 1, 1)))) %>%
  select(input_cols, value_class)

training %>% count(value_class)

smp_size <- floor(0.8 * nrow(training))
set.seed(123)
train_ind <- sample(seq_len(nrow(training)), size = smp_size)
train <- training[train_ind, ]
test <- training[-train_ind, ]

# XGBoost binary model ----
train_matrix <- data.matrix(train)
MCMod <- xgboost(data = subset(train_matrix, select = -value_class), 
                 label = subset(train_matrix, select = value_class), 
                 objective = "binary:logistic", nround = 1000)

test_matrix <- data.matrix(subset(test, select = -value_class))
test$prediction <- predict(MCMod, test_matrix)

check <- test %>% count(value_class, prediction)
View(check)
write.table(check, 'check.csv', row.names = F, sep = "\t")

importance <- xgb.importance(model = MCMod)
write.table(importance, 'importance.csv', row.names = F, sep = "\t")
View(importance)

xgb.plot.importance(importance_matrix = importance)
create_report(not_won_deals[,input_cols])
# Classifying not won deals ----
t_matrix <- data.matrix(not_won_deals[,input_cols])
not_won_deals$prediction <- predict(MCMod, t_matrix)

not_won_deals %>% count(prediction)

not_won_deals_predictions <- not_won_deals %>%
  arrange(desc(prediction))
not_won_deals_predictions <- not_won_deals_predictions[1:25000,] 
write.table(not_won_deals_predictions, 'predictions_sample.csv', row.names = F, sep = "\t")