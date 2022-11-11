# Package loading  ----------------------------------
source("scripts/00. install.R")
library(magrittr)
library(ggplot2)
library(lme4)
library(tidymodels)
library(tidyverse)
library(dbarts)
library(hebartBase)


# Dataset split  ------------------------------------
set.seed(2021)
df_real     <- lme4::sleepstudy %>% set_names(c('y', 'X1', 'group'))
df_real$y   <- df_real$y
data_split  <- initial_split(df_real)
train       <- training(data_split)
test        <- testing(data_split)
groups      <- train$group


data_split <- rsample::vfold_cv(df_real, v = 20) |> 
  dplyr::mutate(
    train = purrr::map(splits, training),
    test  = purrr::map(splits, testing)
  )

# Running the model ----------------------------------
change_sigma_phi <- function(sigma_phi, num_trees, train){
  hb_model <- hebart(formula = y ~ X1,
                     data = train,
                     group_variable = "group", 
                     num_trees = num_trees,
                     priors = list(
                       alpha = 0.95, # Prior control list
                       beta = 2,
                       nu = 2,
                       lambda = 0.1,
                       tau_mu = 16 * num_trees,
                       shape_sigma_phi = 0.5,
                       scale_sigma_phi = 1, 
                       sample_sigma_phi = FALSE
                     ), 
                     inits = list(tau = 1,
                                  sigma_phi = sigma_phi),
                     MCMC = list(iter = 750, 
                                 burn = 250, 
                                 thin = 1,
                                 sigma_phi_sd = 0.5)
  )
  hb_model
}

rmse <- function(model){
  pred <-  predict_hebart(newX = test, new_groups = test$group,
                          hebart_posterior  = model, type = "mean")
  
  rmse_value <- sqrt(mean(pred - test$y)^2)
  return(rmse_value)
}
runs <- list()


for(i in 1:nrow(data_split)){
  print(i)
  train <- data_split$train[[i]]
  values <- seq(0.001, .15, by = 0.01)
  runs[[i]] <- list(values = values, 
                    num_trees = c(5, 10, 15)) |> 
    purrr::cross_df() |> 
    dplyr::mutate(
      model = purrr::map2(values, num_trees, change_sigma_phi, train = train)
    )
}

write_rds(runs, "runs.rds")

rmses <- list()
for(i in 1:nrow(data_split)){
  print(i)
  rmses[[i]] <- runs[[i]] |> 
    dplyr::mutate(
      rmse = purrr::map_dbl(model, rmse)
    ) |> 
    select(-model)
}

write_rds(rmses, "rmses.rds")

df <- bind_rows(rmses) 
df$id <- rep(rep(1:length(values), 20), 3)


df |> 
  group_by(values, num_trees) |> 
  summarise(
    meand = mean(rmse), 
    upp = meand + sd(rmse)/20 * 1.96,
    low = meand - sd(rmse)/20 * 1.96) |> 
  ungroup() |> 
  ggplot(aes(y = meand, x = values)) +
  geom_ribbon(aes(ymin =low, ymax = upp), alpha = 0.3, fill = "lightblue")+
  geom_line() +
  scale_x_continuous(breaks = scales::pretty_breaks(n = 5)) +
  facet_wrap(~num_trees) + 
  geom_point() +
  labs(x = "sigma_phi", y = "mean test RMSE", 
       title = "RMSE per sigma_phi and num_trees values")
# Sampling

hb_model <- hebart(formula = y ~ X1,
                   data = train,
                   group_variable = "group", 
                   num_trees = num_trees,
                   priors = list(
                     alpha = 0.95, # Prior control list
                     beta = 2,
                     nu = 2,
                     lambda = 0.1,
                     tau_mu = 16 * num_trees,
                     shape_sigma_phi = 0.5,
                     scale_sigma_phi = 1, 
                     sample_sigma_phi = FALSE
                   ), 
                   inits = list(tau = 1,
                                sigma_phi = 0.95),
                   MCMC = list(iter = 1500, 
                               burn = 250, 
                               thin = 1,
                               sigma_phi_sd = 0.5)
)

pp <- predict_hebart(newX = test, new_groups = test$group,
                     hebart_posterior  = hb_model, 
                     type = "mean")
sqrt(mean(pp - test$y)^2)

hb_model <- hebart(formula = y ~ X1,
                   data = train,
                   group_variable = "group", 
                   num_trees = num_trees,
                   priors = list(
                     alpha = 0.95, # Prior control list
                     beta = 2,
                     nu = 2,
                     lambda = 0.1,
                     tau_mu = 16 * num_trees,
                     shape_sigma_phi = 0.5,
                     scale_sigma_phi = 1, 
                     sample_sigma_phi = TRUE
                   ), 
                   inits = list(tau = 1,
                                sigma_phi = 0.95),
                   MCMC = list(iter = 1500, 
                               burn = 250, 
                               thin = 1,
                               sigma_phi_sd = 0.5)
)

pp <- predict_hebart(newX = test, new_groups = test$group,
                     hebart_posterior  = hb_model, 
                     type = "mean")
hb_model
sqrt(mean(pp - test$y)^2)
cor(pp, test$y)
