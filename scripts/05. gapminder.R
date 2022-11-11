# -------------------------------------------------------------
# Experiments for the sleepstudy data
# -------------------------------------------------------------
# Package loading   --------------------------------------------
source("scripts/00. install.R")
library(magrittr)
library(ggplot2)
library(lme4)
library(tidyverse)
library(tidymodels)
library(dbarts)
library(hebartBase)
library(firatheme)

# 1. First version: random samples are removed from the test set
# Dataset split  ------------------------------------------------
set.seed(2022)
load("data/gapminder_recent_g20.RData")

df_real     <- gapminder_recent_g20 |> 
  select(year, country, lifeExp, year0, decade0) |> 
  set_names(c('X1', 'group', 'y', "X2", "X3"))

# data_split <- rsample::vfold_cv(df_real, v = 2) |> 
#   dplyr::mutate(
#     train = purrr::map(splits, training),
#     test  = purrr::map(splits, testing)
#   )

split_chunk <- function(df_real){
  years        <- unique(df_real$X1)
  to_remove    <- sample(years, 15)
  train        <- df_real |> filter(!(X1 %in% to_remove))
  test         <- df_real |> filter(X1 %in% to_remove)
  return(list(train = train, test = test))
}

data_split <- tibble(data = rep(list(df_real), 10)) |> 
  dplyr::mutate(split = map(data, split_chunk), 
                test = map(split, "test"),
                train = map(split, "train"))

# Modelling definitions   ----------------------------------------
fit_hebart <- function(train){
  num_trees   <- 5
  hb_model <- hebart(formula = y ~ X1 + X2 + X3,
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
                                  sigma_phi = 1),
                     MCMC = list(iter = 1500, 
                                 burn = 250, 
                                 thin = 1,
                                 sigma_phi_sd = 2)
  )
  
}

fit_bart <- function(train, test){
  bart_0 <-  dbarts::bart2(y ~ X1 + X2 + X3, 
                           data = train,
                           test = test, 
                           keepTrees = TRUE)
  
  bart_0
}


fit_bart_group <- function(train, test){
  bart_0 <-  dbarts::bart2(y ~ X1 + X2 + X3 + group, 
                           data = train,
                           test = test, 
                           keepTrees = TRUE)
  
  bart_0
}


fit_lme <- function(train){
  lm3_m0_normal  <- lmer(y ~ X1 + X2 + X3 + (1 |group), data = train)
  lm3_m0_normal
}

predictions <- function(test, model, type, type_bart = "test"){
  if(type == "hebart"){
    pred <-  predict_hebart(newX = test, new_groups = test$group,
                            hebart_posterior  = model, type = "mean")
    
  } else if(type == "bart"){
    if(type_bart == "test"){
      pred <- model$yhat.test.mean
    } else{
      pred <- model$yhat.train.mean  
    }
    
  } else if(type == "bart_group"){
    if(type_bart == "test"){
      pred <- model$yhat.test.mean
    } else{
      pred <- model$yhat.train.mean  
    }
    
  } else if(type == "lme"){
    #pred <- predict(model, test, re.form=NA)
    pred <- predict(model, test)
  }
  
  df <- data.frame(pred = pred, real = test$y, group = test$group,
                   x = test$X1)
  names(df) <- paste0(names(df), "_", type)
  return(df)
}


rmse <- function(x, y){
  sqrt(mean((x - y)^2))
}
# HEBART Modelling -------------------------------------------
hebart_step <- data_split |> 
  dplyr::mutate(
    hebart_results = purrr::map(train, fit_hebart)
  )

# hb_model <- hebart_step$hebart_results[[1]]
# test <- hebart_step$test[[1]]
# train <- hebart_step$train[[1]]
# pp <- predict_hebart(test, test$group, hb_model, type = "mean")
# sqrt(mean(pp - test$y)^2) # 0.5339674
# cor(pp, scale(test$y))  # 0.9881025
# rmse(pp, test$y)

# BART Modelling ---------------------------------------------
bart_step <- hebart_step |> 
  dplyr::mutate(
    bart_results = purrr::map2(train, test, fit_bart)
  )

bart_step_group <- bart_step |> 
  dplyr::mutate(
    bart_group_results = purrr::map2(train, test, fit_bart_group)
  )

# LME Modelling ---------------------------------------------
lme_step <- bart_step_group |> 
  dplyr::mutate(
    lme_results = purrr::map(train, fit_lme)
  )

# write_rds(lme_step, "model_files/gapminder_all.rds")  
lme_step <- readRDS("model_files/gapminder_all.rds")
# lme_step <- sleepstudy_all
# -------------------------------------------------------------
# Predictions
preds <- lme_step |> 
  dplyr::mutate(
    pred_hebart       = purrr::map2(test, hebart_results, predictions, type = "hebart"),
    pred_bart         = purrr::map2(test, bart_results, predictions, type = "bart"),
    pred_bart_group   = purrr::map2(test, bart_group_results, predictions, 
                                    type = "bart_group"),
    pred_lme          = purrr::map2(test, lme_results, predictions, type = "lme"),
    pred_train_hebart = purrr::map2(train, hebart_results, predictions, type = "hebart"),
    pred_train_bart   = purrr::map2(train, bart_results, predictions, type = "bart",
                                    type_bart = "train"),
    pred_train_bart_group   = purrr::map2(train, bart_group_results, predictions, 
                                          type = "bart_group", type_bart = "train"),
    pred_train_lme    = purrr::map2(train, lme_results, predictions, type = "lme")
  )
preds
# -------------------------------------------------------------
summary_preds <- preds |>
  mutate(id = 1:n()) |> 
  dplyr::select(id, starts_with("pred_train")) |> 
  tidyr::unnest(
    c(pred_train_hebart, pred_train_bart, pred_train_bart_group,  pred_train_lme), 
    names_repair = "unique")


rmses_train <- summary_preds |>
  dplyr::group_by(id) |>
  dplyr::mutate(
    rmse_hebart = rmse(pred_hebart, real_hebart),
    rmse_bart = rmse(pred_bart, real_bart),
    rmse_bart_group = rmse(pred_bart_group, real_bart_group),
    rmse_lme = rmse(pred_lme, real_lme)
  ) |> 
  tidyr::pivot_longer(cols = c(rmse_hebart, rmse_bart, 
                               rmse_lme, rmse_bart_group)) |> 
  dplyr::group_by(name) |> 
  dplyr::summarise(
    mean = mean(value), 
    upp = mean + 1.96 * sd(value),
    low = mean - 1.96 * sd(value),
  ) 


summary_preds <- preds |> 
  mutate(id = 1:n()) |> 
  dplyr::select(id, pred_hebart, pred_bart, pred_bart_group, pred_lme) |> 
  tidyr::unnest(c(pred_hebart, pred_bart, pred_bart_group, pred_lme), 
                names_repair = "unique")

rmses <- summary_preds |> 
  dplyr::group_by(id) |> 
  dplyr::summarise(
    rmse_hebart = rmse(pred_hebart, real_hebart),
    rmse_bart = rmse(pred_bart, real_bart),
    rmse_bart_group = rmse(pred_bart_group, real_bart_group),
    rmse_lme = rmse(pred_lme, real_lme)
  ) |> 
  tidyr::pivot_longer(cols = c(rmse_hebart, rmse_bart, rmse_bart_group, rmse_lme)) |> 
  dplyr::group_by(name) |> 
  dplyr::summarise(
    mean = mean(value), 
    upp = mean + 1.96 * sd(value),
    low = mean - 1.96 * sd(value),
  ) 


rmses |>
  mutate(type = "test") |> 
  bind_rows(rmses_train |> mutate(type = "train")) |> 
  select(name, mean, low, upp, type) |> 
  arrange(type) 
# -------------------------------------------------------------
# -------------------------------------------------------------
# Plots -------------------------------------------------------

BottleRocket2 = c("#FAD510", "#CB2314", "#0E86D4",
                  "#1E1E1E", "#18A558")
sqrt_n <- sqrt(nrow(data_split))

summary_preds <- summary_preds |> 
  dplyr::mutate(
    upp_hebart = pred_hebart + 1 * sd(pred_hebart)/sqrt_n,
    low_hebart = pred_hebart - 1 * sd(pred_hebart)/sqrt_n,
    
    upp_lme = pred_lme + 1 * sd(pred_lme)/sqrt_n,
    low_lme = pred_lme - 1 * sd(pred_lme)/sqrt_n,
    
    upp_bart = pred_bart + 1 * sd(pred_bart)/sqrt_n,
    low_bart = pred_bart - 1 * sd(pred_bart)/sqrt_n,
    
    upp_bart_group = pred_bart_group + 1 * sd(pred_bart_group)/sqrt_n,
    low_bart_group = pred_bart_group - 1 * sd(pred_bart_group)/sqrt_n
  )

selected_countries <- c(
  "South Africa", "Russia", "China", 
  "Turkey", "Indonesia", "Brazil"
)

summary_preds |>  
  filter(group_hebart %in% selected_countries) |> 
  ggplot(aes(x = x_bart, y = pred_hebart)) +
  
  geom_ribbon(aes(ymin=low_bart_group, ymax=upp_bart_group),
              fill = BottleRocket2[4], alpha = 0.3) + 
  
  geom_ribbon(aes(ymin=low_hebart, ymax=upp_hebart),
              fill = BottleRocket2[1], alpha = 0.3) + 
  
  geom_ribbon(aes(ymin=low_lme, ymax=upp_lme),
              fill = BottleRocket2[2], alpha = 0.3) + 
  
  geom_ribbon(aes(ymin=low_bart, ymax=upp_bart),
              fill = BottleRocket2[3], alpha = 0.3) + 
  
  #geom_ribbon(aes(ymin=low_bart_group, ymax=upp_bart_group),
  #            fill = BottleRocket2[4], alpha = 0.3) + 
  
  #geom_ribbon(aes(ymin=low_ci, ymax=upp_ci), fill = "#F96209", alpha = 0.3) +
  geom_line(aes(colour = BottleRocket2[1]), size = 0.5) +
  geom_line(colour = BottleRocket2[1], size = 0.5) +
   geom_point(aes(x = x_bart, y = real_hebart,
                  colour =  BottleRocket2[4]), size = 0.75) + 
  geom_line(aes(x = x_bart, y = pred_lme), colour = BottleRocket2[2]) + 
  geom_line(aes(x = x_bart, y = pred_bart), colour = BottleRocket2[3]) + 
  geom_line(aes(x = x_bart, y = pred_bart_group), colour = BottleRocket2[5]) + 
  geom_line(aes(x = x_bart, y = real_hebart), colour = BottleRocket2[4], 
            size = 0.3, linetype = "dashed") + 
  geom_point(aes(x = x_bart, y = real_hebart), 
             colour = BottleRocket2[4], size = 1) + 
  #geom_ribbon(aes(ymin = lwr_gam, ymax = fit_gam), fill = "lightblue") + 
  # geom_errorbar(aes(ymin = low_ci, ymax = upp_ci),
  #               position = position_dodge(width = 0.2),
  #               width = 0.5, colour = '#F96209') +
  facet_wrap(~group_hebart, ncol = 2) +
  scale_x_continuous(breaks = scales::pretty_breaks(n = 3)) +
  labs(y = "Life expectancy (years)", 
       x = 'Covariate: year', 
       title = "Average predictions per group for HEBART, LME and BART") + 
  scale_y_continuous(breaks = scales::pretty_breaks(n = 5)) +
  theme_linedraw(15) +
  scale_colour_manual(
    name="Source:",
    values=c(Data = BottleRocket2[4], 
             `HEBART`=BottleRocket2[1], 
             `LME`= BottleRocket2[2],
             `BART`= BottleRocket2[3], 
             `BART+Group`= BottleRocket2[5]), 
    guide = guide_legend(override.aes = list(
      size = c(3, 3, 3, 3, 3), shape = c(16, 16, 16, 16, 16)))) + 
  theme(panel.spacing.x = unit(0.5, "lines"), 
        legend.position = "bottom")
#        width = 12, height = 6)

ggsave(file = "paper/predictions_plot_gapminder.png",
       width = 8, height = 8)

# -------------------------------------------------------------
# 
# -------------------------------------------------------------