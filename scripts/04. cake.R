# -------------------------------------------------------------
# Experiments for the cake data
# -------------------------------------------------------------
# Package loading   --------------------------------------------
#source("scripts/00. install.R")
library(magrittr)
library(ggplot2)
library(lme4)
library(tidyverse)
library(tidymodels)
library(dbarts)
library(hebartBase)
library(firatheme)

# Dataset split  ------------------------------------------------
set.seed(2022)
df_real <- lme4::cake 
# Set the group an interaction between replicate and recipe
df_real$group <- paste0(df_real$replicate, "_", df_real$recipe)

# Model parameters ------------
formula <- angle ~  temperature
data_split <- rsample::vfold_cv(df_real, v = 20) |> 
  dplyr::mutate(
    train = purrr::map(splits, training),
    test  = purrr::map(splits, testing)
  )

# Modelling definitions   ----------------------------------------
fit_hebart <- function(train){
  num_trees   <- 10
  hb_model <- hebart(formula = angle ~  temperature,
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
                     MCMC = list(iter = 2000, 
                                 burn = 500, 
                                 thin = 1,
                                 sigma_phi_sd = 0.5)
  )
  
}

fit_bart <- function(train, test){
  bart_0 <-  dbarts::bart2(angle ~  temperature + recipe + replicate, 
                           data = train,
                           test = test, 
                           keepTrees = TRUE)
  
  bart_0
}

fit_lme <- function(train){
  lm3_m0_normal  <- lmer( angle ~  temperature + (1 | recipe : replicate), data = train)
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
  } else if(type == "lme"){
    pred <- predict(model, test, re.form=NA)
  }
  
  df <- data.frame(pred = pred, real = test$angle, group = test$group)
  names(df) <- paste0(names(df), "_", type)
  return(df)
}


rmse <- function(x, y){
  sqrt(mean((x - y )^2)) 
}
# HEBART Modelling -------------------------------------------
hebart_step <- data_split |>
  dplyr::mutate(
    hebart_results = purrr::map(train, fit_hebart)
  )

# BART Modelling ---------------------------------------------
bart_step <- hebart_step |>
  dplyr::mutate(
    bart_results = purrr::map2(train, test, fit_bart)
  )

# LME Modelling ---------------------------------------------
lme_step <- bart_step |> 
  dplyr::mutate(
    lme_results = purrr::map(train, fit_lme)
  )

# -------------------------------------------------------------
# Predictions
preds <- lme_step |> 
  dplyr::mutate(
    pred_hebart = purrr::map2(test, hebart_results, predictions, type = "hebart"),
    pred_bart   = purrr::map2(test, bart_results, predictions, type = "bart"),
    pred_lme    = purrr::map2(test, lme_results, predictions, type = "lme"),
    pred_train_hebart = purrr::map2(train, hebart_results, predictions, type = "hebart"),
    pred_train_bart   = purrr::map2(train, bart_results, predictions, 
                                    type = "bart", type_bart = "train"),
    pred_train_lme    = purrr::map2(train, lme_results, predictions, type = "lme")
  )

#write_rds(preds, "model_files/cake_all.rds")  

# A tibble: 6 × 5
# name         mean   upp   low type 
# <chr>       <dbl> <dbl> <dbl> <chr>
# 1 rmse_bart    4.86  7.45  2.27 test 
# 2 rmse_bart    4.41  4.55  4.26 train
# 3 rmse_hebart  5.27  7.92  2.62 test 
# 4 rmse_hebart  3.13  4.52  1.74 train
# 5 rmse_lme     7.54 11.4   3.68 test 
# 6 rmse_lme     7.70  7.92  7.49 train
# -------------------------------------------------------------
summary_preds <- preds |> 
  dplyr::select(id, starts_with("pred_train")) |> 
  tidyr::unnest(c(pred_train_hebart, pred_train_bart, pred_train_lme), 
                names_repair = "unique")

rmses_train <- summary_preds |> 
  dplyr::group_by(id) |> 
  dplyr::summarise(
    rmse_hebart = rmse(pred_hebart, real_hebart),
    rmse_bart = rmse(pred_bart, real_bart),
    rmse_lme = rmse(pred_lme, real_lme)
  ) |> 
  pivot_longer(cols = c(rmse_hebart, rmse_bart, rmse_lme)) |> 
  dplyr::group_by(name) |> 
  dplyr::summarise(
    mean = mean(value), 
    upp = mean + 1.96 * sd(value),
    low = mean - 1.96 * sd(value),
    low = ifelse(low < 0, 0, low)
  ) 


summary_preds <- preds |> 
  dplyr::select(id, pred_hebart, pred_bart, pred_lme) |> 
  tidyr::unnest(c(pred_hebart, pred_bart, pred_lme), 
                names_repair = "unique")

rmses <- summary_preds |> 
  dplyr::group_by(id) |> 
  dplyr::summarise(
    rmse_hebart = rmse(pred_hebart, real_hebart),
    rmse_bart = rmse(pred_bart, real_bart),
    rmse_lme = rmse(pred_lme, real_lme)
  ) |> 
  pivot_longer(cols = c(rmse_hebart, rmse_bart, rmse_lme)) |> 
  dplyr::group_by(name) |> 
  dplyr::summarise(
    mean = mean(value), 
    upp = mean + 1.96 * sd(value),
    low = mean - 1.96 * sd(value),
    low = ifelse(low < 0, 0, low)
  ) 
  
rmses |>
  mutate(type = "test") |> 
  bind_rows(rmses_train |> mutate(type = "train")) |> 
  arrange(name) 
#-------------------------------------------------------------------------------
summary_preds |> 
  dplyr::mutate(id_obs = 1:n()) |> 
  ggplot(aes(x = real_hebart, y = pred_hebart)) +
  #geom_ribbon(aes(ymin=lwr, ymax=upr), fill = "grey", alpha = 0.3) + 
  #geom_ribbon(aes(ymin=low_ci, ymax=upp_ci), fill = "#F96209", alpha = 0.3) +
  geom_point(aes(colour = "#F96209"), colour = "#F96209", size = 0.7) +
  #geom_line(colour = "#F96209", size = 0.5) +
  # geom_point(aes(x = real_hebart, y = pred_bart, colour = "#75E6DA"),
  #            colour = "#0A7029",
  #            size = 0.7) + 
  # geom_point(aes(x = real_hebart, y = pred_lme, colour = "#FEDE00"),
  #            colour = "#FEDE00", size = 0.7) + 
  #geom_line(aes(x = X1, y = fit), colour = "grey") + 
  #geom_ribbon(aes(ymin = lwr_gam, ymax = fit_gam), fill = "lightblue") + 
  # geom_errorbar(aes(ymin = low_ci, ymax = upp_ci),
  #               position = position_dodge(width = 0.2),
  #               width = 0.5, colour = '#F96209') +
#geom_smooth(colour = "grey80",  alpha = 0.2) +
#facet_wrap(~group_hebart, ncol = 3, scales = "free") +
#scale_x_continuous(breaks = scales::pretty_breaks(n = 10)) +
labs(y = "Predicted average response time (ms)", 
     x = 'True average response time (ms)'
     #title = paste0("RMSE\nHE-BART: ", rss_hbart, ", LME: ", rss_lme)
) + 
  scale_y_continuous(breaks = scales::pretty_breaks(n = 5)) +
  theme_fira() + 
  scale_colour_manual(
    name="Source:",
    values=c(Data="#75E6DA", 
             `HE-BART Prediction`="#F96209", 
             `LME Fit`= 'grey'), 
    guide = guide_legend(override.aes = list(
      size = c(3, 3, 3), shape = c(16, 16, 16)))) + 
  theme(panel.spacing.x = unit(0.5, "lines"), 
        legend.position = "bottom")
