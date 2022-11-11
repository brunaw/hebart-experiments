# In case you're installing, building, or removing the package:
# remove.packages("hebartBase")
# devtools::document()
# devtools::check()
# devtools::install()

# Exemplifying:
# Package loading  ----------------------------------
library(magrittr)
library(ggplot2)
library(tidymodels)
library(tidyverse)
library(hebartBase)
library(firatheme)
library(lme4)

# Dataset split  ------------------------------------
simulate_df <- function(){
  n_obs <- 1000
  n_groups <- 4
  n_trees <- 2
  n_terminal_nodes <- c(3, 2)
  x1 <- runif(n_obs)
  x2 <- runif(n_obs)
  group <- sample(1:n_groups, size = n_obs, replace = TRUE)
  
  # Might want to fiddle around with these values a bit
  tau_mu <- 3
  tau_phi <- 3 
  tau <- 1
  
  # Simulate terminal node parameters
  mu <- vector('list', length = n_trees)
  for (i in 1:length(mu)) {
    mu[[i]] <- rnorm(n_terminal_nodes[i], 0, 1/sqrt(tau_mu))  
  }
  
  # Then for each terminal node there are n_groups phi parameters
  phi <- vector('list', length = n_trees)
  for (i in 1:length(phi)) {
    phi[[i]] <- vector('list', length = n_terminal_nodes[i])
    for(j in 1:length(phi[[i]])) {
      phi[[i]][[j]] <- rnorm(n_groups, mu[[i]][j], sqrt(n_trees / tau_phi))
    }
  }
  
  # Write the functions for each of the trees
  tree_fun1 <- function(x1, x2, group) {
    ans <- if(x1 < 0 & x2 < 0) {
      phi[[1]][[1]][group]
    } else if(x1 < 0 & x2 >= 0) {
      phi[[1]][[2]][group]
    } else {
      phi[[1]][[2]][group]
    }
    return(ans)
  }
  tree_fun2 <- function(x1, x2, group) {
    ans <- if(x2 < 0.5 ) {
      phi[[2]][[1]][group]
    } else {
      phi[[2]][[2]][group]
    }
    return(ans)
  }
  
  # Finally simulate some y values from the tree functions
  y <- rep(NA, length = n_obs)
  for (i in 1:n_obs) {
    y[i] <- rnorm(1, mean = tree_fun1(x1[i], x2[i], group[i]) + 
                    tree_fun2(x1[i], x2[i], group[i]), 
                  sd = 1 / sqrt(tau))
  }
  df <- tibble(y, X1 = x1, X2 = x2, group)
  df
}

set.seed(2022)
df <- simulate_df()
df$y <- c(scale(df$y))

# Model parameters -----------------------------------
group_variable <-  "group"
formula        <- y ~ X1 + X2

data_split <- rsample::vfold_cv(df, v = 10) |> 
  dplyr::mutate(
    train = purrr::map(splits, training),
    test  = purrr::map(splits, testing)
  )


# Modelling definitions   ----------------------------------------
fit_hebart <- function(train){
  num_trees   <- 10
  hb_model <- hebart(formula = y ~ X1 + X2,
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
                     MCMC = list(iter = 500, 
                                 burn = 250, 
                                 thin = 1,
                                 sigma_phi_sd = 0.5)
  )
  
}

fit_bart <- function(train, test){
  bart_0 <-  dbarts::bart2(y ~ X1 + X2, 
                           data = train,
                           test = test, 
                           keepTrees = TRUE)
  
  bart_0
}

fit_lme <- function(train){
  lm3_m0_normal  <- lmer(y ~ X1 + X2  + (1 |group), data = train)
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
    #pred <- predict(model, test, re.form=NA)
    pred <- predict(model, test)
  }
  
  df <- data.frame(pred = pred, real = test$y, group = test$group,
                   x = test$X1)
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

write_rds(lme_step, "model_files/simulated_data2.rds")  
#lme_step <- readRDS("model_files/sleepstudy_all.rds")

# 2nd Case: Missing groups -----------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------
# Predictions
preds <- lme_step |> 
  dplyr::mutate(
    pred_hebart = purrr::map2(test, hebart_results, predictions, type = "hebart"),
    pred_bart   = purrr::map2(test, bart_results, predictions, type = "bart"),
    pred_lme    = purrr::map2(test, lme_results, predictions, type = "lme"),
    pred_train_hebart = purrr::map2(train, hebart_results, predictions, type = "hebart"),
    pred_train_bart   = purrr::map2(train, bart_results, predictions, type = "bart",
                                    type_bart = "train"),
    pred_train_lme    = purrr::map2(train, lme_results, predictions, type = "lme")
  )


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
  tidyr::pivot_longer(cols = c(rmse_hebart, rmse_bart, rmse_lme)) 

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
  tidyr::pivot_longer(cols = c(rmse_hebart, rmse_bart, rmse_lme)) 

rmses |>
  mutate(type = "Test") |> 
  bind_rows(rmses_train |> mutate(type = "Train")) |> 
  mutate(name = str_remove(name, "rmse\\_"), 
         name = str_to_upper(name)) |> 
  ggplot(aes(y = value, x = name)) +
  geom_boxplot(fill = "#F96209", alpha = 0.7) +
  facet_wrap(~type, scales = 'free') +
  labs(y = "Estimated RMSE", 
       x = 'Algorithms'
       #title = paste0("RMSE\nHE-BART: ", rss_hbart, ", LME: ", rss_lme)
  ) + 
  scale_y_continuous(breaks = scales::pretty_breaks(n = 5)) +
  theme_fira()
ggsave(file = "paper/boxplot_simulated_second.png",
       width = 6, height = 3)
# -------------------------------------------------------------
# Compare LME to sampled values
i <- 3
my_lme <- lme_step$lme_results[[i]]
random_effect     <- sqrt(as.data.frame(lme4::VarCorr(my_lme))$vcov[1])
pr <- parameters::model_parameters(my_lme, effects = "random",
                                   #ci_method = "boot",
                                   ci_random = TRUE,
                                   verbose = FALSE)
se <- pr$SE[1]
random_effect_var <- se^2

if(!is.na(se)){
  shape_sigma_phi  <-  (random_effect^2)/random_effect_var
  scale_sigma_phi  <-  random_effect_var/random_effect
}
shape_sigma_phi/(1/scale_sigma_phi^2)
shape_sigma_phi/(1/scale_sigma_phi)

df_gammas <- tibble(sampled = rgamma(25000, shape_sigma_phi, 1/scale_sigma_phi))
df_gammas <- df_gammas |> 
  filter(sampled < 1.6, sampled > 0.5)
mean(df_gammas$sampled)

df <- tibble(sampled = lme_step$hebart_results[[i]]$sigma_phi)

df |> 
  mutate(type = "Sampled") |> 
  bind_rows(df_gammas |> mutate(type = "Prior")) |> 
  ggplot(aes(x = sampled, group = type)) +
  geom_density(aes(fill = type), alpha = 0.7) +
  xlim(0, 1.7) +
  scale_fill_manual(values=c("#CB2314", "#0E86D4")) +
  labs(y = "Values", 
       x = 'Density', 
       fill = "Type:"
  ) + 
  scale_y_continuous(breaks = scales::pretty_breaks(n = 10)) +
  scale_x_continuous(breaks = scales::pretty_breaks(n = 10)) +
  theme_fira()
ggsave(file = "paper/density_comparison.png",
      width = 4, height = 2.3)

# -------------------------------------------------------------
# Mean predictions 

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
# -------------------------------------------------------------
# Plots -------------------------------------------------------


BottleRocket2 = c("#FAD510", "#CB2314", "#0E86D4",
                  "#1E1E1E", "#18A558")
sqrt_n <- sqrt(nrow(data_split))

summary_preds <- summary_preds |> 
  dplyr::mutate(
    upp_hebart = pred_hebart + 1.96 * sd(pred_hebart)/sqrt_n,
    low_hebart = pred_hebart - 1.96 * sd(pred_hebart)/sqrt_n,
    
    upp_lme = pred_lme + 1.96 * sd(pred_lme)/sqrt_n,
    low_lme = pred_lme - 1.96 * sd(pred_lme)/sqrt_n,
    
    upp_bart = pred_bart + 1.96 * sd(pred_bart)/sqrt_n,
    low_bart = pred_bart - 1.96 * sd(pred_bart)/sqrt_n
  )

summary_preds |>  
  ggplot(aes(x = x_bart, y = pred_hebart)) +
  geom_ribbon(aes(ymin=low_hebart, ymax=upp_hebart),
              fill = BottleRocket2[1], alpha = 0.3) + 
  
  geom_ribbon(aes(ymin=low_lme, ymax=upp_lme),
              fill = BottleRocket2[2], alpha = 0.3) + 
  
  geom_ribbon(aes(ymin=low_bart, ymax=upp_bart),
              fill = BottleRocket2[3], alpha = 0.3) + 
  
  #geom_ribbon(aes(ymin=low_ci, ymax=upp_ci), fill = "#F96209", alpha = 0.3) +
  geom_line(aes(colour = BottleRocket2[1]), size = 0.5) +
  geom_line(colour = BottleRocket2[1], size = 0.5) +
  # geom_point(aes(x = x_bart, y = real_hebart,
  #                colour =  BottleRocket2[4]), size = 0.75) + 
  geom_line(aes(x = x_bart, y = pred_lme), colour = BottleRocket2[2]) + 
  geom_line(aes(x = x_bart, y = pred_bart), colour = BottleRocket2[3]) + 
  geom_line(aes(x = x_bart, y = real_hebart), colour = BottleRocket2[4], 
            size = 0.3, linetype = "dashed") + 
  geom_point(aes(x = x_bart, y = real_hebart), colour = BottleRocket2[4], size = 1) + 
  #geom_ribbon(aes(ymin = lwr_gam, ymax = fit_gam), fill = "lightblue") + 
  # geom_errorbar(aes(ymin = low_ci, ymax = upp_ci),
  #               position = position_dodge(width = 0.2),
  #               width = 0.5, colour = '#F96209') +
  facet_wrap(~group_hebart, ncol = 5) +
  scale_x_continuous(breaks = scales::pretty_breaks(n = 10)) +
  labs(y = "Average response time (ms)", 
       x = 'Covariate: days of sleep deprivation', 
       title = "Average predictions per group for HEBART, LME and BART") + 
  scale_y_continuous(breaks = scales::pretty_breaks(n = 5)) +
  theme_linedraw(15) +
  scale_colour_manual(
    name="Source:",
    values=c(Data = BottleRocket2[4], 
             `HEBART Prediction`=BottleRocket2[1], 
             `LME Prediction`= BottleRocket2[2],
             `BART Prediction`= BottleRocket2[3]), 
    guide = guide_legend(override.aes = list(
      size = c(3, 3, 3, 3), shape = c(16, 16, 16, 16)))) + 
  theme(panel.spacing.x = unit(0.5, "lines"), 
        legend.position = "bottom")
#        width = 12, height = 6)

ggsave(file = "paper/predictions_plot_sleepstudy.png",
       width = 10, height = 8)

# -------------------------------------------------------------
# 
# ------------------------------------------------------------- 