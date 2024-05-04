tau <- 1
e <- 1
alpha <- 1
v <- 0.315
v_tilde <- 0.394



# CR
a_c <- 0.5
Q11_1 <- 0.1 * a_c * alpha * v_tilde * 0.974394 / (e + alpha * v_tilde * 0.974394) # s solved by wolframalpha
Q11_2 <- 1 * a_c * alpha * v_tilde * 0.783217 / (e + alpha * v_tilde * 0.783217)
Q11_3 <- 10 * a_c * alpha * v_tilde * 0.234243 / (e + alpha * v_tilde * 0.234243)
Q01_1 <- 0.1 * (1 - a_c) * alpha * v * 0.974394 / (e + alpha * v * 0.974394)
Q01_2 <- 1 * (1 - a_c) * alpha * v * 0.783217 / (e + alpha * v * 0.783217)
Q01_3 <- 10 * (1 - a_c) * alpha * v * 0.234243 / (e + alpha * v * 0.234243)
GTE_CR_1 <- Q11_1 / a_c - Q01_1 / (1 - a_c)
GTE_CR_2 <- Q11_2 / a_c - Q01_2 / (1 - a_c)
GTE_CR_3 <- Q11_3 / a_c - Q01_3 / (1 - a_c)



# LR
a_l <- 0.5
Q11_1 <- 0.1 * 1 * alpha * v_tilde * 0.485785 / (e + alpha * v_tilde * 0.485785)
Q11_2 <- 1 * 1 * alpha * v_tilde * 0.382160 / (e + alpha * v_tilde * 0.382160)
Q11_3 <- 10 * 1 * alpha * v_tilde * 0.107849 / (e + alpha * v_tilde * 0.107849)
Q10_1 <- 0.1 * 1 * alpha * v * 0.488557 / (e + alpha * v * 0.488557)
Q10_2 <- 1 * 1 * alpha * v * 0.401055 / (e + alpha * v * 0.401055)
Q10_3 <- 10 * 1 * alpha * v * 0.127901 / (e + alpha * v * 0.127901)
GTE_LR_1 <- Q11_1 / a_l - Q10_1 / (1 - a_l)
GTE_LR_2 <- Q11_2 / a_l - Q10_2 / (1 - a_l)
GTE_LR_3 <- Q11_3 / a_l - Q10_3 / (1 - a_l)




# GTE
GTE <- c()
GTE[1] <- 0.1 * alpha * v_tilde * 0.972317 / (e + alpha * v_tilde * 0.972317) - 0.1 * alpha * v * 0.976476 / (e + alpha * v * 0.976476)
GTE[2] <- 1 * alpha * v_tilde * 0.767868 / (e + alpha * v_tilde * 0.767868) - 1 * alpha * v * 0.798939 / (e + alpha * v * 0.798939)
GTE[3] <- 10 * alpha * v_tilde * 0.216057 / (e + alpha * v_tilde * 0.216057) - 10 * alpha * v * 0.255398 / (e + alpha * v * 0.255398)




# a_c and a_l related with lambda\tau
a_ck <- c()
a_lk <- c()
a_ck[1] <- (1 - exp(-0.1)) + a_c * exp(-0.1)
a_lk[1] <- a_l * (1 - exp(-0.1)) + exp(-0.1)
a_ck[2] <- (1 - exp(-1)) + a_c * exp(-1)
a_lk[2] <- a_l * (1 - exp(-1)) + exp(-1)
a_ck[3] <- (1 - exp(-10)) + a_c * exp(-10)
a_lk[3] <- a_l * (1 - exp(-10)) + exp(-10)


# TSRN
GTE_TSRN_1 <- 0.1 * a_ck[1] * 0.264708 / (a_ck[1] * a_lk[1]) - (0.1 * (1 - a_ck[1]) * 0.223627 + 0.1 * a_ck[1] * 0.010613 + 0.1 * (1 - a_ck[1]) * 0.011206) / (1 - a_ck[1] * a_lk[1])
GTE_TSRN_2 <- 1 * a_ck[2] * 0.161351 / (a_ck[2] * a_lk[2]) - (1 * (1 - a_ck[2]) * 0.133401 + 1 * a_ck[2] * 0.062046 + 1 * (1 - a_ck[2]) * 0.064114) / (1 - a_ck[2] * a_lk[2])
GTE_TSRN_3 <- 10 * a_ck[3] * 0.039217 / (a_ck[3] * a_lk[3]) - (10 * (1 - a_ck[3]) * 0.031626 + 10 * a_ck[3] * 0.037208 + 10 * (1 - a_ck[3]) * 0.037502) / (1 - a_ck[3] * a_lk[3])


#TSRI-1
GTE_TSRI_1_1 <- exp(-0.1) * (0.1 * a_ck[1] * 0.264708 / (a_ck[1] * a_lk[1]) - 0.1 * (1 - a_ck[1]) * 0.223627 / ((1 - a_ck[1]) * a_lk[1]) - 1 * (1 - exp(-0.1)) * (
    0.1 * (1 - a_ck[1]) * 0.011206 / ((1 - a_ck[1]) * (1 - a_lk[1])) - 0.1 * (1 - a_ck[1]) * 0.223627 /
      ((1 - a_ck[1]) * a_lk[1]))) + 
  (1 - exp(-0.1)) * (0.1 * a_ck[1] * 0.264708 / (a_ck[1] * a_lk[1]) - 0.1 * a_ck[1] * 0.010613 / (a_ck[1] * (1 - a_lk[1])) - 1 * exp(-0.1) * (
    0.1 * (1 - a_ck[1]) * 0.011206 / (1 - a_ck[1]) / (1 - a_lk[1]) - 0.1 * a_ck[1] * 0.010613 / (a_ck[1] * (1 - a_lk[1]))))

GTE_TSRI_1_2 <- exp(-1) * (1 * a_ck[2] * 0.161351 / (a_ck[2] * a_lk[2]) - 1 * (1 - a_ck[2]) * 0.133401 / ((1 - a_ck[2]) * a_lk[2]) - 1 * (1 - exp(-1)) * (1 * (1 - a_ck[2]) * 0.064114 / ((1 - a_ck[2]) * (1 - a_lk[2])) - 1 * (1 - a_ck[2]) * 0.133401 /((1 - a_ck[2]) * a_lk[2]))) + 
  (1 - exp(-1)) * (1 * a_ck[2] * 0.161351 / (a_ck[2] * a_lk[2]) - 1 * a_ck[2] * 0.062046 / (a_ck[2] * (1 - a_lk[2])) - 1 * exp(-1) * (
    1 * (1 - a_ck[2]) * 0.064114 / (1 - a_ck[2]) / (1 - a_lk[2]) - 1 * a_ck[2] * 0.062046 / (a_ck[2] * (1 - a_lk[2]))))

GTE_TSRI_1_3 <- exp(-10) * (10 * a_ck[3] * 0.039217 / (a_ck[3] * a_lk[3]) - 10 * (1 - a_ck[3]) * 0.031626 / ((1 - a_ck[3]) * a_lk[3]) - 1 * (1 - exp(-10)) * (
    10 * (1 - a_ck[3]) * 0.037502 / ((1 - a_ck[3]) * (1 - a_lk[3])) - 10 * (1 - a_ck[3]) * 0.031626 /
      ((1 - a_ck[3]) * a_lk[3]))) + 
  (1 - exp(-10)) * (10 * a_ck[3] * 0.039217 / (a_ck[3] * a_lk[3]) - 10 * a_ck[3] * 0.037208 / (a_ck[3] * (1 - a_lk[3])) - 1 * exp(-10) * (
    10 * (1 - a_ck[3]) * 0.037502 / (1 - a_ck[3]) / (1 - a_lk[3]) - 10 * a_ck[3] * 0.037208 / (a_ck[3] * (1 - a_lk[3]))))


#TSRI-2
GTE_TSRI_2_1 <- exp(-0.1) * (0.1 * a_ck[1] * 0.264708 / (a_ck[1] * a_lk[1]) - 0.1 * (1 - a_ck[1]) * 0.223627 / ((1 - a_ck[1]) * a_lk[1]) - 2 * (1 - exp(-0.1)) * (
    0.1 * (1 - a_ck[1]) * 0.011206 / ((1 - a_ck[1]) * (1 - a_lk[1])) - 0.1 * (1 - a_ck[1]) * 0.223627 /
      ((1 - a_ck[1]) * a_lk[1]))) + (1 - exp(-0.1)) * (0.1 * a_ck[1] * 0.264708 / (a_ck[1] * a_lk[1]) - 0.1 * a_ck[1] * 0.010613 / (a_ck[1] * (1 - a_lk[1])) - 2 * exp(-0.1) * (
    0.1 * (1 - a_ck[1]) * 0.011206 / (1 - a_ck[1]) / (1 - a_lk[1]) - 0.1 * a_ck[1] * 0.010613 / (a_ck[1] * (1 - a_lk[1]))))

GTE_TSRI_2_2 <- exp(-1) * (1 * a_ck[2] * 0.161351 / (a_ck[2] * a_lk[2]) - 1 * (1 - a_ck[2]) * 0.133401 / ((1 - a_ck[2]) * a_lk[2]) - 2 * (1 - exp(-1)) * (1 * (1 - a_ck[2]) * 0.064114 / ((1 - a_ck[2]) * (1 - a_lk[2])) - 1 * (1 - a_ck[2]) * 0.133401 /((1 - a_ck[2]) * a_lk[2]))) + (1 - exp(-1)) * (1 * a_ck[2] * 0.161351 / (a_ck[2] * a_lk[2]) - 1 * a_ck[2] * 0.062046 / (a_ck[2] * (1 - a_lk[2])) - 2 * exp(-1) * (
    1 * (1 - a_ck[2]) * 0.064114 / (1 - a_ck[2]) / (1 - a_lk[2]) - 1 * a_ck[2] * 0.062046 / (a_ck[2] * (1 - a_lk[2]))))

GTE_TSRI_2_3 <- exp(-10) * (10 * a_ck[3] * 0.039217 / (a_ck[3] * a_lk[3]) - 10 * (1 - a_ck[3]) * 0.031626 / ((1 - a_ck[3]) * a_lk[3]) - 2 * (1 - exp(-10)) * (
    10 * (1 - a_ck[3]) * 0.037502 / ((1 - a_ck[3]) * (1 - a_lk[3])) - 10 * (1 - a_ck[3]) * 0.031626 /
      ((1 - a_ck[3]) * a_lk[3]))) + (1 - exp(-10)) * (10 * a_ck[3] * 0.039217 / (a_ck[3] * a_lk[3]) - 10 * a_ck[3] * 0.037208 / (a_ck[3] * (1 - a_lk[3])) - 2 * exp(-10) * (
    10 * (1 - a_ck[3]) * 0.037502 / (1 - a_ck[3]) / (1 - a_lk[3]) - 10 * a_ck[3] * 0.037208 / (a_ck[3] * (1 - a_lk[3]))))

# plot
df <- data.frame(
    gte = GTE,
    cr = c(GTE_CR_1, GTE_CR_2, GTE_CR_3),
    lr = c(GTE_LR_1, GTE_LR_2, GTE_LR_3),
    tsrn = c(GTE_TSRN_1, GTE_TSRN_2, GTE_TSRN_3),
    tsri1 = c(GTE_TSRI_1_1, GTE_TSRI_1_2, GTE_TSRI_1_3),
    tsri2 = c(GTE_TSRI_2_1, GTE_TSRI_2_2, GTE_TSRI_2_3)
  )
df[, 2:6] <- apply(df[, 2:6], 2, function(x)  abs(x - df[, 1]) / df[, 1])
df$lambda <- c(0.1, 1, 10)

df_long <- reshape2::melt(df[, 2:7], id.vars = 'lambda')
format_y_labels <- function(y) {
  y / 1000
}
p <- ggplot(df_long, aes(x = factor(lambda), y = value * 1000, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_y_log10(labels = format_y_labels) +
  theme_minimal() +
  labs(title = "Mean Field Bias",
       x = "Relative Demand λ/τ",
       y = "Bias/GTE") +
  scale_fill_manual(
    values = c(
      "cr" = "lightblue",
      "lr" = "orange",
      "tsrn" = "lightgreen",
      "tsri1" = "yellow",
      "tsri2" = "purple"
    ),
    labels = c(
      "cr" = "Customer-Side",
      "lr" = "Listing-Side",
      "tsrn" = "TSR-Naive",
      "tsri1" = "TSRI-1",
      "tsri2" = "TSRI-2"
    )
  )


p <- p + theme(
  plot.title = element_text(hjust = 0.5),
  legend.title = element_blank()
)

print(p)
