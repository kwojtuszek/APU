install.packages('neuralnet')
library(neuralnet)

prepare_data <- function(n) {
  x <- seq(1, 10, length.out = n)
  y <- 1/sqrt(x)
  data.frame(x = x, y = y)
}


set.seed(123)
training_data <- prepare_data(100)

maxs <- apply(training_data[ ,1:2], 2, max)
mins <- apply(training_data[ ,1:2], 2, min)

scaled.training_data <- as.data.frame(scale(training_data[ , 1:2], center = mins, scale = maxs - mins))

print(head(scaled.training_data, 10))

learn_net <- neuralnet(y ~ x, training_data, hidden = c(6,4))
print(learn_net)
plot(learn_net)

learn_net_function <- neuralnet(y ~ x, training_data, hidden = c(6,4), linear.output = TRUE)
plot(training_data$x, training_data$y, col = 'green', type = "l", lwd = 2, xlab = "x", ylab = "f(x)")
lines(training_data$x, predict(learn_net_function, newdata = training_data), col = "red", lty = 2, lwd = 2)
legend("topleft", legend = c("Rzeczywista funkcja", "Sieć neuronowa"), col = c("green", "red"), lty = 1:2, lwd = 2)

