library(neuralnet)

data <- read.csv("data.csv")
features <- data[, c("wyswietlacz", "pamiec_RAM", "pamiec_wbudowana", "aparat_foto", "cena", "liczba_opinii", "oceny_klientow")]

target <- data$cena

set.seed(123)
split_ratio <- 0.8
num_samples <- nrow(data)
training_indicies <- sample(1:num_samples, size = round(split_ratio * num_samples))

training_data <- features[training_indicies, ]
training_target <- target[training_indicies]

test_data <- features[-training_indicies, ]
test_target <- target[-training_indicies]

model <- neuralnet(cena ~ wyswietlacz + pamiec_RAM + pamiec_wbudowana + aparat_foto + liczba_opinii + oceny_klientow,
                   data = data.frame(cena = training_target, training_data), hidden = c(6, 4))

predictions <- predict(model, newdata = test_data)

results <- data.frame(Przewidywana_Cena = predictions, Rzeczywista_Cena = test_target)
print(results)
plot(model)
