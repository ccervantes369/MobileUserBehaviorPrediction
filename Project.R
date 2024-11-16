#### 1. Librerías necesarias ####
library(caTools)        # Para dividir el dataset
library(class)          # Para KNN
library(caret)          # Para validación cruzada y ajuste de modelos
library(dplyr)          # Para manipulación de datos
library(fastDummies)    # Para generar variables dummy
library(e1071)          # Para SVM
library(naivebayes)     # Para Naive Bayes
library(ggplot2)        # Para crear gráficas

#### 2. Lectura y Preprocesamiento de los datos ####
set.seed(1999)

# Cargar el dataset
datos <- read.csv("user_behavior_dataset.csv", header = TRUE)

# Eliminar columnas innecesarias
datos$User.ID <- NULL
datos$App.Usage.Time..min.day. <- NULL
datos$Screen.On.Time..hours.day. <- NULL
datos$Battery.Drain..mAh.day. <- NULL
datos$Data.Usage..MB.day. <- NULL

# Renombrar las columnas
colnames(datos) <- c("Device", "System", "Apps", "Age", "Gender", "Behavior")

# Crear variables dummy para las columnas 'Gender', 'Device' y 'System'
datos <- dummy_cols(datos, select_columns = c("Gender", "Device", "System"))

# Eliminar las columnas originales de 'Gender', 'Device' y 'System'
datos$Gender <- NULL
datos$Device <- NULL
datos$System <- NULL

# Convertir 'Behavior' a tipo factor
datos$Behavior <- as.factor(datos$Behavior)

#### 3. Normalizar solo las variables numéricas relevantes ####
# Solo normalizamos 'Age' y 'Apps', que son numéricas
datos <- datos %>%
  mutate(
    Age = scale(Age),
    Apps = scale(Apps)
  )

#### 4. División de los datos en conjuntos de entrenamiento y prueba ####
split <- sample.split(datos, SplitRatio = 0.7)
train <- subset(datos, split == TRUE)
test <- subset(datos, split == FALSE)

#### 5. Entrenamiento de modelos ####
# Definir control de validación cruzada
trainControl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

# Definir grid para KNN
grid <- expand.grid(.k = seq(1, 20, by = 1))

# Modelos: KNN, SVM y Naive Bayes
fit.knn <- train(Behavior ~ ., data = train, method = "knn", 
                 metric = "Accuracy", tuneGrid = grid, trControl = trainControl)

fit.svm <- train(Behavior ~ ., data = train, method = "svmLinear", 
                 metric = "Accuracy", trControl = trainControl)

fit.nb <- train(Behavior ~ ., data = train, method = "naive_bayes", 
                metric = "Accuracy", trControl = trainControl)

#### 6. Predicción y Evaluación de los Modelos ####
# Predicción y evaluación de KNN
prediction.knn <- predict(fit.knn, newdata = test)
cf.knn <- confusionMatrix(prediction.knn, test$Behavior)
print("Evaluación KNN:")
print(cf.knn)

# Predicción y evaluación de SVM
prediction.svm <- predict(fit.svm, newdata = test)
cf.svm <- confusionMatrix(prediction.svm, test$Behavior)
print("Evaluación SVM:")
print(cf.svm)

# Predicción y evaluación de Naive Bayes
prediction.nb <- predict(fit.nb, newdata = test)
cf.nb <- confusionMatrix(prediction.nb, test$Behavior)
print("Evaluación Naive Bayes:")
print(cf.nb)

#### 7. Pruebas de Hipótesis ####
# Extraer accuracy de validación cruzada
accuracy.knn <- fit.knn$resample$Accuracy
accuracy.svm <- fit.svm$resample$Accuracy
accuracy.nb <- fit.nb$resample$Accuracy

# Resumen de accuracies
summary(accuracy.knn)
summary(accuracy.svm)
summary(accuracy.nb)

# Test de normalidad Shapiro-Wilk
shapiro.test(accuracy.knn) # Para KNN
shapiro.test(accuracy.svm) # Para SVM
shapiro.test(accuracy.nb)  # Para Naive Bayes

# Comparaciones entre modelos
t.test(accuracy.knn, accuracy.svm, alternative = "two.sided") # KNN vs SVM
wilcox.test(accuracy.nb, accuracy.svm, alternative = "two.sided", exact = FALSE)
wilcox.test(accuracy.nb, accuracy.knn, alternative = "two.sided", exact = FALSE)

# Boxplot para comparar accuracies
x11()
boxplot(accuracy.knn, accuracy.svm, accuracy.nb, 
        names = c("KNN", "SVM", "NB"),
        main = "Comparación de precisiones", 
        ylab = "Accuracy",
        col = rgb(0.1, 0.2, 0.5, 0.5),  # Color azul con opacidad
        border = rgb(0.1, 0.2, 0.5),     # Borde azul
        whiskcol = "darkblue",           # Color de las líneas de los bigotes
        boxwex = 0.5)                   # Ancho de las cajas
