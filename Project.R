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
datos <- read.csv("user_behavior_dataset.csv", header = TRUE)
df = datos
datos$User.ID = NULL  # Eliminar la columna User.ID
datos$App.Usage.Time..min.day. = NULL
datos$Screen.On.Time..hours.day. = NULL
datos$Battery.Drain..mAh.day. = NULL
datos$Data.Usage..MB.day. = NULL

# Renombrar las columnas
colnames(datos) <- c("Device", "System", "Apps", "Age", "Gender", "Behavior")

# Crear variables dummy para las columnas 'Gender', 'Device' y 'System'
datos <- dummy_cols(datos, select_columns = c("Gender", "Device", "System"))

# Eliminar las columnas originales de 'Gender', 'Device' y 'System'
datos$Gender = NULL
datos$Device = NULL
datos$System = NULL

# Convertir 'Behavior' a tipo factor
datos$Behavior = as.factor(datos$Behavior)

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

#### 5. Entrenamiento del modelo KNN ####
trainControl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid <- expand.grid(.k = seq(1, 20, by = 1))

fit.knn <- train(Behavior ~ ., data = train, method = "knn", 
                 metric = "Accuracy", tuneGrid = grid, trControl = trainControl)

#### 6. Entrenamiento del modelo SVM ####
fit.svm <- train(Behavior ~ ., data = train, method = "svmLinear", 
                 metric = "Accuracy", trControl = trainControl)

#### 7. Entrenamiento del modelo Naive Bayes ####
fit.nb <- train(Behavior ~ ., data = train, method = "naive_bayes", 
                metric = "Accuracy", trControl = trainControl)

#### 8. Predicción y Evaluación del Modelo KNN ####
prediction.knn <- predict(fit.knn, newdata = test)
cf.knn <- confusionMatrix(prediction.knn, test$Behavior)
print("Evaluación KNN:")
print(cf.knn)

#### 9. Predicción y Evaluación del Modelo SVM ####
prediction.svm <- predict(fit.svm, newdata = test)
cf.svm <- confusionMatrix(prediction.svm, test$Behavior)
print("Evaluación SVM:")
print(cf.svm)

#### 10. Predicción y Evaluación del Modelo Naive Bayes ####
prediction.nb <- predict(fit.nb, newdata = test)
cf.nb <- confusionMatrix(prediction.nb, test$Behavior)
print("Evaluación Naive Bayes:")
print(cf.nb)

# Gráficas

# Gráfico 1: Distribución del Tiempo de Uso de la Aplicación por Género
x11()  # Abrir una nueva ventana de gráfico
ggplot(df, aes(x = App.Usage.Time..min.day., fill = Gender)) +
  geom_histogram(binwidth = 30, alpha = 0.6, position = "identity") +
  labs(title = "Distribución del Tiempo de Uso de la Aplicación por Género", 
       x = "Tiempo de Uso (min/día)", y = "Frecuencia") +
  theme_minimal()

# Gráfico 2: Relación entre el Tiempo de Pantalla y el Drenaje de Batería
x11()  # Abrir una nueva ventana de gráfico
ggplot(df, aes(x = Screen.On.Time..hours.day., y = Battery.Drain..mAh.day., color = Gender)) +
  geom_point(alpha = 0.7) +
  labs(title = "Relación entre el Tiempo de Pantalla y el Drenaje de Batería", 
       x = "Tiempo de Pantalla (hrs/día)", y = "Drenaje de Batería (mAh/día)") +
  theme_minimal() +
  scale_color_manual(values = c("blue", "pink"))

# Gráfico 3: Relación entre el Tiempo de Uso de la App y el Drenaje de Batería
x11()  # Abrir una nueva ventana de gráfico
ggplot(df, aes(x = App.Usage.Time..min.day., y = Battery.Drain..mAh.day.)) +
  geom_point(aes(color = Operating.System), alpha = 0.7) +
  labs(title = "Relación entre el Tiempo de Uso de la App y el Drenaje de Batería",
       x = "Tiempo de Uso (min/día)",
       y = "Drenaje de Batería (mAh/día)") +
  theme_minimal()

# Gráfico 4: Promedio de Uso de Datos por Género
x11()  # Abrir una nueva ventana de gráfico
ggplot(df, aes(x = Gender, y = Data.Usage..MB.day., fill = Gender)) +
  geom_bar(stat = "summary", fun = "mean", position = "dodge") +
  labs(title = "Promedio de Uso de Datos por Género",
       x = "Género",
       y = "Uso de Datos (MB/día)") +
  theme_minimal()

# Gráfico 5: Número de Aplicaciones Instaladas por Modelo de Dispositivo
x11()  # Abrir una nueva ventana de gráfico
ggplot(df, aes(x = Device.Model, y = Number.of.Apps.Installed, fill = Device.Model)) +
  geom_bar(stat = "identity") +
  labs(title = "Número de Aplicaciones Instaladas por Modelo de Dispositivo",
       x = "Modelo de Dispositivo",
       y = "Número de Apps Instaladas") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Gráfico 6: Relación entre el Tiempo de Uso y el Número de Apps Instaladas
x11()  # Abrir una nueva ventana de gráfico
ggplot(df, aes(x = Number.of.Apps.Installed, y = App.Usage.Time..min.day.)) +
  geom_point(aes(color = Device.Model), alpha = 0.7) +
  labs(title = "Relación entre el Tiempo de Uso y el Número de Apps Instaladas",
       x = "Número de Apps Instaladas",
       y = "Tiempo de Uso (min/día)") +
  theme_minimal()

# Crear una nueva variable de rangos de edad
df$Age.Range <- cut(df$Age, 
                    breaks = c(17, 24, 34, 44, 54, Inf), 
                    labels = c("18-24", "25-34", "35-44", "45-54", "55+"),
                    right = TRUE)

# Gráfico 7: Densidad del Tiempo de Pantalla según Rango de Edad
x11()  # Abrir una nueva ventana de gráfico
ggplot(df, aes(x = Screen.On.Time..hours.day., fill = Age.Range, color = Age.Range)) +
  geom_density(alpha = 0.5, size = 1) +  # Curvas de densidad suavizadas con transparencia
  scale_fill_brewer(palette = "Set1") +  # Colores vivos para el relleno
  scale_color_brewer(palette = "Set1") +  # Colores vivos para las curvas
  labs(title = "Densidad del Tiempo de Pantalla Encendida del Celular",
       x = "Tiempo de Uso (min/día)",
       y = "Densidad") +
  theme_minimal() +  # Tema limpio
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold", color = "#2c3e50"),
    axis.title = element_text(size = 14, face = "bold", color = "#34495e"),
    axis.text = element_text(size = 12, color = "#34495e"),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  )

# Gráfico 8: Comparación de Drenaje de Batería por Modelo de Dispositivo
x11()  # Abrir una nueva ventana de gráfico
ggplot(df, aes(x = Device.Model, y = Battery.Drain..mAh.day., fill = Device.Model)) +
  geom_boxplot() +
  labs(title = "Drenaje de Batería por Modelo de Dispositivo",
       x = "Modelo de Dispositivo",
       y = "Drenaje de Batería (mAh/día)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
