###################################################################
### Load Libraries and Data #######################################
###################################################################

# Load necessary libraries
library(readr)
library(dplyr)
library(lubridate)
library(ggplot2)
library(tseries)
library(pheatmap)
library(forecast)
library(tidyverse)
library(vars)
library(fastDummies)

# Load the dataset
bike_data <- read_csv("data/SeoulBikeData.csv", locale=locale(encoding="latin1"))

###################################################################
### Data Exploration ##############################################
###################################################################

# Preview the first few rows
head(bike_data)

# Check structure of the dataset
str(bike_data)
bike_data$Date <- dmy(bike_data$Date) # Convert the date column to Date type
bike_data$Timestamp <- bike_data$Date + hours(bike_data$Hour) # Create the timestamp column

# Rename columns
colnames(bike_data)[colnames(bike_data) %in% c("Rented Bike Count", "Temperature(°C)", "Humidity(%)", "Wind speed (m/s)", "Visibility (10m)", "Dew point temperature(°C)", "Solar Radiation (MJ/m2)", "Rainfall(mm)", "Snowfall (cm)", "Functioning Day")] <- c("Rented_Bike_Count", "Temperature", "Humidity", "Wind_Speed", "Visibility", "Dew_Point_Temperature", "Solar_Radiation", "Rainfall", "Snowfall", "Functioning_Day")

# Summary statistics of the dataset
summary(bike_data)

# Check for missing values
colSums(is.na(bike_data)) # no missing values found

# Get distinct values in Seasons
unique(bike_data$Seasons)

# Get distinct values in Holiday
unique(bike_data$Holiday)

# Get distinct values in Functioning_Day
unique(bike_data$Functioning_Day)

# Export data to a CSV file
cleaned_data <- bike_data
write.csv(bike_data, "data/cleaned_data.csv", row.names = FALSE)

###################################################################
### Data Visualizations ###########################################
###################################################################

# Plot distribution of Rented Bike Count
ggplot(cleaned_data, aes(x = `Rented_Bike_Count`)) +
  geom_histogram(binwidth = 100, fill = "blue", color = "black", alpha = 0.7) +
  geom_density(aes(y = ..count..), color = "red", size = 1) +
  labs(title = "Distribution of Rented Bike Count",
       x = "Rented Bike Count",
       y = "Frequency")

# Plot average rentals by hour
cleaned_data %>%
  group_by(Hour) %>%
  summarise(Average_Rentals = mean(`Rented_Bike_Count`)) %>%
  ggplot(aes(x = Hour, y = Average_Rentals)) +
  geom_col(fill = "blue") +
  labs(
    title = "Average Rentals by Hour",
    x = "Hour",
    y = "Average Rented Bikes"
  )

# Plot average rentals by season
cleaned_data %>%
  group_by(Seasons) %>%
  summarise(Average_Rentals = mean(`Rented_Bike_Count`)) %>%
  ggplot(aes(x = Seasons, y = Average_Rentals, fill = Seasons)) +
  geom_bar(stat = "identity") +
  labs(title = "Average Rentals by Season",
       x = "Season",
       y = "Average Rented Bikes")

# Plot trend chart for total rentals
cleaned_data %>%
  mutate(Month = floor_date(Date, unit = "month")) %>%  # Extract the month
  group_by(Month) %>%
  summarise(Total_Rentals = sum(`Rented_Bike_Count`)) %>%  # Aggregate rentals by month
  ggplot(aes(x = Month, y = Total_Rentals)) +
  geom_line(color = "blue", size = 1) +  # Line plot for monthly trend
  labs(
    title = "Monthly Trend of Rented Bike Count",
    x = "Month",
    y = "Total Rented Bikes"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

# Plot bike rentals by Temperature
ggplot(cleaned_data, aes(x = `Temperature`, y = `Rented_Bike_Count`)) +
  geom_point(alpha = 0.5, color = "blue") +
  geom_smooth(method = "lm", color = "red") +
  labs(title = "Temperature vs Rented Bike Count",
       x = "Temperature (°C)",
       y = "Rented Bike Count")

# Plot bike rentals by Temperature
ggplot(cleaned_data, aes(x = `Humidity`, y = `Rented_Bike_Count`)) +
  geom_point(alpha = 0.5, color = "blue") +
  geom_smooth(method = "lm", color = "red") +
  labs(title = "Humidity vs Rented Bike Count",
       x = "Humidity (%)",
       y = "Rented Bike Count")

# Calculate the correlation matrix
cor_matrix <- cor(cleaned_data[sapply(cleaned_data, is.numeric)])

# Plot the heatmap
pheatmap(cor_matrix, 
         display_numbers = TRUE,  # Show correlation values on the heatmap
         clustering_distance_rows = "euclidean", 
         clustering_distance_cols = "euclidean", 
         clustering_method = "complete", 
         color = colorRampPalette(c("blue", "white", "red"))(50))

###################################################################
### Data Preprocessing ############################################
###################################################################

processed_data <- cleaned_data

# Replace Holiday values and convert to numeric
processed_data$Holiday <- ifelse(processed_data$Holiday == "Holiday", 1, 0)
processed_data$Holiday <- as.numeric(processed_data$Holiday)

# Replace Holiday values and convert to numeric
processed_data$Functioning_Day <- ifelse(processed_data$Functioning_Day == "Yes", 1, 0)
processed_data$Functioning_Day <- as.numeric(processed_data$Functioning_Day)

# Remove any duplicate rows
processed_data <- processed_data %>% distinct()

# Check duplicate rows
sum(duplicated(processed_data))

# Make timestamp as rownames
processed_data <- processed_data %>%
  column_to_rownames("Timestamp")

# Drop unnecessary columns
processed_data <- processed_data %>% dplyr::select(-Seasons, -Hour, -Dew_Point_Temperature)

# Perform Augmented Dickey-Fuller Test
adf.test(processed_data$Rented_Bike_Count, alternative = "stationary")

###################################################################
### Data Splitting and Partition ##################################
###################################################################

# Define a split date
split_date <- as.Date("2018-11-23")

# Split the data into training and testing sets
train_data <- processed_data[processed_data$Date <= split_date, ]
test_data <- processed_data[processed_data$Date > split_date, ]

# Check the dimensions of the split
cat("Training data:", nrow(train_data), "rows\n")
cat("Testing data:", nrow(test_data), "rows\n")

# Drop date column
train_data <- train_data %>% dplyr::select(-Date)
test_data <- test_data %>% dplyr::select(-Date)

###################################################################
### Model Fitting (Model 1: VAR) ##################################
###################################################################

# Create time series object
ts_train <- ts(train_data, frequency = 24*365)
ts_test <- ts(test_data, frequency = 24*365)

# Determine optimal lag order
lag_order <- VARselect(ts_train, lag.max = 48, type = "const")
optimal_lag <- lag_order$selection["AIC(n)"]
print(optimal_lag)

# Fit VAR model
var_model <- VAR(ts_train, p = optimal_lag, type = "const")

summary(var_model$varresult$Rented_Bike_Count)

# Construct forecasting equation
model_summary <- summary(var_model) # Get the summary of the VAR model
coefficients <- model_summary$varresult$Rented_Bike_Count$coefficients # Extract coefficients and p-values for the target variable
significant_vars <- coefficients[coefficients[, 4] < 0.01, , drop = FALSE] # Filter variables with p-value < 0.05
forecasting_equation <- paste(
  "bike_rentals_count =",
  paste(
    sprintf(
      "%.6f * %s",
      significant_vars[, 1], # Coefficient estimates
      rownames(significant_vars) # Variable names
    ),
    collapse = " + "
  )
)
cat(forecasting_equation)

###################################################################
### Forecast (Model 1: VAR) #######################################
###################################################################

# Forecast using the VAR model
forecast_horizon <- nrow(ts_test)
var_forecast <- forecast(var_model, h = forecast_horizon)

# Plot forecasts
single_forecast <- var_forecast$forecast[["Rented_Bike_Count"]]
plot(single_forecast, main = "Forecast for Rented_Bike_Count", xlab = "Time", ylab = "Value")

###################################################################
### Evaluate Results (Model 1: VAR) ###############################
###################################################################

# Extract forecasted values for Rented_Bike_Count
bike_count_forecast <- var_forecast$forecast$Rented_Bike_Count

# Calculate RMSE
actual_values <- test_data$Rented_Bike_Count
predicted_values <- bike_count_forecast$mean
rmse <- sqrt(mean((actual_values - predicted_values)^2))
cat("RMSE:", rmse, "\n")

# Calculate R-squared
ss_total <- sum((actual_values - mean(actual_values))^2)
ss_residual <- sum((actual_values - predicted_values)^2)
r_squared <- 1 - (ss_residual / ss_total)

# Calculate Adjusted R-squared
n <- length(actual_values) # Number of observations
p <- 1 # Number of predictors (in this case, 1 for simple regression)
adjusted_r_squared <- 1 - ((1 - r_squared) * (n - 1) / (n - p - 1))

cat("R-squared:", r_squared, "\n")
cat("Adjusted R-squared:", adjusted_r_squared, "\n")

###################################################################
### Plot Actual Vs Predicted (Model 1: VAR) #######################
###################################################################

plot_data <- data.frame(
  Date = index(test_data),
  Actual = actual_values,
  Predicted = predicted_values
)

ggplot(plot_data, aes(x = Date)) +
  geom_line(aes(y = Actual, color = "Actual")) +
  geom_line(aes(y = Predicted, color = "Predicted")) +
  labs(title = "Actual vs Predicted Rented Bike Count",
       x = "Date",
       y = "Rented Bike Count") +
  scale_color_manual(values = c("Actual" = "blue", "Predicted" = "red")) 
  theme_minimal()

###################################################################
### Model Fitting (Model 2: ARIMAX) ###############################
###################################################################

# Fit ARIMAX model
arimax_model <- auto.arima(
  train_data$Rented_Bike_Count,
  xreg = as.matrix(train_data %>% dplyr::select(-Rented_Bike_Count)),
  seasonal = TRUE,
  stepwise = FALSE,
  approximation = FALSE
)

# Summary of the model
summary(arimax_model) 

# Calculate R-squared
r_squared <- 1 - sum(arimax_model$residuals^2) / sum((train_data$Rented_Bike_Count - mean(train_data$Rented_Bike_Count))^2)
cat("R-squared for ARIMAX model:", round(r_squared, 4), "\n")

###################################################################
### Forecast (Model 2: ARIMAX) ####################################
###################################################################

# Forecast using the ARIMAX model
forecast_horizon <- nrow(test_data)
arimax_forecast <- forecast(arimax_model, 
                            xreg = as.matrix(test_data %>% dplyr::select(-Rented_Bike_Count)),
                            h = forecast_horizon)

# Plot the forecast
plot(arimax_forecast)

###################################################################
### Evaluate Results (Model 2: ARIMAX) ############################
###################################################################
# Calculate RMSE
actual_values <- test_data$Rented_Bike_Count
predicted_values <- arimax_forecast$mean
rmse <- sqrt(mean((actual_values - predicted_values)^2))
cat("RMSE:", rmse, "\n")

# Calculate R-squared
ss_total <- sum((actual_values - mean(actual_values))^2)
ss_residual <- sum((actual_values - predicted_values)^2)
r_squared <- 1 - (ss_residual / ss_total)

# Calculate Adjusted R-squared
n <- length(actual_values) # Number of observations
p <- 1 # Number of predictors (in this case, 1 for simple regression)
adjusted_r_squared <- 1 - ((1 - r_squared) * (n - 1) / (n - p - 1))

cat("R-squared:", r_squared, "\n")
cat("Adjusted R-squared:", adjusted_r_squared, "\n")

###################################################################
### Plot Actual Vs Predicted (Model 2: ARIMAX) #######################
###################################################################

plot_data <- data.frame(
  Date = index(test_data),
  Actual = actual_values,
  Predicted = predicted_values
)

ggplot(plot_data, aes(x = Date)) +
  geom_line(aes(y = Actual, color = "Actual")) +
  geom_line(aes(y = Predicted, color = "Predicted")) +
  labs(title = "Actual vs Predicted Rented Bike Count (ARIMAX)",
       x = "Date",
       y = "Rented Bike Count") +
  scale_color_manual(values = c("Actual" = "blue", "Predicted" = "red")) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5))

###################################################################
### Further Exploration in Tableau ################################
###################################################################  
  
plot_data <- data.frame(
  Date = dimnames(test_data)[[1]],
  Actual = actual_values,
  Predicted = predicted_values
)
  
# Export plot data to a CSV file
write.csv(plot_data, "data/actual_vs_predicted_plot_data_arimax.csv", row.names = FALSE)

