if (!require("jsonlite")) install.packages("jsonlite")
if (!require("dplyr")) install.packages("dplyr")
library(jsonlite)
library(dplyr)

source("/opt/BAA/FARM/FARM_Skript/R/farm.R")
source("/opt/BAA/FARM/FARM_Skript/R/farm.dist.R")
source("/opt/BAA/FARM/FARM_Skript/R/tw.decomp.R")

csv_file <- "/opt/BAA/FARM/data/Rathausquai.csv"

data <- read.csv(csv_file)

csv_name <- tools::file_path_sans_ext(basename(csv_file))

data$date <- as.POSIXct(data$time, format = "%Y-%m-%d %H:%M:%S")

target <- data$visitors
weather_columns <- c("Wind.Speed", "Sunshine.Duration", "Air.Pressure", 
                     "Absolute.Humidity", "Precipitation.Duration", "Air.Temperature")

output_dir <- "plots/"

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

for (weather_col in weather_columns) {
  refTS <- data[[weather_col]]
  
  if (any(is.na(refTS)) || any(is.na(target))) {
    next
  }
  
  result <- farm(
    refTS = refTS,
    qryTS = target,
    lcwin = 3,
    rel.th = 10,
    fuzzyc = 1,
    metric.space = TRUE
  )
  
  global_relevance <- result$rel.global
  data$local_relevance <- result$rel.local

  data$hour <- format(data$date, "%H")
  data$hour <- as.numeric(data$hour)
  hourly_avg <- data %>%
    group_by(hour) %>%
    summarise(avg_relevance = mean(local_relevance, na.rm = TRUE))

  plot_file <- paste0(output_dir, csv_name, "_relevance_", gsub("\\.", "_", weather_col), ".png")
  
  png(filename = plot_file, width = 800, height = 600)

  plot(hourly_avg$hour, hourly_avg$avg_relevance, type = "l", col = "blue", lwd = 2,
       xlab = "Hour of Day", ylab = "Relevance",
       main = paste("Relevance of", gsub("\\.", " ", weather_col), "in", csv_name),
       ylim = c(0, max(hourly_avg$avg_relevance, global_relevance) * 1.2))

  abline(h = global_relevance, col = "red", lty = 2, lwd = 2)

  abline(v = hourly_avg$hour, col = "gray", lty = 3)

  legend("topright", legend = c("Local Relevance", "Global Relevance"),
         col = c("blue", "red"), lty = c(1, 2), lwd = 2)

  dev.off()
}

print(paste("Relevance plots have been saved to the 'plots' directory under the name", csv_name))
