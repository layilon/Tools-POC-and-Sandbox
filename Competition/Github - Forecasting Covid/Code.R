#==============================================================#
# Forecasting New Covid-19 Cases in Canada - Time Series Model #
#==============================================================#

# Import libraries 
library(dplyr)
library(readr)
library(readxl)
library(tidyverse)
library(mice)
library(glmnet)
library(tidyr)
library(forecast)
library(fpp)

# Import from Data Source: https://github.com/owid/covid-19-data/tree/master/public/data
Covid19casesfile="https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
Covid19cases.data<-read_csv(url(Covid19casesfile))

# Prepare Canada dataset
CAN.cases.data = Covid19cases.data %>% filter(Covid19cases.data$iso_code == "CAN")
CANNewcase<- CAN.cases.data$new_cases
plot.ts(CAN.cases.data$new_cases)
plot.ts(CANNewcase, xlab="No. of Days since 2020-01-26", ylab="No. of New Covid cases in Canada")

##### Part 1 #####
## This part use auto.arima on the original time series (i.e., without preprocessing) to make forecasting ##

model.raw <- auto.arima(CANNewcase, stepwise=FALSE, seasonal= FALSE)
model.raw # It suggests a ARIMA(0,1,5) model
checkresiduals(model.raw) # Check the quality of fit. 

# Use the auto selected model to make forecasting 
fit.raw <- Arima(CANNewcase, order=c(0,1,5))
fit.raw # AIC = 9796.12
autoplot(forecast(fit.raw,20) ) # Plot the forecasting

##### Part 2 #####
## This part forecasting by first preprocessing the time series before applying auto.arima. ##

# Preprocessing 

CANNewcase.processed<-tail(CANNewcase,-100) # Remove the first 100 days when the case numbers are small
CANNewcase.processed <- ts(CANNewcase.processed, frequency=365, start=c(2020, 122))
plot.ts(CANNewcase, xlab="No. of days since August 9", ylab="Canada New cases")

#-----------(1) Stabilizing the Variance---------------
# Transform the original data using Box-Cox Transform
CANNewcase.lambda <- BoxCox.lambda(CANNewcase.processed)
# The transformation will use a parameter of lambda = 1
CANNewcase.BoxCox<-BoxCox(CANNewcase.processed, CANNewcase.lambda)

# Check the transformed data and compare it with log transform
par(mfrow=c(1,2))
plot.ts(CANNewcase.BoxCox, xlab="Date", ylab="(Box-Cox)Canada New cases")
plot.ts(log(CANNewcase.processed), xlab="Date", ylab="log Canada New cases")

#-----------(2) Remove Seasonality through Seasonal Difference ---------------
par(mfrow=c(1,2))
Acf(diff(CANNewcase.BoxCox,1),lag.max =25)
Acf(diff(log(CANNewcase.processed),1),lag.max =25) 

# Remove the seasonality using the seasonal difference (p=7)
CANNewcase.BoxCox.deSeasonality <- diff(CANNewcase.BoxCox,7) 
logCANNewcase.processed.deSeasonality <- diff(log(CANNewcase.processed),7) 
logCANNewcase.processed.deSeasonality

# Check the transformed data and compare it with log transform
plot.ts(CANNewcase.BoxCox.deSeasonality, xlab="Date", ylab="(BoxCox) Canada New Case after removing trend and seasonality")
plot.ts(logCANNewcase.processed.deSeasonality, xlab="Date", ylab="log Canada New Case after removing trend and seasonality")

# Check the period of cyclic pattern again with the autocorrelation function 
Acf(CANNewcase.BoxCox.deSeasonality,lag.max =25)
Acf(logCANNewcase.processed.deSeasonality,lag.max =25) 


#-------------Check Stationarity -------------------
# Perform the augmented Dickey-Fuller (ADF) test to check stationarity. The null hypothesis assumes that the series is non-stationary.
adf.test(logCANNewcase.processed.deSeasonality,alternative = "stationary") # p-value = p value = 0.01

#-------------Automatic ARIMA Modeling -------------------
# Use auto.arima function to recommend ARIMA() parameters 
model.auto.logCan.deseason <- auto.arima( logCANNewcase.processed.deSeasonality, stepwise=FALSE, seasonal= FALSE) 
model.auto.logCan.deseason
checkresiduals(model.auto.logCan.deseason)
# Use logCan.deseason model with ARIMA(0,0,5) with zero mean


# We can use the auto selected model to make forecasting 
fit.logCan <- Arima(log(CANNewcase.processed), order=c(0,0,5), seasonal=list(order=c(0,1,0),period=7)) # The seasonal differencing with period=7 is equivalent to "seasonal=list(order=c(0,1,0),period=7)"
fit.logCan
autoplot( forecast(fit.logCan,14) )

# Plot the forecasting in the original scale
fc.Can<-forecast(fit.logCan,14)

fc.Can$x <- exp(fc.Can$x)
fc.Can$mean <- exp(fc.Can$mean)
fc.Can$lower <- exp(fc.Can$lower)
fc.Can$upper <- exp(fc.Can$upper)
fc.Can$series <- "No of New Cases in Canada"
par(mfrow=c(1,1))
autoplot(fc.Can)

##### Part 3 #####
#-------------Improving the Automatically selected Model -------------------
# Explore other models manually to yield a lower criterion than the existing model
AIC(Arima(log(CANNewcase.processed), order=c(0,0,5), seasonal=list(order=c(0,1,0),period=7)))
AIC(Arima(log(CANNewcase.processed), order=c(0,0,7), seasonal=list(order=c(0,1,0),period=7)))
AIC(Arima(log(CANNewcase.processed), order=c(0,0,9), seasonal=list(order=c(0,1,0),period=7)))
AIC(Arima(log(CANNewcase.processed), order=c(0,0,11), seasonal=list(order=c(0,1,0),period=7)))
AIC(Arima(log(CANNewcase.processed), order=c(0,0,13), seasonal=list(order=c(0,1,0),period=7)))
AIC(Arima(log(CANNewcase.processed), order=c(0,0,15), seasonal=list(order=c(0,1,0),period=7)))
AIC(Arima(log(CANNewcase.processed), order=c(0,0,17), seasonal=list(order=c(0,1,0),period=7)))
AIC(Arima(log(CANNewcase.processed), order=c(0,0,19), seasonal=list(order=c(0,1,0),period=7)))
AIC(Arima(log(CANNewcase.processed), order=c(0,0,21), seasonal=list(order=c(0,1,0),period=7)))


fit.alternative1 <- Arima(log(CANNewcase.processed), order=c(0,0,15), seasonal=list(order=c(0,1,0),period=7)) 
fit.alternative1
checkresiduals(fit.alternative1)
fc1<-forecast(fit.alternative1,14)

fc1$x <- exp(fc1$x)
fc1$mean <- exp(fc1$mean)
fc1$lower <- exp(fc1$lower)
fc1$upper <- exp(fc1$upper)
fc1$series <- "No of New Cases in Canada"
autoplot(fc1) # AIC = 235.15


##### Part 4 #####
vaccinationfile="https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/vaccinations.csv"
vaccination.data<-read_csv(url(vaccinationfile))

CAN.vax.data = vaccination.data %>% filter(vaccination.data$iso_code == "CAN")
CAN.vax.count<- CAN.vax.data$daily_vaccinations

# Lagged predictors: P=+90,+180,+270
CAN.vax.count.lag <- cbind(CAN.vax.count[],
                           c(NA,CAN.vax.count[90:275]),
                           c(NA,NA,CAN.vax.count[180:275]),
                           c(NA,NA,NA,CAN.vax.count[270:275]))
colnames(CAN.vax.count.lag) <- paste("DailyVaccinations",0:3,sep="")
CAN.vax.count.lag <-tail(CAN.vax.count.lag,-1)
 
CANNewcase.Tminus274 <- tail(CANNewcase.processed,274)

# Choose optimal lag length for advertising based on AIC
# Restrict data so models use same fitting period
fit1 <- auto.arima(log(CANNewcase.Tminus274[5:274]), xreg=CAN.vax.count.lag[5:274,1], d=0)
fit2 <- auto.arima(log(CANNewcase.Tminus274[5:274]), xreg=CAN.vax.count.lag[5:274,1:2], d=0)
fit3 <- auto.arima(log(CANNewcase.Tminus274[5:274]), xreg=CAN.vax.count.lag[5:274,1:3], d=0)
fit4 <- auto.arima(log(CANNewcase.Tminus274[5:274]), xreg=CAN.vax.count.lag[5:274,1:4], d=0)

# Compute Akaike Information Criteria
AIC(fit1)
AIC(fit2)
AIC(fit3)
AIC(fit4)

# Compute Bayesian Information Criteria
BIC(fit1)
BIC(fit2)
BIC(fit3)
BIC(fit4)

#Best fit (as per AIC and BIC) is with all data (1:4), so the final model becomes
fit.dym <- auto.arima(log(CANNewcase.Tminus274[5:274]), xreg=CAN.vax.count.lag[5:274,1:4], d=0)
fit.dym

CAN.vax.count.AvgTminus14 <- mean(tail(CAN.vax.count,14))

regx1 <- cbind(c(CAN.vax.count.lag[274,1],rep(CAN.vax.count.AvgTminus14,13)),rep(CAN.vax.count.AvgTminus14,14),rep(CAN.vax.count.AvgTminus14,14),rep(CAN.vax.count.AvgTminus14,14))
colnames(regx1) <- paste("DailyVaccinations",0:3,sep="")

fc.dym.1 <- forecast(fit.dym, xreg=regx1, h=14)
fc.dym.1$x <- exp(fc.dym.1$x)
fc.dym.1$mean <- exp(fc.dym.1$mean)
fc.dym.1$lower <- exp(fc.dym.1$lower)
fc.dym.1$upper <- exp(fc.dym.1$upper)
fc.dym.1$series <- "No of New Cases in Canada"
autoplot(fc.dym.1) 
plot(fc.dym.1 , main="Forecast new cases with Daily vaccinations set to Average of Last 14 days", ylab="New Covid Cases")

summary(CAN.vax.count) # median=154254 

regx2 <- cbind(c(CAN.vax.count.lag[274,1],rep(154254,13)),rep(154254,14),rep(154254,14),rep(154254,14))
colnames(regx2) <- paste("DailyVaccinations",0:3,sep="")
fc.dym.2 <- forecast(fit.dym, xreg=regx2, h=14)
fc.dym.2$x <- exp(fc.dym.2$x)
fc.dym.2$mean <- exp(fc.dym.2$mean)
fc.dym.2$lower <- exp(fc.dym.2$lower)
fc.dym.2$upper <- exp(fc.dym.2$upper)
fc.dym.2$series <- "No of New Cases in Canada"
autoplot(fc.dym.2) 
plot(fc.dym.2, main="Forecast new cases with Daily vaccinations set to Median of Daily Vaccinations YTD", ylab="New Covid Cases")
