
# Homework 3 - Survival Analysis Report

## Findings

Several Accelerated Failure Time (AFT) models were fitted to estimate churn risk using the telecom dataset.
The models included Weibull, Log-Normal, Log-Logistic and Exponential distributions. After comparing
AIC values, coefficient significance, and interpretability, the Weibull AFT model was selected as the
final model.

The analysis showed that customer tenure strongly reduces churn risk. Customers with longer service
history are less likely to churn. Internet and voice service users generally demonstrate higher retention,
while customers in lower-value categories are more likely to leave earlier. Income also contributes
positively to retention because higher-income subscribers tend to maintain stable telecom subscriptions.

The most valuable segment consists of long-tenure customers with bundled services such as internet and
voice. These customers exhibit both high survival probabilities and higher estimated CLV values. Customers
with low tenure and basic services are identified as the highest-risk group.

The estimated annual retention budget was calculated using customers identified as high-risk within the
next year based on survival probabilities and predicted CLV values. A practical retention strategy would
focus on targeted promotions, loyalty rewards, bundled offers, and personalized engagement campaigns
for high-value at-risk subscribers.
