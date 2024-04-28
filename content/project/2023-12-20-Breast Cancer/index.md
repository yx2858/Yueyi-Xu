---
title: "Predicting Mortality Status in Patients with Breast Cancer Based on Multiple Approaches"
author: "Jixin Li, Yueyi Xu, Peng Su, Tianhui Huang"
date: "2023-12-11"
excerpt: For my Biostatistical Method I class, my group used Logistic Regression, Survival Analysis, Decision Tree, and Random Forest to predict mortality status in patients with breat cancer.
---



<!--# logistic Regression-->

<!--## clean data-->




## Abstract

This study evaluates the predictive performance of four methods—logistic regression, survival analysis, decision tree, and random forest—in determining the status of breast cancer patients. The results indicate that random forest exhibits the highest prediction accuracy, reaching 89.59%. Additionally, although variations in accuracy exist among the models, there are minimal differences in accuracy within each model when predicting outcomes for both white and black individuals.

## Introduction

Breast cancer is the most common cancer in women and the second leading cause of cancer death in women in the World [1]. In recent years, the incidence of breast cancer has been on the rise, and approximately 685,000 women in the world die from breast cancer every year [2]. Therefore, it is crucial to effectively predict the death status of breast cancer patients. This study uses four methods to predict the death status of breast cancer patients. The second section outlines the steps and algorithms of each method. The third section presents the fitted model and its accuracy, and the fourth section compares and summarizes the results.

## Methods
### Logistics Regression

The first method is intended to conduct a series of logistic regression analyses and explore the interaction between different variables. The process begins by loading the training dataset and converting the status variable to binary numbers. After diagnosing assumptions, variables demonstrating multicollinearity are omitted, and two outliers are removed, a new dataset called “trainOut” is created. Then the logistic regression is fitted (“new_model”) and the first round of feature selection is performed to select the most relevant variables. After that, the most significant interaction is evaluated and added to the model, called “best_fit”. Finally, the second round feature selection is performed on the model with interaction effect and the final model is obtained, called “best_fit2”. The accuracy of best_fit2 is then predicted by applying it to the testing dataset. Lastly, separate accuracy predictions are then carried out for each of these racial categories (white / black) to evaluate potential variations in model performance.

<!-- ## fit a logistic regression model -->



<!-- ## check for assumptions for logistic regression model -->




<!--## fit a new logistic regression model--> 




<!--## interaction -->


<!--## significant interaction -->


<!--## model selection--> 







|                                                               |          x|
|:--------------------------------------------------------------|----------:|
|(Intercept)                                                    |  0.1537419|
|age                                                            | -0.0022053|
|raceOther                                                      |  0.2884935|
|raceWhite                                                      |  0.2209561|
|t_stageT2                                                      | -0.1660725|
|t_stageT3                                                      | -0.2147289|
|t_stageT4                                                      | -0.2764899|
|differentiatePoorly differentiated                             | -0.0670680|
|differentiateUndifferentiated                                  | -0.5659631|
|differentiateWell differentiated                               |  0.3324712|
|estrogen_statusPositive                                        |  0.0582558|
|progesterone_statusPositive                                    |  0.3711592|
|regional_node_examined                                         |  0.0028254|
|reginol_node_positive                                          | -0.0261071|
|survival_months                                                |  0.0096127|
|raceOther:survival_months                                      | -0.0032434|
|raceWhite:survival_months                                      | -0.0025598|
|t_stageT2:survival_months                                      |  0.0018766|
|t_stageT3:survival_months                                      |  0.0022880|
|t_stageT4:survival_months                                      |  0.0017931|
|differentiatePoorly differentiated:progesterone_statusPositive | -0.0849460|
|differentiateUndifferentiated:progesterone_statusPositive      |  0.1160276|
|differentiateWell differentiated:progesterone_statusPositive   | -0.0602197|
|differentiatePoorly differentiated:survival_months             |  0.0013630|
|differentiateUndifferentiated:survival_months                  |  0.0042187|
|differentiateWell differentiated:survival_months               | -0.0031977|
|progesterone_statusPositive:survival_months                    | -0.0037659|
|reginol_node_positive:survival_months                          |  0.0001977|




<!--## prediction -->



## Method
### Survival Analysis

To find the effect of covariates on survival time and their contributions to the outcome, survival analysis using the Cox proportional hazards model is conducted, which allows to compare hazard ratios among the covariates.
1.	Fitted a full Cox model with all variables and possible pairwise interactions as predictors on training data
2.	After evaluating model fit and testing assumptions by “cox.zph()” function, variables with non-significant p-values, which meet the assumptions, were selected, and a new Cox model
3.	The variable selection process was performed by stepwise selection and the final model was tested by the Receiver Operating Characteristic Curve (ROC) and the model’s Area Under the Curve (AUC)
4.	The difference in prediction accuracy of the race was adjusted by adding “weight” parameter to the model.

<!--# survival analysis -->






<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-19-1.png" width="672" />


  

|term                               |    coef| exp(coef)| se(coef)|       z| Pr(>z)|
|:----------------------------------|-------:|---------:|--------:|-------:|------:|
|differentiatePoorly differentiated |  0.5275|    1.6947|   0.1062|  4.9654| 0.0000|
|reginol_node_positive              |  0.0602|    1.0620|   0.0144|  4.1692| 0.0000|
|age                                |  0.0225|    1.0227|   0.0059|  3.7822| 0.0002|
|n_stageN2                          |  0.5015|    1.6512|   0.1326|  3.7815| 0.0002|
|t_stageT4                          |  0.8655|    2.3762|   0.2300|  3.7636| 0.0002|
|regional_node_examined             | -0.0301|    0.9703|   0.0080| -3.7584| 0.0002|
|t_stageT3                          |  0.5533|    1.7390|   0.1585|  3.4914| 0.0005|
|t_stageT2                          |  0.4432|    1.5577|   0.1274|  3.4789| 0.0005|
|marital_statusSeparated            |  1.0386|    2.8252|   0.3467|  2.9956| 0.0027|
|differentiateWell differentiated   | -0.6456|    0.5243|   0.2253| -2.8655| 0.0042|
|n_stageN3                          |  0.5755|    1.7781|   0.2251|  2.5563| 0.0106|
|raceOther                          | -0.6101|    0.5433|   0.2597| -2.3491| 0.0188|
|differentiateUndifferentiated      |  1.1590|    3.1868|   0.5143|  2.2536| 0.0242|
|raceWhite                          | -0.3399|    0.7118|   0.1615| -2.1049| 0.0353|
|marital_statusWidowed              |  0.2766|    1.3186|   0.2155|  1.2836| 0.1993|
|marital_statusSingle               |  0.1725|    1.1883|   0.1844|  0.9354| 0.3496|
|marital_statusMarried              | -0.0518|    0.9495|   0.1527| -0.3391| 0.7345|


```
## Accuracy 0.709062
```




<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-24-1.png" width="672" />




```
## Accuracy white 0.7156398
```




```
## Accuracy black 0.6748768
```




```
## Accuracy 0.6613672
```





<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-32-1.png" width="672" />




```
## Accuracy 0.6682464
```




```
## Accuracy 0.6256158
```


<!--# Machine Learning -->


<!--## Read Dataset-->


<!--## Splite data into training and testing-->


## Method
### Decision Tree

The decision tree is an algorithm for solving classification problems, and the model uses a tree structure. Each node of the decision tree follows the if-then-else rule. Specifically, each node of the tree is judged using a certain attribute value, and based on the judgment result, it is decided which branch to enter until the leaf is reached. In addition, decision tree model is not restricted by any strict assumptions. 

1. Obtain the data \( D = \{ (x_n, y_n) \}_{n=1}^N \).
2. Split the data into two regions \( R_1 \) and \( R_2 \), where \( R_1(s, w) = \{ x : x \leq w \} \) and \( R_2(s, w) = \{ x : x \geq w \} \) for any \( x \in \mathbb{R}^d \).
3. Find \( s \) and \( w \) such that the \( RSS(s, w) \) is minimized:
   \[ RSS(s, w) = \sum_{x_i \in R_1} (y_i - \bar{y}_{R_1})^2 + \sum_{x_i \in R_2} (y_i - \bar{y}_{R_2})^2 \]
4. Repeat Step 1-3 to split the regions based on the regions that could lead to the largest decrease in \( RSS \).


## Decision Tree


<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-40-1.png" width="672" />


\

The accuracy of decision tree model = 89.19%

## Method
### Random Forest Model

Random forest is a model composed of many independent decision trees, used for classification problems. Random forest uses bootstrap sampling, which is a self-service sampling method, to reduce variance and overfitting problems. Specifically, a data set D containing p predictors is randomly sampled n times with replacement to obtain a new dataset containing m predictors. Grow a decision tree using only m<p predictors. The random forest model then evaluates and averages the classification results of each decision tree and treats it as the result. Random forest is not affected by variable multicollinearity and non-constant variance problems.

1. Obtain the data \( D = \{ (x_n, y_n) \}_{n=1}^N \).
2. Draw a bootstrap sample \( (x_1^{(b)}, y_1^{(b)}), (x_2^{(b)}, y_2^{(b)}), \ldots, (x_n^{(b)}, y_n^{(b)}) \), for \( b = 1,2,\ldots,B \).
3. Grow \( n \) decision tree based on the bootstrap sample with only \( m \) selected variables.
4. Evaluate the prediction result from \( B \) bootstrap samples \( \hat{y}^{(1)}, \hat{y}^{(2)}, \ldots, \hat{y}^{(B)} \), obtain the final result \( \hat{y} = B^{-1} \sum_{b=1}^B \hat{y}^{(b)} \).


## Random Forest Model



```
## Warning: Using `size` aesthetic for lines was deprecated in ggplot2 3.4.0.
## ℹ Please use `linewidth` instead.
## This warning is displayed once every 8 hours.
## Call `lifecycle::last_lifecycle_warnings()` to see where this warning was
## generated.
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-43-1.png" width="672" />



<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-45-1.png" width="672" />



<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-47-1.png" width="672" />
\ 

The accuracy of random forest model = 89.43%


<!--## Access the accuracy of white and black-->


* Decision Tree White Accuracy = 89.57%
* Decision Tree Black Accuracy = 84.00%
* Random Forest White Accuracy = 89.86%
* Random Forest Black Accuracy = 86.00%


<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-49-1.png" width="672" />

## Results
### Dataset Description

The dataset “breast cancer” includes 4025 observations and 16 variables. The original dataset is split into a training set (70%) and a testing set (30%). The detailed variable description and summary statistics of numerical and categorical variables are presented in Table1 and Table2.

## Results
### Logistic Regression

“trainOut” removes two outliers 1004 and 2114, shown in Fig 8, and two of three correlated variables n_stage and x6th_stage, shown in Fig 9. “new_model” indicates a refined regression without outliers and correlated variables to meet assumptions. Significant variables include age, race, t stage, differentiate, estrogen status, progesterone status, regional node examined, reginol node positive, and survival months. After the pairwise interaction evaluation process, interactions between survival_months, t_stage, estrogen_status, progesterone_status, and reginol_node_positive are significant and selected into “best_fit”. Following two rounds of model selection, the finalized model, “best_fit2”, is defined as status ~ age + race + t stage + differentiate + estrogen status + progesterone status + reginol node examined + reginol node positive + survival months + race:survival months + t stage:survival months + differentiate:progesterone status + differentiate:survival months + progesterone status:survival months + reginol node positive:survival months. (full model shown below table3). The accuracy of best_fit2 applied to the testing data is 88.00%. Notably, the accuracy for the majority race group White is 88.06%, while for the minority group Black, it stands at 85.00%, shown in table3.   

## Results
### Survival Analysis

After building the full Cox model with all variables that were interested in and several interaction terms, the result of the “cox.zph()” function was displayed in Table 4. The residuals of these covariables against time were shown in Fig 1. It is noticeable that their residuals had certain patterns, which could act as evidence to indicate that these four covariables may violate the assumptions of the Cox model. To keep the GLOBAL p-value (0.87) non-significant, those four variables were excluded, and a new Cox model was established with remaining covariables.
The variable selection process was performed by backward selection, and the final model was constructed as below, and the parameter of the fitted model is shown in Table 5. 
Surv(survival_months, status) ~ age + race + marital_status + t_stage + n_stage + differentiate + regional_node_examined + reginol_node_positive
Finally, the model was adjusted by adding a “weight” parameter to balance the sample size of different races, the prediction accuracy of the original final model and model adjusted by race were calculated for all races (White, Black, and Others). Results were displayed in Table 6 and ROC curves for three classes were included in Fig 2.

## Results
### Decision Tree

The fitted decision tree model is shown in Fig3. The feature selection in the decision tree model is a black box algorithm to obtains the most efficient and accurate model. The decision tree model uses 4 variables for classification, including survival month, age, differential, and marital status. By using the testing dataset, the prediction accuracy of the decision tree is 89.19%, the prediction accuracy for the white race is 89.57%, and for the black race is 84.00%.

## Results
### Random Forest

Random Forest evaluates 500 decision tree models and obtains average results.  After testing the accuracy of 5 -12 variables to grow the tree, the results show that using 11 variables has the highest accuracy, reaching 89.5867% (Fig 4). The prediction accuracy for the white race is 89.9526%, and the accuracy for the black is 87.00%.
In addition,  is also worth noting that the importance of each variable in the random forest tree is obtained by comparing the “Mean Decrease Accuracy”, that is, how much the accuracy of the model will decrease if that variable is removed from the model. According to Fig 5, the variable survival month is the most useful one, followed by reginol node-positive, progesterone status, age, X6th stage, and so on. 


## Conclusion

In the logistic regression, the best_fit2 model suggests that with all else equal, the log odds for t_stage T2 are 0.166 lower than those for t_stage T1, while the log odds for t_stage T3 are 0.214 lower than t_stage T1. Similar interpretations apply to other covariates from Table 1. The impact of selected covariates on status is influenced by the interaction with survival month. The 0.88 accuracy suggests correct predictions for a significant portion (88%) of the dataset. Differences in accuracy between race groups hint at slightly better performance in the White group.
In the survival analysis, Age, T stage, N stage, Regional Node Examined, and Reginol Node Positive had significantly affected the survival of a breast cancer patient. Furthermore, by comparing the predictive accuracy of the final model across race groups, the model exhibited slightly superior predictive performance for White individuals (71.56%) compared to black (67.49%). This observation is further supported by the AUC, where the AUC for black individuals was found to be lower than that for white counterparts. 
After adjusting the model by introducing “weight” parameters associated with the race variable during the model refinement process, analysis of the adjusted model’s accuracy metrics and ROC curves reveal that the augmentation of weights, although leading to an overall decline in the model’s performance, concurrently results in a reduction of the accuracy gap in predicting survival outcomes among individuals of diverse racial backgrounds.
The decision tree model gives a promising prediction accuracy of 89.19% and it includes survival month, age, differential, and marital status. According to the model in Fig 3, if the individual’s survival month is greater than or equal to 48 months, the probability that the individual will be alive is 89%. If the individual’s survival month is less than 48 months and the individual is less than 56 years old and the Tumor size is less than 16, then the probability of being alive is only 1%. The interpretation in other branches is similar.
The prediction accuracy of random forest using 11 variables is the most optimal one, with an accuracy of 89.59%. The random forest does not give a model with a parameter but based on the “Mean Decrease Accuracy”, the model uses the following 11 variables to perform the classification: Survival.Months, Reginol.Node.Positive, Progesterone.Status, Age, X6th.Stage, Estrogen.Status, Tumor.Size, N.Stage, Regional.Node.Examined, differentiate and T.Stage. It is worth noting that the importance of variable survival month is much higher than other variables. 
Generally, four models are used to predict the status of breast cancer patients. The prediction accuracy between different methods is shown in Fig6. The performance of the random forest is the best, with an accuracy of 89.59%.  The prediction accuracy of the White and Black Race by different methods is shown in Fig7. The accuracy between different races by the same method does not differ a lot. Therefore, our model is not limited by race type. 

## Reference

[1]	Loibl, Sibylle et al. “Breast cancer.” Lancet (London, England) vol. 397,10286 (2021): 1750-1769. doi:10.1016/S0140-6736(20)32381-3

[2]	Arnold, Melina et al. “Current and future burden of breast cancer: Global statistics for 2020 and 2040.” Breast (Edinburgh, Scotland) vol. 66 (2022): 15-23. doi:10.1016/j.breast.2022.08.010




