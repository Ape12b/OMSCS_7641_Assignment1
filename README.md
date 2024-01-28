• A description of your classification problems, and why you feel they are interesting. Think hard about
this. To be interesting the problems should be non-trivial on the one hand, but capable of admitting
comparisons and analysis of the various algorithms on the other. Avoid the mistake of working on the
largest most complicated and messy dataset you can find. The key is to be interesting and clear, no points
for hairy and complex.
• The training and testing error rates you obtained running the various learning algorithms on your problems.
At the very least you should include graphs that show performance on both training and test data as a
function of training size (note that this implies that you need to design a classification problem that has
more than a trivial amount of data) and – for the algorithms that are iterative – training times/iterations.
Both of these kinds of graphs are referred to as learning curves.
• Graphs for each algorithm showing training and testing error rates as a function of selected hyperparameter
ranges. This type of graph is referred to as a model complexity graph (also sometimes validation curve).
Please experiment with more than one hyperparameter and make sure the results and subsequent analysis
you provide are meaningful.
• Analyses of your results. Why did you get the results you did? Compare and contrast the different
algorithms. What sort of changes might you make to each of those algorithms to improve performance?
How fast were they in terms of wall clock time? Iterations? Would cross validation help? How much
performance was due to the problems you chose? Which algorithm performed best? How do you define
best? Be creative and think of as many questions you can, and as many answers as you can.

# Heart Failure Prediction Dataset

Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide. Four out of 5 CVD deaths are due to heart attacks and strokes, and one-third of these deaths occur prematurely in people under 70 years of age. Heart failure is a common event caused by CVDs, and this dataset contains 11 features that can be used to predict a possible heart disease.

People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidemia, or already established disease) need early detection and management wherein a machine learning model can be of great help.

I chose this dataset because I have a Ph.D. in mechanobiology where I studied vascular cells. My projects focussed on phenotyping the cells and differentiation between healthy and diseased cells. Hence, I wanted to delve more into the causes of heart diseases using Machine Learning. 

## Attribute Information

- **Age**: age of the patient [years]
- **Sex**: sex of the patient [M: Male, F: Female]
- **ChestPainType**: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
- **RestingBP**: resting blood pressure [mm Hg]
- **Cholesterol**: serum cholesterol [mm/dl]
- **FastingBS**: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
- **RestingECG**: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
- **MaxHR**: maximum heart rate achieved [Numeric value between 60 and 202]
- **ExerciseAngina**: exercise-induced angina [Y: Yes, N: No]
- **Oldpeak**: oldpeak = ST [Numeric value measured in depression]
- **ST_Slope**: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
- **HeartDisease**: output class [1: heart disease, 0: Normal]

## Source

This dataset was created by combining different datasets already available independently but not combined before. In this dataset, 5 heart datasets are combined over 11 common features which makes it the largest heart disease dataset available so far for research purposes. The five datasets used for its curation are:

- Cleveland: 303 observations
- Hungarian: 294 observations
- Switzerland: 123 observations
- Long Beach VA: 200 observations
- Stalog (Heart) Data Set: 270 observations

**Total:** 1190 observations  
**Duplicated:** 272 observations  
**Final dataset:** 918 observations

Every dataset used can be found under the Index of heart disease datasets from UCI Machine Learning Repository on the following link: [UCI Heart Disease Datasets](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/)

## Citation

fedesoriano. (September 2021). Heart Failure Prediction Dataset. Retrieved [Date Retrieved] from [Heart Failure Prediction Dataset on Kaggle](https://www.kaggle.com/fedesoriano/heart-failure-prediction).

## Acknowledgements

**Creators:**

- Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
- University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
- University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
- V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.

**Donor:**

David W. Aha (aha '@' ics.uci.edu) (714) 856-8779


# New York Houses Dataset

## Description

This dataset provides information on New York houses, offering valuable insights into the real estate market in the region. It includes details such as broker titles, house types, prices, number of bedrooms and bathrooms, property square footage, addresses, state, administrative and local areas, street names, and geographical coordinates.

I chose this dataset as I live in New York and I have personally struggled with finding affordable housing here. This gives me a chance to undesratand the driving forces behind the prices in the New York real estate market. 

## Key Features

- **BROKERTITLE**: Title of the broker
- **TYPE**: Type of the house
- **PRICE**: Price of the house
- **BEDS**: Number of bedrooms
- **BATH**: Number of bathrooms
- **PROPERTYSQFT**: Square footage of the property
- **ADDRESS**: Full address of the house
- **STATE**: State of the house
- **MAIN_ADDRESS**: Main address information
- **ADMINISTRATIVE_AREA_LEVEL_2**: Administrative area level 2 information
- **LOCALITY**: Locality information
- **SUBLOCALITY**: Sublocality information
- **STREET_NAME**: Street name
- **LONG_NAME**: Long name
- **FORMATTED_ADDRESS**: Formatted address
- **LATITUDE**: Latitude coordinate of the house
- **LONGITUDE**: Longitude coordinate of the house