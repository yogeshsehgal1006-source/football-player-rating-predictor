# football-player-rating-predictor
A machine learning project investigating if player statistics can create a rating that evaluates closer to fan ratings, rather then purely statistic based ratings presented by applications such as Fotmob.

Overview:
An issue I came across as an Arsenal fan was that fans would sometimes rate a player’s performance much higher or lower than a statistic-based rating model i.e. like Fotmob. 
I wanted to create a model that benefits from the “eye-test” perspective and not just strictly from a statistical standpoint.
So, this project aims to produce a rating that considers the fan perception of a football players performance whilst combining it with the Fotmob rating. This model aims to produce a rating system that is balanced between stats and fan perception.

Objectives:
1.	Create a fresh dataset of the Arsenal player performances over the 25/26 UEFA champions league.
2.	Investigate the relationship between stats and fan ratings.
3.	Group players into Defenders, Midfielders and Attackers and create position based statistical weights.
4.	Convert player statistics into per 90 statistics.
5.	Create a statistical player rating from the weighted statistics.
6.	Combine the statistical player rating with the Fotmob	rating.
7.	Find the best balance between Fotmob and the statistical rating to provide the lowest Mean Absolute Error(MAE)
8.	Compare the final model against the original Fotmob ratings using Mean Absolute Error(MAE) and R^2.
9.	Investigate if the new model produces ratings with a better MAE and R^2 then Fotmob.

Dataset:
I manually constructed the dataset using the UEFA champions league data from the 25/26 season. The dataset consists of 40 columns and 151 rows. However, after preprocessing we only have 140 usable rows of data since I could not gain the fan ratings for one of the matches. 

The general positions like LB,CB,RB were grouped into “Defenders”. CM.CDM.CAM were grouped into “Midfielders” and LW,RW,ST were grouped into “Attackers”. Thus, we had 56 observations of Defenders, 42 of Midfielders and 42 of Attackers. This was important as I needed to create position specific weights for each feature we were using. For example, a goal may provide a higher weight rating for an attacker then a midfielder.

Preparing the data:
I cleaned positions by removing whitespace and converting it all to uppercase letters. I also made sure to remove any rows with null data regarding fan ratings. Statistics would not be relevant if they were not based on a per 90 scale. This allows players to be fairly rated based on the number of minutes that they play.

Feature Selection:
Different features were used for each position group, as each group have different roles. I did not use every feature like “xg_xa” as this was simply a product of the features “xg” and “xa”.

Lasso Regression:
This was used to learn the relationship between stats and fan ratings; the model was made to train separately for each position group. I used this as when testing some stats were given a negative contribution which was something i wanted to avoid.

Trained Weights:
Defenders:
The model placed the most weight for goals per 90, clearances per 90 and the lowest for blocks per 90.
Midfielders:
The model placed the most weight for interceptions per 90, goals per 90, xg per 90 and the lowest for passes per 90.
Attackers:
The model placed the most weight for goals per 90, shots per 90 and the lowest weight for ground duel percentage.

Statistical Rating:
For each position group we had the following formula:
Statistical Score = The sum of (standardized statistic * position-specific weight)
Statistical Rating = 7.2 + 0.65 * standardized statistical score.(I performed a grid search from base ratings 2.00 to 9.99 and spread values from 0.00 to 1.00, incrementing in intervals of 0.05. These ranges were chosen to provide full coverage of the scale of player ratings. As a consequence, a base of 7.2 and a spread of 0.65 produced the lowest Mean Absolute Error(MAE).

Combining Fotmob and Statistical Rating:
The final rating was calculated using the following formula:
(1 – lambda) * Fotmob Rating + lambda* Statistical Rating
With this formula we could see what value of lambda provided the best value for the influence of the statistical rating with relevance to Fotmob.

Choosing Lambda:
Used 5-fold cross validation to select the lambda which provided the lowest MAE value, in this case lambda = 0.85 provided the lowest MAE value.

Model Evaluation:
The model was evaluated against the actual fan ratings. The metrics we used to measure this was MAE and R^2 values. MAE is a measure of the average difference between the actual rating and predicted rating. We aspire for the lowest MAE value possible.
The R^2 value measures how much of the variation is explained via the model.
Our results between Fotmob and the model was of the following:
MAE: 0.819(Fotmob) vs 0.747(The model)
The MAE was reduced by 0.072.
R^2: 0.231 (Fotmob( vs 0.360(The model)
The R^2 value increased 0.129.
Overall the model decreased MAE and Increased R^2.

Cross-Validation Results:
The model produced the following 5-fold cross-validations results:
CV MAE : 0.751
CV MAE standard deviation: 0.061
CV R^2 : 0.342
The CV MAE was very close to overall MAE of 0.747, so the model was consistent over the 5 folds. The standard deviation is low as well, so the MAE did not vary much between the folds.

Interpretation:
The final model achieved some great results as per the following:
8.82% lower MAE then Fotmob, a higher R^2 value and a cross validation of ~0.75
Thus. The combined model produced ratings that were closer to fan ratings than the original Fotmob ratings.

Limitations:
1.	Dataset Size: The dataset after preprocessing only had 140 observations, the results of this data may not generalize to other teams or competitions. I aim to collect more data from the 26/27 season and to use the 25/26 premier league season data. I could not use the 25/26 premier league season data as the data was not openly available for me to collect directly from Fotmob' API.
2.	Limited Variables: The model could only use the available statistics in the dataset. Some key metrics were missing, but I aim to improve on this when I can get the whole dataset information from the Fotmob API rather than having to make the dataset manually.
3.	Position Grouping: Positions were grouped into 3 categories, but a CAM has a different responsibility to a CDM.

Conclusion:
I was able to produce a position-specific fan adjusted player rating model using data from the UEFA Champions League 25/26 season for Arsenal. I collected fan ratings from reddit whereby a survey was collected after every match, and an average of the fan ratings produced the final fan ratings.
I used Lasso Regression to learn statistical relationships with fan ratings and was able to develop models depending on the players positions.
My final model achieved an MAE of 0.747 as opposed to Fotmob`0.819, which is an 8.82% improvement in MAE. The r^2 also increased from 0.231 to 0.360.
This data suggests that I have created a fan-adjusted player rating system which is more in line with what fans rate players then what Fotmob rates players in comparison with fan ratings. Despite this, I know there are ways I can improve this model and in the near future I aim to create a model with more data, variables and accuracy in order to lower the MAE value and to increase the R^2 value.

How to Run this project:

1) Run the following in the terminal: git clone https://github.com/yogeshsehgal1006-source/football-player-rating-predictor.git

2) Then run this:
cd football-player-rating-predictor

3) Create a virtual environment:
python -m venv venv

Activate environment (Windows PowerShell)
.\venv\Scripts\Activate

Activate environment (Mac / Linux)
source venv/bin/activate

4) Then install the dependencies:
pip install -r requirements.txt

5) Then run this:
cd PlayerRatingML/final_project

6) Then run these scripts:
python training.py
python rating.py











