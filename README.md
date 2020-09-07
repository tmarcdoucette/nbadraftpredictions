# nbadraftpredictions
Predicting which NCAA players from the 2019-2020 season will be selected for the NBA draft.

A re-creation of the content of our term project for Big Data Analytics, using different classification models to predict which NCAA players will get drafted in the 2020 NBA Draft given information on players from the 2010-2019 NCAA Basketball Seasons.

Link to related medium article: https://medium.com/@tmarcdoucette/machine-learning-applications-in-predicting-nba-draft-picks-and-reflecting-on-rushed-year-b61c2cf2e9f

Code will run with the specified columns of variables in the 'featured_cols' lists. To change lists, change how they are reference to X_pred.
Produces a Confusion Matrix, the relevant confusion matrix ratio, a culmulative gain chart, and a dataframe of who will be drafted from said model.

Code to export: resultdf.to_csv('FILEPATH/drafted.csv')

