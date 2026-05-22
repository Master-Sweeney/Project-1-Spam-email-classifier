This is a Supervised Machine Learning model I built to detect Spam emails from user input. 
The model trained was a Multinomial Naive Bayes model and the CountVector was used to 
convert text into a matrix of token counts. This created a "vocabulary" of all words in the emails read into the pandas dataframe.
The model's performance was evaluated using the performance metrics of:
-> Accuracy score
-> Precision
-> Recall
-> Confusion Matrix 
And finally, a Classification Report was provided as well.
Future developments of the model will include using cross validation to split the dat into more folds for training and validation. 
As well as a further exploration into optimizing the hyperparameters by using a grid search.
