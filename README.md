# new_user_booking_pred

# Airbnb New User Booking Predictions
## Introduction

Predict new users' first bookings for their stay in a specific country. Details can be found in a Kaggle competion [here](https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings). 

## Data Overview

There are 6 files provided. Two of these files provide background information (countries.csv and age_gender_bkts.csv), while sample_submission_NDF.csv provides an example of how the submission file containing our final predictions should be formatted. The three remaining files are the key ones:

&nbsp;&nbsp;&nbsp;&nbsp;1. *train_users_2.csv* – This dataset contains data on Airbnb users, including the destination countries.

&nbsp;&nbsp;&nbsp;&nbsp;2. *test_users.csv* – This dataset also contains data on Airbnb users, in the same format as train_users_2.csv, except without the destination country. These are the users for which we will have to make our final predictions.

&nbsp;&nbsp;&nbsp;&nbsp;3. *sessions.csv* – This data is supplementary data that can be used to train the model and make the final predictions. It contains information about the actions (e.g. clicked on a listing, updated a  wish list, ran a search etc.) taken by the users in both the testing and training datasets above.
