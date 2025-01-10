import pandas as pd
import joblib
import numpy as np

ratings_df = pd.read_csv("Ratings.csv")
books_df = pd.read_csv("Books.csv")

svd_model = joblib.load("svd.pkl")
linreg_model = joblib.load("linreg.pkl")

user_zero_ratings = ratings_df[ratings_df["Book-Rating"] == 0]
user_zero_counts = user_zero_ratings.groupby("User-ID").size()
user_with_most_zeros = user_zero_counts.idxmax()

user_zero_books = user_zero_ratings[user_zero_ratings["User-ID"] == user_with_most_zeros]["ISBN"].values

svd_predictions = svd_model.test([(user_with_most_zeros, book, 0) for book in user_zero_books])
high_rating_books = [pred.iid for pred in svd_predictions if pred.est >= 8]

books_for_linreg = books_df[books_df["ISBN"].isin(high_rating_books)].copy()
books_for_linreg["Predicted-Rating"] = linreg_model.predict(books_for_linreg[["Book-Title", "Book-Author", "Publisher", "Year-Of-Publication"]])

sorted_books = books_for_linreg.sort_values(by="Predicted-Rating", ascending=False)

top_recommendations = sorted_books[["ISBN", "Book-Title", "Book-Author", "Predicted-Rating"]].head(10)

print(top_recommendations)
