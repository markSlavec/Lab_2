# -*- coding: cp1251 -*-
import numpy as np 
import pandas as pd

# ������� 1.1
# �������� ������ ��������
recipes = pd.read_csv('recipes_sample.csv', parse_dates=['submitted'], date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d'))
#recipes.name = "recipes"
# �������� ������ �������
reviews = pd.read_csv("reviews_sample.csv", index_col=0)
#reviews.name = "reviews"

# ������� 1.2

def analyze_dataframe(df):
    #print(f"��� �������: {df.name}")
    print(f"���������� ����� ������ (�����): {df.shape[0]}")
    print(f"���������� ��������: {df.shape[1]}")
    print("���� ������ ������� �������:")
    print(df.dtypes)
    print("\n")

print("������� 1.2: \n")
print("Recipes: ")
analyze_dataframe(recipes)
print("Reviews: ")
analyze_dataframe(reviews)


# ������� 1.3
def explore_missing_values(df):
    df_gaps = df.isna().any(axis=1).mean()
    print(f"���� ����� � �������������� ���������� � �������: {df_gaps:.2%} \n")
    print()

print("������� 1.3 \n")

print("Recipes ")
explore_missing_values(recipes)

print("Reviews ")
explore_missing_values(reviews)


# ������� 1.4

# ������� �������� ��� ������� ��������� ������� � ������� ��������
recipe_minutes_mean = recipes['minutes'].mean()
recipe_n_ingredients_mean = recipes['n_ingredients'].mean()

# ������� �������� ��� ������� �������� � ������� �������
review_rating_mean = reviews['rating'].mean()


# ������� 1.5

# 10 ��������� �������� ��������
random_indices = np.random.choice(len(recipes), size=10, replace=False)
random_recipes = recipes.iloc[random_indices]

print("������� 1.5: \n")
print("10 ��������� ��������:")
print(random_recipes )
print()


# ������� 1.6
# ����� ������� ������� �������
reviews = reviews.reset_index(drop=True)


# ������� 1.7
result = recipes[(recipes['minutes'] <= 20) & (recipes['n_ingredients'] <= 5)]
print("������� 1.7\n")
print(f"���������� � ��������, ��������� �� ����� 20 ����� � �� ����� 5 ������������: /n{result}/n")


# ������� 2.1
# �������������� �������� 'submitted' � ������ �������
recipes['submitted'] = pd.to_datetime(recipes['submitted'], format='%Y-%m-%d')

# ���������� DataFrame
print("������� 2.1: \n")
print(recipes.head())


#������� 2.2
# �������, �������������� �� 2011 ����
recipes_after_2009 = recipes[recipes['submitted'] >= '2010-01-01']

# display information about filtered recipes
print("������� 2.2: \n")
print(recipes_after_2009[['id', 'name', 'submitted']])
print()


# ������� 3.1
# ���������� ������ �������
recipes['description_length'] = recipes['description'].str.len()


# ������� 3.2
# ��������� �������� ��������, ����� �������, ��� ������ ����� �
# �������� ���������� � ��������� �����.
recipes['name'] = recipes['name'].apply(lambda x: x.capitalize())


# ������� 3.3
# ������� �������� �������� � ���������� ����
name_word_count = recipes['name'].str.replace(r'\s+', ' ').str.strip().str.title().str.split().apply(len)
# add ������� name_word_count � ������� ��������
recipes['name_word_count'] = name_word_count


# ������� 4.1 

# ���������� ��������, �������������� ������ ����������.
contributions = recipes.groupby('contributor_id')['id'].count()

# ����� ���������, ������� ������� ������������ ���������� ��������
max_contributor = contributions.idxmax()
print("(4.1):")
print(f"\n��������, ���������� ������������ ���������� ��������: {max_contributor}")
print()


# ������� 4.2

# ������� ������� ��� ������� �������
avg_ratings = reviews.groupby('recipe_id')['rating'].mean()

recipe_ratings = pd.merge(recipes[['name', 'id']], avg_ratings, left_on='id', right_on='recipe_id', how='left')

# ���������� �������� � �������������� ��������
missing_reviews = recipe_ratings['rating'].isnull().sum()

print("������� 4.2 \n")
print("������� ������ ��� ������� �������: ")
print(recipe_ratings[['name', 'id', 'rating']])
print("���������� �������� � �������������� ��������:", missing_reviews)


# ������� 4.3

recipes['year'] = pd.to_datetime(recipes['submitted']).dt.year

# ������� ���������� �������� �� �����r
recipe_count = recipes.groupby('year')['id'].count()


print("������� 4.3: ")
print("���������� �������� �� �����:")
print(recipe_count)
print()


# ������� 5.1

#  ������ ��� �������
recipe_without_reviews = recipes.loc[~recipes['id'].isin(reviews['recipe_id']), 'id'].iloc[0]

df_without_review = pd.merge(reviews[['recipe_id', 'rating', 'user_id']], recipes[['id', 'name']], left_on='recipe_id', right_on='id')

df_without_review = df_without_review[['id', 'name', 'user_id', 'rating']]

df_without_review = df_without_review[df_without_review['id'].isin(reviews.recipe_id)]

# ��������, ���� �� ������ ��� �������
print("������� 5.1: \n")
if recipe_without_reviews in df_without_review['id'].tolist():
    print("������ ��� ������� � �������" + recipe_without_reviews)
else:
    print("������ ��� ������� �� ������ � �������")




#5.2

df_with_review = pd.merge(recipes, reviews.groupby('recipe_id').size().reset_index(name='review_count'), how='left', left_on='id', right_on='recipe_id')
df_with_review = df_with_review[['id', 'name', 'review_count']].fillna(0)

# ���������, ���� �� ������ ��� �������
print("������� 5.2: \n")
if  recipe_without_reviews in df_with_review['id'].tolist():
    print("������ ��� ������� � �������:", recipe_without_reviews)
else:
    print("Recipe without reviews is not in the table")



#5.3

merged = pd.merge(recipes, reviews, left_on='id', right_on='recipe_id')

# ������� 'submitted' � ������ ������� ���� � ������� ���
merged['year'] = pd.to_datetime(merged['submitted']).dt.year

# ������������� ����� ������ �� ����� � ���������� ������� ������� ��� ������ ������
grouped = merged.groupby('year')['rating'].mean().reset_index()

# ���������� DataFrame �� �������� �������� � ������� �����������
sorted_df = grouped.sort_values('rating')

print(f"��� � ����� ������ ������� ��������� {sorted_df.iloc[0]['year']}")



#6.1

recipes['name_word_count'] = recipes['name'].str.split().str.len()

recipes = recipes.sort_values(by='name_word_count', ascending=False)

# ��������� ���������� ������� �������� � CSV-����
recipes.to_csv('modified_recipes.csv', index=False)



#6.2

# ���������� ������ � ���� Excel
writer = pd.ExcelWriter('results.xlsx')

# ��������� ���������� ������� 5.1 
df_without_review.to_excel(writer, sheet_name='Rated Recipes', index=False)

#  ���������� ������� 5.2 �� ����� ��� ���������
df_with_review.to_excel(writer, sheet_name='Number of feedback on recipes', index=False)

writer.save()
