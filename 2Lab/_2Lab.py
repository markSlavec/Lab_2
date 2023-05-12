# -*- coding: cp1251 -*-
import numpy as np 
import pandas as pd

# Задание 1.1
# Загрузка данных рецептов
recipes = pd.read_csv('recipes_sample.csv', parse_dates=['submitted'], date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d'))
#recipes.name = "recipes"
# Загрузка данных обзоров
reviews = pd.read_csv("reviews_sample.csv", index_col=0)
#reviews.name = "reviews"

# Задание 1.2

def analyze_dataframe(df):
    #print(f"Имя таблицы: {df.name}")
    print(f"Количество точек данных (строк): {df.shape[0]}")
    print(f"Количество столбцов: {df.shape[1]}")
    print("Типы данных каждого столбца:")
    print(df.dtypes)
    print("\n")

print("Задание 1.2: \n")
print("Recipes: ")
analyze_dataframe(recipes)
print("Reviews: ")
analyze_dataframe(reviews)


# Задание 1.3
def explore_missing_values(df):
    df_gaps = df.isna().any(axis=1).mean()
    print(f"Доля строк с отсутствующими значениями в таблице: {df_gaps:.2%} \n")
    print()

print("Задание 1.3 \n")

print("Recipes ")
explore_missing_values(recipes)

print("Reviews ")
explore_missing_values(reviews)


# Задание 1.4

# Среднее значение для каждого числового столбца в таблице рецептов
recipe_minutes_mean = recipes['minutes'].mean()
recipe_n_ingredients_mean = recipes['n_ingredients'].mean()

# Среднее значение для столбца рейтинга в таблице отзывов
review_rating_mean = reviews['rating'].mean()


# Задание 1.5

# 10 случайных названий рецептов
random_indices = np.random.choice(len(recipes), size=10, replace=False)
random_recipes = recipes.iloc[random_indices]

print("Задание 1.5: \n")
print("10 случайных рецептов:")
print(random_recipes )
print()


# Задание 1.6
# Сброс индекса таблицы отзывов
reviews = reviews.reset_index(drop=True)


# Задание 1.7
result = recipes[(recipes['minutes'] <= 20) & (recipes['n_ingredients'] <= 5)]
print("Задание 1.7\n")
print(f"Информация о рецептах, требующих не более 20 минут и не более 5 ингредиентов: /n{result}/n")


# Задание 2.1
# преобразование столбеца 'submitted' в формат времени
recipes['submitted'] = pd.to_datetime(recipes['submitted'], format='%Y-%m-%d')

# измененный DataFrame
print("Задание 2.1: \n")
print(recipes.head())


#Задание 2.2
# Рецепты, представленные до 2011 года
recipes_after_2009 = recipes[recipes['submitted'] >= '2010-01-01']

# display information about filtered recipes
print("Задание 2.2: \n")
print(recipes_after_2009[['id', 'name', 'submitted']])
print()


# Задание 3.1
# Добавление нового столбца
recipes['description_length'] = recipes['description'].str.len()


# Задание 3.2
# Изменение названий рецептов, таким образом, что каждое слово в
# названии начиналось с прописной буквы.
recipes['name'] = recipes['name'].apply(lambda x: x.capitalize())


# Задание 3.3
# Подсчет названий рецептов и количества слов
name_word_count = recipes['name'].str.replace(r'\s+', ' ').str.strip().str.title().str.split().apply(len)
# add столбец name_word_count в таблицу рецептов
recipes['name_word_count'] = name_word_count


# Задание 4.1 

# Количество рецептов, представленных каждым участником.
contributions = recipes.groupby('contributor_id')['id'].count()

# Поиск участника, который добавил максимальное количество рецептов
max_contributor = contributions.idxmax()
print("(4.1):")
print(f"\nУчастник, добавивший максимальное количество рецептов: {max_contributor}")
print()


# Задание 4.2

# средний рейтинг для каждого рецепта
avg_ratings = reviews.groupby('recipe_id')['rating'].mean()

recipe_ratings = pd.merge(recipes[['name', 'id']], avg_ratings, left_on='id', right_on='recipe_id', how='left')

# количество рецептов с отсутствующими отзывами
missing_reviews = recipe_ratings['rating'].isnull().sum()

print("Задание 4.2 \n")
print("Средние оценки для каждого рецепта: ")
print(recipe_ratings[['name', 'id', 'rating']])
print("Количество рецептов с отсутствующими отзывами:", missing_reviews)


# Задание 4.3

recipes['year'] = pd.to_datetime(recipes['submitted']).dt.year

# Подсчет количества рецептов по годамr
recipe_count = recipes.groupby('year')['id'].count()


print("Задание 4.3: ")
print("Количество рецептов по годам:")
print(recipe_count)
print()


# Задание 5.1

#  рецепт без отзывов
recipe_without_reviews = recipes.loc[~recipes['id'].isin(reviews['recipe_id']), 'id'].iloc[0]

df_without_review = pd.merge(reviews[['recipe_id', 'rating', 'user_id']], recipes[['id', 'name']], left_on='recipe_id', right_on='id')

df_without_review = df_without_review[['id', 'name', 'user_id', 'rating']]

df_without_review = df_without_review[df_without_review['id'].isin(reviews.recipe_id)]

# проверка, есть ли рецепт без отзывов
print("Задание 5.1: \n")
if recipe_without_reviews in df_without_review['id'].tolist():
    print("рецепт без отзывов в таблице" + recipe_without_reviews)
else:
    print("рецепт без отзывов не входит в таблицу")




#5.2

df_with_review = pd.merge(recipes, reviews.groupby('recipe_id').size().reset_index(name='review_count'), how='left', left_on='id', right_on='recipe_id')
df_with_review = df_with_review[['id', 'name', 'review_count']].fillna(0)

# Проверьте, есть ли рецепт без отзывов
print("Задание 5.2: \n")
if  recipe_without_reviews in df_with_review['id'].tolist():
    print("рецепт без отзывов в таблице:", recipe_without_reviews)
else:
    print("Recipe without reviews is not in the table")



#5.3

merged = pd.merge(recipes, reviews, left_on='id', right_on='recipe_id')

# столбец 'submitted' в объект времени даты и извлечь год
merged['year'] = pd.to_datetime(merged['submitted']).dt.year

# Сгруппировать кадры данных по годам и рассчитать средний рейтинг для каждой группы
grouped = merged.groupby('year')['rating'].mean().reset_index()

# Сортировка DataFrame по среднему рейтингу в порядке возрастания
sorted_df = grouped.sort_values('rating')

print(f"Год с самым низким средним рейтингом {sorted_df.iloc[0]['year']}")



#6.1

recipes['name_word_count'] = recipes['name'].str.split().str.len()

recipes = recipes.sort_values(by='name_word_count', ascending=False)

# Сохранить измененную таблицу рецептов в CSV-файл
recipes.to_csv('modified_recipes.csv', index=False)



#6.2

# сохранение данных в файл Excel
writer = pd.ExcelWriter('results.xlsx')

# Сохраните результаты задания 5.1 
df_without_review.to_excel(writer, sheet_name='Rated Recipes', index=False)

#  результаты задания 5.2 на листе под названием
df_with_review.to_excel(writer, sheet_name='Number of feedback on recipes', index=False)

writer.save()
