# Кластерный анализ вина

В данном проекте мною проведена обработка физико-химических данных о вине методами машинного обучения, визуализация данных и кластеризация этого набора данных

Данные получены из открытого датасета с Kaggle - https://www.kaggle.com/datasets/harrywang/wine-dataset-for-clustering

## О наборе данных

Эти данные являются результатами химического анализа вин, выращенных в одном и том же регионе Италии, но полученных из трех разных сортов. Анализ определил количество 13 компонентов, обнаруженных в каждом из трех типов вин.

## Обработка и визуализация данных

Данные были обработаны методами главных компонент ( sklearn.decomposition.PCA ) и t-SNE ( sklearn.manifold.TSNE ) и визуализированы в виде диаграмм рассеяния

Из графиков стало понятно что данные принадлежат трём группам, этот вывод необходим для начала кластеризации

## Кластеризация данных

Из визуализации видно что для кластреризации подойдёт метод "К средних", используем его реализацию из библиотеки scikit learn - https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

Данные кластеризованы выбранным методом на 3 класса и визуализированны, из графика видно что данные разделены верно