# Level 1 - Task 3: EDA Summary - Iris Dataset

- **The three species are linearly separable** in petal space: Setosa is a clear
  low-value cluster, while Versicolor and Virginica overlap only slightly,
  which is why the iris data is a classic classification benchmark.
- **Petal features separate species better than sepal features**: the
  petal-length-vs-petal-width scatter shows tight, well-separated clusters,
  whereas sepal width strongly overlaps between Versicolor and Virginica.
- **Strong positive correlation** between petal_length and petal_width
  (r ~ 0.96) and between both petal features and sepal_length
  (r ~ 0.87 / 0.82): larger petals come with longer sepals.
- **sepal_width is the odd one out**: it correlates weakly with everything else
  (near zero / slightly negative), indicating it carries little class signal.
- **No missing values and no outliers** were found in the data; all four
  features are continuous and approximately normally distributed per species.
