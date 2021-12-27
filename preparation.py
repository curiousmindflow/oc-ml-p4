# %% [markdown]
# ***
# # 0 Dependencies import

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from math import ceil

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

np.random.seed(0)

# %% [markdown]
# # 1 Configuration

# %%
config = {
    "data_loading_disparate": {
        "loading": True,
        "study": True,
        "merging": True,
        "saving": True
    },
    "overview": {
        "overview_plot": True
    },
    "cleaning": {
        "global": {
            "usability": {
                "id_features_retrieval": True,
                "drop": False
            },
        },
        "categoricals": {
            "overview": {
                "describe": True
            }
        },
        "numericals": {
            "overview": {
                "describe": True,
                "plot": False
            },
            "correlations": {
                "overview_heatmap_plot": True
            },
            "outliers": {
                "analyse": True,
                "remove": False
            }
        },
        "datetimes": {
            "overview": {
                "describe": True,
                "plot": True
            }
        }
    }
}


# %%


def dataframe_distribution_overview(data, figsize=(10, 3)):
    plt.figure(figsize=figsize)

    sns.barplot(x=data.columns, y=data.count())

    plt.title("Number of values per column", size=20)
    plt.xticks(rotation=45, size=16, ha="right")
    plt.yticks(size=16)
    plt.ylabel("Number values", size=16)
    plt.show()

# %%


def index_lth(data, percentage: int):
    percentage = percentage / 100
    less_than = data.count() < data.shape[0] * percentage
    # index_less_than = less_than[less_than == True].index
    index_less_than = less_than[less_than].index
    return index_less_than

# %%


def multi_plot_numerical(
    data,
    features,
    kind="hist",
    n_cols=8,
    figsize=(30, 10),
    wspace=0.35,
    hspace=0.35
):
    feature_nb = len(features)
    n_rows = ceil(feature_nb / n_cols)
    index = 0

    plt.subplots(n_rows, n_cols, figsize=figsize)
    plt.subplots_adjust(wspace=wspace, hspace=hspace)

    for r in range(n_rows):
        for c in range(n_cols):
            if index >= feature_nb:
                break

            plt.subplot(n_rows, n_cols, index+1)

            feature = features[index]

            if kind == "box":
                plot = sns.boxplot(y=data[feature])
            elif kind == "hist":
                plot = sns.histplot(data=data[feature], kde=True)
            else:
                plot = sns.histplot(data=data[feature], kde=True)

            plot.set_xlabel(feature, fontsize=12)
            plot.set_ylabel(None)

            index += 1

    plt.show()

# %%


def create_subplot(dataset, feature, n_rows, index, is_numeric):
    plt.subplot(n_rows, 2, index)
    uniques = dataset[feature].unique()

    if is_numeric:
        plot = sns.boxplot(y=dataset[feature])
    else:
        if uniques.size <= 20 and uniques.size > 0:
            plot = sns.countplot(x=dataset[feature])
            plt.xticks(rotation=45, size=8, ha="right")
        else:
            dist = pd.DataFrame(
                data=[[uniques.size, dataset.shape[0] - uniques.size]],
                columns=["uniques", "not_uniques"])
            plot = sns.barplot(data=dist)

    plot.set_xlabel(None)
    plot.set_ylabel(feature, fontsize=14)


def dataset_diff_analysis(data1, data2, exclude=[], figsize=(15, 200)):
    features = data1.columns.union(data2.columns).difference(exclude)
    n_cols = 2
    n_rows = len(features)
    # col_id = 0
    index = 1

    plt.subplots(n_rows, n_cols, figsize=figsize)
    plt.subplots_adjust(wspace=0.35, hspace=0.5)

    for f in features:
        is_numeric = False
        if f in data1.columns:
            f_type = data1[f].dtype
        else:
            f_type = data2[f].dtype
        if f_type in ["int64", "float64"]:
            is_numeric = True

        if f in data1.columns.values:
            create_subplot(data1, f, n_rows, index, is_numeric)
        if f in data2.columns.values:
            create_subplot(data2, f, n_rows, index+1, is_numeric)

        index += 2

    print(f"features: {features}")
    plt.show()

# %%


def decribe_several(feature, *df):
    data = {}
    index = 1
    for d in df:
        data[feature + "_" + str(index)] = d[feature]
        index += 1
    final_dataframe = pd.DataFrame(data)
    return final_dataframe.describe()

# %%


def head_several(feature, nb, *df):
    data = {}
    index = 1
    for d in df:
        data[feature + "_" + str(index)] = d[feature]
        index += 1
    final_dataframe = pd.DataFrame(data)
    return final_dataframe.head(nb)

# %%


class OutlierProcessor():
    def __init__(self, data, features, lower_trig, upper_trig):
        self.data = data
        self.features = features
        self.lower_trig = lower_trig
        self.upper_trig = upper_trig
        self.__above = 0
        self.__below = 0
        self.__total = 0

    def __print(self):
        print(f"lower_trig: {self.lower_trig}")
        print(f"upper_trig: {self.upper_trig}")
        print(f"below: {self.__below}")
        print(f"above: {self.__above}")
        print(f"total: {self.__total}")

    def analyse(self):
        self.__below = self.data[
            self.data[self.features] < self.lower_trig
            ][self.features].count()
        self.__above = self.data[
            self.data[self.features] > self.upper_trig
            ][self.features].count()
        self.__total = self.__below + self.__above
        self.__print()

    def replace(self, replace_by=np.nan, inplace=False):
        result = self.data.loc[:, self.features].where(
            cond=lambda x: ((x > self.lower_trig) & (self.upper_trig > x)),
            other=replace_by)
        if inplace:
            self.data[self.features] = result
        else:
            return result

# %%


class OutlierIqrProcessor(OutlierProcessor):
    def __init__(self, data, features, exclude=[]):
        self.features = [feature for feature in features
                         if feature not in exclude]
        self.__q1 = data[self.features].quantile(0.25)
        self.__q3 = data[self.features].quantile(0.75)
        self.__iqr = self.__q3 - self.__q1
        upper_trig = self.__q3 + (1.5 * self.__iqr)
        lower_trig = self.__q1 - (1.5 * self.__iqr)
        super().__init__(data, self.features, lower_trig, upper_trig)

# %%


def correlation_heatmap(dataset, figsize=(30, 20)):
    plt.figure(figsize=figsize)

    correlation = dataset.corr()
    mask = np.triu(np.ones_like(correlation, dtype=bool))

    sns.heatmap(data=correlation, mask=mask, annot=True, vmax=.75, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    plt.title("Correlation heatmap", size=20)
    plt.xticks(rotation=45, size=16, ha="right")
    plt.yticks(size=16)
    plt.show()

# %%


def unique_several(dataset, features, take=20):
    uniques_data = {}
    too_many_uniques = []
    only_one_uniques = []
    only_two_uniques = []

    for f in features:
        f_uniques = dataset[f].unique()
        if f_uniques.size <= take:
            if f_uniques.size == 1:
                only_one_uniques.append(f)
            elif f_uniques.size == 2:
                only_two_uniques.append(f)
            else:
                uniques_data[f] = pd.Series(
                    data=f_uniques, name=f, dtype="object")
        else:
            too_many_uniques.append((f, f_uniques.size))

    print(f"Only one unique: {only_one_uniques}")
    print(f"Only two uniques: {only_two_uniques}")
    print(f"Too many uniques: {too_many_uniques}")

    return pd.DataFrame(data=uniques_data)

# %%


def strip_and_lower(dataset, features, to_strip=None, inplace=False):
    result_data = {}
    for f in features:
        result_data[f] = pd.Series(
            data=dataset[f].str.strip(to_strip=to_strip),
            name=f, dtype="object")
        result_data[f] = result_data[f].str.lower()
    result = pd.DataFrame(data=result_data)
    if inplace:
        dataset[features] = result
    else:
        return result

# %%


# def fuzzy_matching_several(dataset, fuzzy, limit=10):
#     fuzzy_data = {}
#     for feature, match in fuzzy:
#         fuzzy_matches = fuzzywuzzy.process.extract(
#             match,
#             dataset[feature],
#             limit=limit,
#             scorer=fuzzywuzzy.fuzz.token_sort_ratio)
#         fuzzy_data[feature] = pd.Series(
#             data=fuzzy_matches, name=feature, dtype="object")
#     return pd.DataFrame(data=fuzzy_data)

# %%


def feature_distribution_multivar(
    data,
    features,
    figsize=(10, 3),
    wspace=0.2,
    hspace=0.2,
    corr_scale=(0.75, 0),
    xlim=(None, None),
    ylim=(None, None)
):
    plt.subplots(1, 2, figsize=figsize)
    plt.subplots_adjust(wspace=wspace, hspace=hspace)

    plt.subplot(1, 2, 1)
    for f in features:
        sns.kdeplot(data=data[f], label=f, shade=True)

    plt.title("Distribution", size=20)
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.xlabel(None)
    plt.ylabel("Density", size=16)
    plt.legend()

    plt.subplot(1, 2, 2)
    correlation = data.loc[:, features].corr()
    mask = np.triu(np.ones_like(correlation, dtype=bool))

    sns.heatmap(
        data=correlation, mask=mask,
        annot=True, vmax=corr_scale[0],
        center=corr_scale[1], square=True,
        linewidths=.5, cbar_kws={"shrink": .5})

    plt.title("Correlation", size=20)
    plt.xticks(rotation=45, size=12, ha="right")
    plt.yticks(rotation=0, size=12, va="center")

    plt.show()

# %%


def get_categorical_features_name(
    dataset,
    split_by_unique_count=True,
    split_count=10
):
    features_name = dataset.select_dtypes(["object", "bool"]).columns
    if split_by_unique_count:
        less_uniques = [feature_name for feature_name in features_name
                        if dataset[feature_name].nunique() <= split_count]
        lot_uniques = features_name.difference(less_uniques).tolist()
        return (less_uniques, lot_uniques)
    else:
        return features_name.values

# %%


def get_numerical_features_name(dataset):
    features_name = dataset.select_dtypes([
        "int64",
        "float64"
        ]).columns.values.tolist()
    return features_name

# %%


def get_datetime_features_name(dataset):
    features_name = dataset.select_dtypes(
        ["datetime64"]
        ).columns.values.tolist()
    return features_name

# %% [markdown]
# ***
# # 2 Data Loading disparate

# %% [markdown]
# ## 2.1 Loading

# %%


if config["data_loading_disparate"]["loading"]:
    customers = pd.read_csv("data/olist_customers_dataset.csv", delimiter=",")
    geoloc = pd.read_csv("data/olist_geolocation_dataset.csv", delimiter=",")
    items = pd.read_csv("data/olist_order_items_dataset.csv", delimiter=",")
    payments = pd.read_csv(
        "data/olist_order_payments_dataset.csv",
        delimiter=",")
    reviews = pd.read_csv(
        "data/olist_order_reviews_dataset.csv",
        delimiter=",")
    orders = pd.read_csv("data/olist_orders_dataset.csv", delimiter=",")
    products = pd.read_csv("data/olist_products_dataset.csv", delimiter=",")
    sellers = pd.read_csv("data/olist_sellers_dataset.csv", delimiter=",")
    category = pd.read_csv(
        "data/product_category_name_translation.csv",
        delimiter=",")

# %% [markdown]
# ***
# ## 2.2 Overview

# %% [markdown]
# ### 2.2.1 Customers

# %%
if config["data_loading_disparate"]["study"]:
    customers.info()

# %%
customers.head()

# %%
if config["data_loading_disparate"]["study"]:
    customers.describe(include="all")

# %%
if config["data_loading_disparate"]["study"]:
    customers.drop_duplicates(subset=["customer_unique_id"], inplace=True)

# %%
if config["data_loading_disparate"]["study"]:
    customers.info()

# %% [markdown]
# ***
# ### 2.2.2 Geolocation

# %%
if config["data_loading_disparate"]["study"]:
    geoloc.info()

# %%
if config["data_loading_disparate"]["study"]:
    geoloc.describe(include="all")

# %%
if config["data_loading_disparate"]["study"]:
    geoloc.drop_duplicates(
        subset=["geolocation_lat", "geolocation_lng"],
        inplace=True)

# %%
if config["data_loading_disparate"]["study"]:
    geoloc.info()

# %% [markdown]
# ***
# ### 2.2.3 Items

# %%
if config["data_loading_disparate"]["study"]:
    items.info()

# %%
items.head()

# %%
if config["data_loading_disparate"]["study"]:
    items.describe(include="all")

# %%
# if config["data_loading_disparate"]["study"]:
#     items.drop_duplicates(subset=["order_id"], inplace=True)

# %%
if config["data_loading_disparate"]["study"]:
    items.info()

# %% [markdown]
# ***
# ### 2.2.4 Payments

# %%
if config["data_loading_disparate"]["study"]:
    payments.info()

# %%
if config["data_loading_disparate"]["study"]:
    payments.describe(include="all")

# %%
if config["data_loading_disparate"]["study"]:
    payments.drop_duplicates(subset=["order_id"], inplace=True)

# %%
if config["data_loading_disparate"]["study"]:
    payments.info()

# %% [markdown]
# ***
# ### 2.2.5 Reviews

# %%
if config["data_loading_disparate"]["study"]:
    reviews.info()

# %%
if config["data_loading_disparate"]["study"]:
    reviews.describe(include="all")

# %%
if config["data_loading_disparate"]["study"]:
    reviews.drop_duplicates(subset=["order_id"], inplace=True)

# %%
if config["data_loading_disparate"]["study"]:
    reviews.info()

# %% [markdown]
# ***
# ### 2.2.6 Orders

# %%
if config["data_loading_disparate"]["study"]:
    orders.info()

# %%
if config["data_loading_disparate"]["study"]:
    orders.describe(include="all")

# %%
# if config["data_loading_disparate"]["study"]:
#     orders.drop_duplicates(subset=["order_id"], inplace=True)

# %%
if config["data_loading_disparate"]["study"]:
    orders.info()

# %% [markdown]
# ***
# ### 2.2.7 Products

# %%
if config["data_loading_disparate"]["study"]:
    products.info()

# %%
if config["data_loading_disparate"]["study"]:
    products.describe(include="all")

# %%
if config["data_loading_disparate"]["study"]:
    products.drop_duplicates(subset=["product_id"], inplace=True)

# %%
if config["data_loading_disparate"]["study"]:
    products.info()

# %% [markdown]
# ***
# ### 2.2.8 Sellers

# %%
if config["data_loading_disparate"]["study"]:
    sellers.info()

# %%
if config["data_loading_disparate"]["study"]:
    sellers.describe(include="all")

# %%
if config["data_loading_disparate"]["study"]:
    sellers.drop_duplicates(subset=["seller_id"], inplace=True)

# %%
if config["data_loading_disparate"]["study"]:
    sellers.info()

# %% [markdown]
# ***
# ### 2.2.9 Category

# %%
if config["data_loading_disparate"]["study"]:
    category.info()

# %%
if config["data_loading_disparate"]["study"]:
    category.describe(include="all")

# %%
if config["data_loading_disparate"]["study"]:
    category.drop_duplicates(inplace=True)

# %%
if config["data_loading_disparate"]["study"]:
    category.info()

# %% [markdown]
# ***
# ## 2.3 Merging

# %%
if config["data_loading_disparate"]["merging"]:
    data = orders.merge(customers, how="left", on="customer_id")
    data = data.merge(reviews, on="order_id")
    data = data.merge(payments, on="order_id")
    data = data.merge(items, on="order_id")
    data = data.merge(products, on="product_id")
    data = data.merge(sellers, on="seller_id")
    data.info()

# %%
if config["data_loading_disparate"]["merging"] and False:
    data = data.merge(
        geoloc,
        left_on="customer_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        suffixes=("_customer", "_customer"))

    data.info()

# %%
if config["data_loading_disparate"]["merging"] and False:
    data = data.merge(
        geoloc,
        left_on="seller_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        suffixes=("_seller", "_seller"))

    data.info()

# %% [markdown]
# ***
# ## 2.4 Saving

# %%
if config["data_loading_disparate"]["saving"]:
    data.to_csv("data/data_merged.csv", sep=",", index=False)

# %% [markdown]
# ***
# # 3 Data loading merged


# %%
data = pd.read_csv("data/data_merged.csv", delimiter=",")

# %%
categ_cols = get_categorical_features_name(data, split_by_unique_count=False)
num_cols = get_numerical_features_name(data)
datetime_cols = get_datetime_features_name(data)

# %% [markdown]
# ***
# # 4 Overview

# %%
if config["overview"]["overview_plot"]:
    dataframe_distribution_overview(data, figsize=(30, 3))

# %% [markdown]
# ***
# # 5 Cleaning

# %% [markdown]
# ## 5.1 Global

# %% [markdown]
# ### 5.1.1 Usability

# %%
if config["cleaning"]["global"]["usability"]["id_features_retrieval"]:
    id_features = data.columns[data.columns.str.endswith("_id")]
    id_features

# %%
if config["cleaning"]["global"]["usability"]["drop"]:
    features_to_delete = id_features
    data.drop(columns=features_to_delete, inplace=True)

# %%
categ_cols = get_categorical_features_name(data, split_by_unique_count=False)
num_cols = get_numerical_features_name(data)

# %% [markdown]
# ***
# ## 5.2 Categoricals

# %% [markdown]
# ### 5.2.1 Overview

# %%
if config["cleaning"]["categoricals"]["overview"]["describe"]:
    data[categ_cols].describe()

# %% [markdown]
# ***
# ### 5.2.2 Inconsistencies

# %%
unique_several(data, data[categ_cols], take=50)

# %% [markdown]
# ***
# ## 5.3 Numericals

# %% [markdown]
# ### 5.3.1 Overview

# %%
if config["cleaning"]["numericals"]["overview"]["describe"]:
    data[num_cols].describe()

# %%
if config["cleaning"]["numericals"]["overview"]["plot"]:
    multi_plot_numerical(
        data,
        num_cols,
        n_cols=5,
        hspace=0.4,
        wspace=0.2,
        figsize=(30, 30))

# %% [markdown]
# ***
# ### 5.3.1 Usability

# %%
(data[num_cols] <= 0).any()

# %%
data[num_cols].shape

# %%
data[num_cols] = data[num_cols].where(data[num_cols] > 0, np.nan)

# %%
data[num_cols].shape

# %%
(data[num_cols] <= 0).any()

# %% [markdown]
# ***
# ### 5.3.3 Correlations

# %%
if config["cleaning"]["numericals"]["correlations"]["overview_heatmap_plot"]:
    correlation_heatmap(data[num_cols], figsize=(15, 15))

# %% [markdown]
# ***
# ### 5.3.4 Outliers

# %%
if config["cleaning"]["numericals"]["outliers"]["analyse"]:
    outlier_proc = OutlierIqrProcessor(
        data,
        data[num_cols],
        exclude=[
            "customer_zip_code_prefix",
            "review_score",
            "payment_sequential",
            "payment_installments"])
    outlier_proc.analyse()

# %%
if config["cleaning"]["numericals"]["outliers"][
    "analyse"
    ] and config[
        "cleaning"
        ]["numericals"]["outliers"]["remove"]:
    outlier_proc.replace(inplace=True)

# %% [markdown]
# ***
# ## 5.4 Datetimes

# %% [markdown]
# ### 5.4.1 Overview

# %%
if config["cleaning"]["datetimes"]["overview"]["describe"]:
    pass
# data[datetime_cols].describe(datetime_is_numeric=True)

# %%
for col in datetime_cols:
    data[col] = data[col].astype("object")
    data[col] = pd.to_datetime(data[col], format="%Y-%m-%d %H:%M:%S")

# %%
if config["cleaning"]["datetimes"]["overview"]["describe"]:
    pass
# data[datetime_cols].describe(datetime_is_numeric=True)

# %%
if config["cleaning"]["datetimes"]["overview"]["plot"]:
    pass

# %% [markdown]
# ***
# # 6 Saving

# %%
data.to_csv("data/data_cleaned.csv", sep=",", index=False)
