# %% [markdown]
# # 0 Configuration

# %%
config = {
    "overview": {

    },
    "fe": {
        "id": False,
        "volume": False,
        "order_purchase_timestamp": False,
        "shipping_limit_date": False,
        "order_estimated_delivery_date": False,
        "price_agg": False,
        "last_order_datetime": False,
        "frequency": False
    },
    "model": {
        "rfm_raw": {
            "preparation": True,
            "n_cluster": True,
            "silhouette": False,
            "explain": True,
            "persona": True
        },

    },
    "geo": {
        "preproc": True,
        "arange": True,
        "all": True,
        "per_cluster": True
    },
    "global": {
        "do": True,
        "elbow": True,
        "explain_plot": True,
        "features_plot": True,
        "cluster_plot": True
    },
    "maintenance": {
        "do": True
    }
}

# %% [markdown]
# ***
# # 1 Dependencies import

# %% [markdown]
# ## 1.0 Libraries

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from yellowbrick.cluster import SilhouetteVisualizer, KElbowVisualizer

from math import ceil

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

np.random.seed(0)

# %% [markdown]
# ***
# ## 1.1 Features selection

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
        return features_name.values.tolist()

# %%


def get_numerical_features_name(dataset):
    features_name = dataset.select_dtypes(
        ["int64", "float64"]
        ).columns.values.tolist()
    return features_name

# %%


def progressiveFeatureSelection(df, n_clusters=3, max_features=4,):
    feature_list = list(df.columns)
    selected_features = list()
    # select starting feature
    initial_feature = ""
    high_score = 0
    for feature in feature_list:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        data_ = df[feature]
        labels = kmeans.fit_predict(data_.to_frame())
        score_ = silhouette_score(data_.to_frame(), labels)
        print("Proposed new feature {} with score {}". format(feature, score_))
        if score_ >= high_score:
            initial_feature = feature
            high_score = score_
    print("The initial feature is {} with a silhouette score of {}."
          .format(initial_feature, high_score))
    feature_list.remove(initial_feature)
    selected_features.append(initial_feature)
    for _ in range(max_features-1):
        high_score = 0
        selected_feature = ""
        print("Starting selection {}...".format(_))
        for feature in feature_list:
            selection_ = selected_features.copy()
            selection_.append(feature)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            data_ = df[selection_]
            labels = kmeans.fit_predict(data_)
            score_ = silhouette_score(data_, labels)
            print("Proposed new feature {} with score {}"
                  .format(feature, score_))
            if score_ > high_score:
                selected_feature = feature
                high_score = score_
        selected_features.append(selected_feature)
        feature_list.remove(selected_feature)
        print("Selected new feature {} with score {}".
              format(selected_feature, high_score))
    return selected_features

# %% [markdown]
# ***
# ## 1.2 Pipeline construction

# %%


def evaluate(dataset, model, scoring="neg_root_mean_squared_error"):
    numerical_cols = get_numerical_features_name(dataset)

    X = dataset[numerical_cols]

    num_pipe = Pipeline(steps=[
        ("simple_imputer", SimpleImputer(strategy="mean")),
        ("minmax_scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ("num_pipe", num_pipe, numerical_cols),
    ])

    pipeline = Pipeline([
        ("transforms", preprocessor),
        ("model", model)
    ])

    preprocessor.fit(X)
    array_preproc = preprocessor.transform(X)
    data_preproc = pd.DataFrame(data=array_preproc, columns=X.columns)

    return pipeline, preprocessor, data_preproc

# %%


def pca_preprocessor(dataset):
    num_cols = get_numerical_features_name(dataset)
    categ_cols, _ = get_categorical_features_name(dataset)

    num_pipe = Pipeline(steps=[
        ("imputer_01", SimpleImputer(strategy="mean")),
        ("scaler_01", StandardScaler())
    ])

    categ_pipe = Pipeline(steps=[
        ("imputer_01", SimpleImputer(strategy="most_frequent")),
        ("encoder_01", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num_pipe", num_pipe, num_cols),
        ("categ_pipe", categ_pipe, categ_cols)
    ])

    return preprocessor, num_cols, categ_cols

# %%


def preprocess_data(dataset):
    preprocessor, num_cols, categ_cols = pca_preprocessor(dataset)

    raw_preprocessed_data = preprocessor.fit_transform(dataset)

    if categ_cols:
        categ_cols_preprocessed = preprocessor.transformers_[1][1][
            "encoder_01"
            ].get_feature_names_out(categ_cols).tolist()
    else:
        categ_cols_preprocessed = []

    col_names = num_cols + categ_cols_preprocessed

    preprocessed_data = pd.DataFrame(
        data=raw_preprocessed_data,
        columns=col_names)

    return preprocessed_data

# %% [markdown]
# ***
# ## 1.3 Cluster evaluation

# %%


def elbow_plot(range, data, figsize=(10, 10)):
    intertia_list = []
    for n in range:
        kmeans = KMeans(n_clusters=n, random_state=1)
        kmeans.fit(data)
        intertia_list.append(kmeans.inertia_)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    sns.lineplot(y=intertia_list, x=range, ax=ax)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Inertia")
    ax.set_xticks(list(range))
    fig.show()

# %%


def silhouette_plot(
    range, data,
    n_cols=2,
    figsize=(20, 10),
    wspace=0.35,
    hspace=0.35
):
    if range[0] == 0:
        raise Exception("Range must not include 0")

    n_rows = ceil(len(range)/n_cols)
    index = 0

    plt.subplots(n_rows, n_cols, figsize=figsize)
    plt.subplots_adjust(wspace=wspace, hspace=hspace)

    for n in range:
        kmeans = KMeans(n, random_state=1)

        plt.subplot(n_rows, n_cols, index+1)
        sv = SilhouetteVisualizer(kmeans, colors="yellowbrick")
        sv.fit(data)

        index += 1
    plt.show()
# %%


def display_clusters(dataset_preproc, n_clusters):
    pca, pca_data, loadings = apply_pca(
        dataset_preproc,
        dataset_preproc.columns,
        2)
    reduced_data = pca_data.to_numpy()
    model = KMeans(init="k-means++", n_clusters=n_clusters)
    model.fit(reduced_data)

    h = 0.02

    x_min = reduced_data[:, 0].min() - 1
    x_max = reduced_data[:, 0].max() + 1

    y_min = reduced_data[:, 1].min() - 1
    y_max = reduced_data[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(
        Z, interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=plt.cm.Paired,
        aspect="auto",
        origin="lower"
    )

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)

    centroids = model.cluster_centers_
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="x",
        s=169,
        linewidths=3,
        color="w",
        zorder=10
    )

    plt.title("K-means clustering, PCA reduced dataset")

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()

# %% [markdown]
# ***
# ## 1.4 Dimensionality reduction

# %%


def scree_plot(dataset, figsize=(15, 5)):
    pca = PCA()
    pca.fit(dataset)

    plt.figure(figsize=figsize)
    explain_variance = pd.Series(pca.explained_variance_ratio_)
    explain_variance.plot(kind="bar", alpha=0.7)

    total = 0
    var_ls = []
    for x in explain_variance:
        total = total + x
        var_ls.append(total)

    pd.Series(var_ls).plot(marker="o", alpha=0.7)
    plt.xlabel("Principle Components", fontsize="x-large")
    plt.ylabel("Percentage Variance Explained", fontsize="x-large")
    plt.title("Scree plot", fontsize="xx-large")
    plt.show()

    return pca

# %%


def apply_pca(dataset, columns, n_components):
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(dataset[columns])
    components_name = [f"PC{i+1}" for i in range(pca_data.shape[1])]
    pca_data = pd.DataFrame(data=pca_data, columns=components_name)
    loadings = pd.DataFrame(
        data=pca.components_.T,
        columns=components_name,
        index=columns)
    return pca, pca_data, loadings

# %%


def plot_variance(pca, width=8, dpi=100):
    fig, axs = plt.subplots(1, 2)
    n = pca.n_components_
    grid = np.arange(1, n + 1)
    evr = pca.explained_variance_ratio_

    axs[0].bar(grid, evr)
    axs[0].set(
        xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0)
    )

    cumulative_variance = np.cumsum(evr)
    axs[1].plot(np.r_[0, grid], np.r_[0, cumulative_variance], "o-")
    axs[1].set(
        xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0)
    )

    fig.set(figwidth=8, dpi=100)
    return axs

# %% [markdown]
# ***
# ## 1.5 RFM

# %%


def rfm_distplot(dataset, customer_id, figsize=(20, 5)):
    warnings.filterwarnings('ignore')
    dataset = dataset.loc[:, dataset.columns.difference(customer_id)]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for i, feature in enumerate(dataset.columns):
        sns.distplot(dataset[feature], ax=axes[i])

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


def elbow(dataset, k=(2, 12)):
    model = KMeans(random_state=1)
    elbow_visualizer = KElbowVisualizer(model, k=k)

    elbow_visualizer.fit(dataset)
    elbow_visualizer.poof()

# %%


def silhouette(dataset, n):
    model = KMeans(n, random_state=1)
    silhouette_visualizer = SilhouetteVisualizer(model)

    silhouette_visualizer.fit(dataset)
    silhouette_visualizer.poof()

# %%


def explain(dataset, dataset_preproc):
    model = KMeans(n_clusters=4, random_state=1).fit(dataset_preproc)
    data_explain = dataset
    data_explain["Cluster"] = model.labels_

    data_explain_melt = pd.melt(
        data_explain,
        id_vars=["customer_id", "Cluster"],
        value_vars=["Récence", "Fréquence", "Montant"],
        var_name="Features",
        value_name="Value")
    sns.lineplot("Features", "Value", hue="Cluster", data=data_explain_melt)
    plt.legend()

    return data_explain.groupby("Cluster").agg({
        "Récence": ["mean", "min", "max"],
        "Fréquence": ["mean", "min", "max"],
        "Montant": ["mean", "min", "max", "count"]
    })

# %% [markdown]
# ***
# ## 1.6 Explainability

# %%


def explain_global(dataset, dataset_preproc, clusters, figsize=(30, 10)):
    model = KMeans(n_clusters=clusters, random_state=1).fit(dataset_preproc)
    data_explain = dataset_preproc.copy()
    data_explain.loc[:, "customer_id"] = dataset.loc[:, "customer_id"]
    data_explain.loc[:, "Cluster"] = model.labels_

    data_explain_melt = pd.melt(
        data_explain,
        id_vars=["customer_id", "Cluster"],
        value_vars=dataset_preproc.columns,
        var_name="Features",
        value_name="Value")

    plt.figure(figsize=figsize)
    sns.lineplot(
        x="Features",
        y="Value",
        hue="Cluster",
        data=data_explain_melt,
        palette="deep")
    plt.legend()

    ops = {}
    for col in dataset_preproc.columns:
        ops[col] = ["mean", "min", "max", "count"]

    return data_explain.groupby("Cluster").agg(ops)

# %% [markdown]
# ***
# # 2 Data loading

# %%


data = pd.read_csv("data/data_cleaned.csv", delimiter=",")

# %% [markdown]
# ***
# # 3 Overview

# %%
data.info()

# %%
data.describe()

# %%
data.describe(include="object")

# %% [markdown]
# ***
# # 4 Feeture engineering

# %% [markdown]
# ## 4.0 Remove *_id

# %%


def fe_id_remove(dataset, diff=[]):
    id_cols = dataset.columns[dataset.columns.str.contains("_id")]
    id_cols = id_cols.difference(diff)
    dataset.drop(columns=id_cols, inplace=True)
    return dataset

# %% [markdown]
# ***
# ## 4.1 Volume

# %%


def fe_volume(dataset):
    dataset["product_volume_cm3"] = dataset[
                                        "product_length_cm"
                                    ] * dataset[
                                        "product_height_cm"
                                    ] * dataset[
                                        "product_width_cm"
                                    ]
    return dataset

# %% [markdown]
# ***
# ## 4.2 Order purchase timestamp

# %%


def fe_purchase_timestamp(dataset):
    dataset.loc[:, "order_purchase_timestamp"] = dataset.loc[
        :,
        "order_purchase_timestamp"
        ].apply(pd.to_datetime)
    opt = dataset.loc[:, "order_purchase_timestamp"]
    dataset["order_purchase_year"] = opt.dt.year
    dataset["order_purchase_month"] = opt.dt.month
    dataset["order_purchase_day"] = opt.dt.day
    dataset["order_purchase_hour"] = opt.dt.hour
    return dataset

# %% [markdown]
# ***
# ## 4.3 Shipping limit date

# %%


def fe_shipping_date(dataset):
    dataset.loc[:, "shipping_limit_date"] = dataset.loc[
        :,
        "shipping_limit_date"
        ].apply(pd.to_datetime)
    sld = dataset.loc[:, "shipping_limit_date"]
    dataset["shipping_limit_year"] = sld.dt.year
    dataset["shipping_limit_month"] = sld.dt.month
    dataset["shipping_limit_day"] = sld.dt.day
    dataset["shipping_limit_hour"] = sld.dt.hour

    return dataset

# %% [markdown]
# ***
# ## 4.4 Estimated delivery date

# %%


def fe_delivery_date(dataset):
    dataset.loc[:, "order_estimated_delivery_date"] = dataset.loc[
        :,
        "order_estimated_delivery_date"
        ].apply(pd.to_datetime)
    oedd = dataset.loc[:, "order_estimated_delivery_date"]
    dataset["order_estimated_delivery_year"] = oedd.dt.year
    dataset["order_estimated_delivery_month"] = oedd.dt.month
    dataset["order_estimated_delivery_day"] = oedd.dt.day
    dataset["order_estimated_delivery_hour"] = oedd.dt.hour

    return dataset

# %% [markdown]
# ***
# ## 4.5 Price agg

# %%


def fe_price(dataset):
    customer_price_agg = dataset.groupby("customer_id").agg({
        "price": ["min", "max", "mean", "sum"]
    })

    customer_price_agg.rename(columns={
        "min": "price_min",
        "max": "price_max",
        "mean": "price_mean",
        "sum": "price_sum"
    }, inplace=True)

    dataset = dataset.merge(customer_price_agg.price, on="customer_id")
    return dataset


# %% [markdown]
# ***
# ## 4.6 Recency

# %%


def fe_recency(dataset):
    dataset.loc[:, "order_purchase_timestamp"] = dataset.loc[
        :,
        "order_purchase_timestamp"
        ].apply(pd.to_datetime)
    data_end = max(dataset.loc[
        :,
        "order_purchase_timestamp"] + pd.Timedelta(days=1))

    opt_agg = dataset.groupby("customer_id").agg({
        "order_purchase_timestamp": lambda x: (data_end - max(x)).days
    })

    opt_agg.columns = ["recency"]
    opt_agg = opt_agg.reset_index()

    dataset = dataset.merge(opt_agg, on="customer_id")

    return dataset

# %% [markdown]
# ***
# ## 4.7 Frequency

# %%


def fe_frequency(dataset):
    frequency = dataset.groupby("customer_id").agg({
        "order_id": ["count"]
    })

    frequency = frequency.order_id.reset_index()
    frequency.columns = ["customer_id", "frequency"]

    dataset = dataset.merge(frequency, how="left", on="customer_id")
    return dataset

# %% [markdown]
# ***
# # 5 Modelisation

# %% [markdown]
# ## 5.1 RFM

# %% [markdown]
# ### 5.1.1 Preparation

# %% [markdown]
# RFM: Récence (dat de la dernière commande), Fréquence (des commandes),
# Montant (de la dernière commande ou sur une période donnée)

# %%


if config["model"]["rfm_raw"]["preparation"]:
    rfm_cols = ["customer_id", "order_purchase_timestamp", "order_id", "price"]
    data_rfm = data.loc[:, rfm_cols]

# %%
if config["model"]["rfm_raw"]["preparation"]:
    data_rfm = fe_id_remove(
        data_rfm,
        diff=["customer_id", "order_id", "product_id"])
    data_rfm = fe_recency(data_rfm)
    data_rfm = fe_price(data_rfm)
    data_rfm = fe_frequency(data_rfm)

# %%
if config["model"]["rfm_raw"]["preparation"]:
    data_rfm.drop(columns=[
        "price", "price_min",
        "price_max", "price_mean",
        "order_purchase_timestamp", "order_id"
        ], inplace=True)
    data_rfm.rename(columns={
        "recency": "Récence",
        "price_sum": "Montant",
        "frequency": "Fréquence"
        }, inplace=True)

# %%
display = None
if config["model"]["rfm_raw"]["preparation"]:
    display = data_rfm.describe()
display

# %%
if config["model"]["rfm_raw"]["preparation"]:
    rfm_distplot(data_rfm, ["customer_id"])

# %%
if config["model"]["rfm_raw"]["preparation"]:
    correlation_heatmap(data_rfm, figsize=(20, 5))

# %%
if config["model"]["rfm_raw"]["preparation"]:

    model = KMeans(n_clusters=4)

    pipeline, preproc, data_preproc = evaluate(data_rfm, model)

# %%
display = None
if config["model"]["rfm_raw"]["preparation"]:
    display = data_preproc.head()
display

# %%
if config["model"]["rfm_raw"]["preparation"]:
    rfm_distplot(data_preproc, ["customer_id"])

# %% [markdown]
# ***
# ### 5.1.2 Cluster N analysis

# %%
if config["model"]["rfm_raw"]["n_cluster"]:
    elbow(data_preproc.copy(), (2, 12))

# %%
if config["model"]["rfm_raw"]["silhouette"]:
    silhouette(data_preproc.copy(), 4)

# %% [markdown]
# Silhouette for n=4
# ![alt text](plots/kmeans_4n.png "Silhouette for n=4")

# %% [markdown]
# ***
# ### 5.1.3 Explaination

# %%
display = None
if config["model"]["rfm_raw"]["explain"]:
    display = explain(data_rfm, data_preproc)
display

# %%
if config["model"]["rfm_raw"]["explain"]:
    pred = KMeans(n_clusters=4, random_state=1).fit_predict(data_preproc)

# %%
if config["model"]["rfm_raw"]["explain"]:
    plt.subplots(3, 3, figsize=(30, 15))
    plt.subplots_adjust(wspace=0.15, hspace=0.15)

    plt.subplot(3, 3, 1)
    sns.scatterplot(
        x=data_rfm["Récence"],
        y=data_rfm["Récence"], hue=pred, palette="deep")

    plt.subplot(3, 3, 2)
    sns.scatterplot(
        x=data_rfm["Récence"],
        y=data_rfm["Fréquence"], hue=pred, palette="deep")

    plt.subplot(3, 3, 3)
    sns.scatterplot(
        x=data_rfm["Récence"],
        y=data_rfm["Montant"], hue=pred, palette="deep")

    plt.subplot(3, 3, 4)
    sns.scatterplot(
        x=data_rfm["Fréquence"],
        y=data_rfm["Récence"], hue=pred, palette="deep")

    plt.subplot(3, 3, 5)
    sns.scatterplot(
        x=data_rfm["Fréquence"],
        y=data_rfm["Fréquence"], hue=pred, palette="deep")

    plt.subplot(3, 3, 6)
    sns.scatterplot(
        x=data_rfm["Fréquence"],
        y=data_rfm["Montant"], hue=pred, palette="deep")

    plt.subplot(3, 3, 7)
    sns.scatterplot(
        x=data_rfm["Montant"],
        y=data_rfm["Récence"], hue=pred, palette="deep")

    plt.subplot(3, 3, 8)
    sns.scatterplot(
        x=data_rfm["Montant"],
        y=data_rfm["Fréquence"], hue=pred, palette="deep")

    plt.subplot(3, 3, 9)
    sns.scatterplot(
        x=data_rfm["Montant"],
        y=data_rfm["Montant"], hue=pred, palette="deep")

# %%
if config["model"]["rfm_raw"]["explain"]:
    display_clusters(data_preproc, 4)

# %% [markdown]
# ***
# ### 5.1.4 Persona

# %%
if config["model"]["rfm_raw"]["persona"]:
    display

# %%
if config["model"]["rfm_raw"]["persona"]:
    data_preproc["cluster"] = pred
    data_preproc["order_purchase_timestamp"] = data["order_purchase_timestamp"]
    data_preproc = fe_purchase_timestamp(data_preproc)

# %%
if config["model"]["rfm_raw"]["persona"]:
    data_preproc.head()

# %%
if config["model"]["rfm_raw"]["persona"]:
    plt.figure(figsize=(20, 10))
    sns.kdeplot(
        data=data_preproc,
        x="order_purchase_month", hue="cluster", shade=False, palette="deep")

# %%
if config["model"]["rfm_raw"]["persona"]:
    data_preproc["product_category_name"] = data["product_category_name"]

# %%
if config["model"]["rfm_raw"]["persona"]:
    data_preproc.head()

# %%
if config["model"]["rfm_raw"]["persona"]:
    tmp = data_preproc.groupby("cluster").agg({
        "product_category_name": "value_counts"
    })

# %%
if config["model"]["rfm_raw"]["persona"]:
    tmp.info()

# %%
if config["model"]["rfm_raw"]["persona"]:
    tmp.head()

# %%


# %%
if config["model"]["rfm_raw"]["persona"]:
    plt.figure(figsize=(30, 10))
    sns.countplot(
        data=data_preproc,
        x="product_category_name", hue="cluster", palette="deep")

# %% [markdown]
# #### Cluster 0 - Persona 1 - Anciens
#
# Dernier achat: il y a plus de 1 an<br>
# Nombre d'achats: 1.2<br>
# Montant: 122€<br>
# Population: 45 156 / 40.42%<br>
#
# Actions:<br>
# ...
#
#
# #### Cluster 1 - Persona 2 - Fidèle
#
# Dernier achat: il y a 8 mois<br>
# Nombre d'achats: 5.4<br>
# Montant: 358€<br>
# Population: 4 659 / 4.17%<br>
#
#
# #### Cluster 2 - Persona 3 - Nouveaux
#
# Dernier achat: il y a 4 mois<br>
# Nombre d'achats: 1.2<br>
# Montant: 122€<br>
# Population: 60 284 / 53.96%<br>
#
#
# #### Cluster 3 - Persona 4 - Fort potentiel
#
# Dernier achat: il y a 8 mois<br>
# Nombre d'achats: 1.7<br>
# Montant: 819€<br>
# Population: 1 606 / 1.43%<br>
#
# Actions:<br>
# Client à fort potentiel. Mise en place d'une action de
# fidélisation nécessaire<br>

# %% [markdown]
# ***
# ## 5.2 Global

# %%
data.head()

# %%
display = None

if config["global"]["do"]:
    global_cols = data.select_dtypes(["int64", "float64"]).columns
    global_cols = global_cols.append(data.loc[
            :,
            [
                "customer_id", "order_id",
                "order_purchase_timestamp", "shipping_limit_date",
                "order_estimated_delivery_date"
            ]].columns)

    global_data = data[global_cols]

    global_data = fe_volume(global_data)
    global_data = fe_price(global_data)
    global_data = fe_frequency(global_data)
    global_data = fe_recency(global_data)
    global_data = fe_shipping_date(global_data)
    global_data = fe_delivery_date(global_data)

    global_data_preproc = preprocess_data(global_data)

    display = global_data_preproc.head()

display

# %%
if config["global"]["do"]:
    pca = scree_plot(global_data_preproc)

# %%
if config["global"]["elbow"]:
    elbow(global_data_preproc)

# %%
display = None
if config["global"]["explain_plot"]:
    explain_data = explain_global(global_data, global_data_preproc, 6)
    display = explain_data
display

# %%
if config["global"]["do"]:
    global_data_preproc.drop(
        columns=[
            "customer_zip_code_prefix", "review_score",
            "payment_installments", "payment_value",
            "order_item_id", "product_name_lenght",
            "product_description_lenght", "product_photos_qty",
            "product_length_cm", "product_height_cm",
            "product_width_cm", "seller_zip_code_prefix",
            "shipping_limit_year", "shipping_limit_month",
            "shipping_limit_day", "shipping_limit_hour",
            "order_estimated_delivery_year", "order_estimated_delivery_month",
            "order_estimated_delivery_day", "order_estimated_delivery_hour",
            "payment_sequential", "price_min", "price_max", "price_mean"
            ], inplace=True)

# %%
if config["global"]["do"]:
    pca = scree_plot(global_data_preproc)

# %%
if config["global"]["elbow"]:
    elbow(global_data_preproc)

# %%
display = None
if config["global"]["explain_plot"]:
    explain_data = explain_global(global_data, global_data_preproc, 12)
    display = explain_data
display

# %%
display = None
if config["global"]["explain_plot"]:
    explain_data = explain_global(global_data, global_data_preproc, 10)
    display = explain_data
display

# %%
display = None
if config["global"]["explain_plot"]:
    explain_data = explain_global(global_data, global_data_preproc, 8)
    display = explain_data
display

# %%
display = None
if config["global"]["explain_plot"]:
    explain_data = explain_global(global_data, global_data_preproc, 6)
    display = explain_data
display

# %%
if config["global"]["do"]:
    pred = KMeans(
        n_clusters=6,
        random_state=1
        ).fit_predict(global_data_preproc)

# %% [markdown]
# Cluster description
#
# |Cluster:|0|1|2|3|4|5|
# |--------|-|-|-|-|-|-|
# |price:|<span class="ml">medium-low</span>|<span class="l">low</span>|
# <span class="h">high</span>|<span class="l">low</span>|
# <span class="l">low</span>|<span class="m">medium</span>|
# |freight_value:|<span class="m">medium</span>|<span class="l">low</span>|
# <span class="m">medium</span>|<span class="l">low</span>|
# <span class="l">low</span>|<span class="h">high</span>|
# |weight:|<span class="mh">medium-high</span>|<span class="l">low</span>|
# <span class="ml">medium-low</span>|<span class="l">low</span>|
# <span class="l">low</span>|<span class="h">high</span>|
# |volume:|<span class="mh">medium-high</span>|<span class="l">low</span>|
# <span class="ml">medium-low</span>|<span class="l">low</span>|
# <span class="l">low</span>|<span class="h">high</span>|
# |price_sum:|<span class="ml">medium-low</span>|<span class="l">low</span>|
# <span class="h">high</span>|<span class="l">low</span>|
# <span class="ml">medium-low</span>|<span class="m">medium</span>|
# |frequency:|<span class="l">low</span>|<span class="l">low</span>|
# <span class="l">low</span>|<span class="l">low</span>|
# <span class="h">high</span>|<span class="l">low</span>|
# |recency:|<span class="m">medium</span>|<span class="l">low</span>|
# <span class="m">medium</span>|<span class="h">high</span>|
# <span class="m">medium</span>|<span class="m">medium</span>|
#
# <style>
#     .l {
#         color: red;
#     }
#
#     .ml {
#         color: orange;
#     }
#
#     .m {
#         color: yellow
#     }
#
#     .mh {
#         color: teal
#     }
#
#     .h {
#         color: green
#     }
# </style>

# %% [markdown]
# We can deduce:<br>
#
# Clusters 1 and 3 -> Losts<br>
# Are very similar and are the leasts interesting clusters.<br>
# Maybe low purchasing power customer ?
#
# Cluster 4 -> Loyals<br>
# Represent the clients who bought the more often.<br>
# Middle class customer ?<br>
# From town ?<br>
#
# Cluster 5 -> Big articles<br>
# Represents customers who bought heavy,
#  big articles from far away sellers.<br>
# Maybe for particular occasion, like a relocation ?<br>
# Maybe customer from the countryside ?
#
# Cluster 2 -> Best<br>
# Represents customers who ave spent the most.<br>
# The bought articles are small, light, most certainyl commonly bought.<br>
# Big purchasing power ?<br>
# From city ?<br>
#
# Cluster 0 -> Potential<br>
# Represents customers who ave not spent much but on <br>
# bigger and heavier articlesthan average with average freight<br>
# so most commonly boughts articles.<br>
# Maybe a lak of peticular articles ?

# %%
if config["global"]["features_plot"]:
    global_data["frequency"]

    nb_rows = 4

    plt.subplots(nb_rows, 3, figsize=(30, 15))
    plt.subplots_adjust(wspace=0.15, hspace=0.25)

    plt.subplot(nb_rows, 3, 1)
    sns.scatterplot(
        x=global_data["price"],
        y=global_data["price"], hue=pred, palette="deep")

    plt.subplot(nb_rows, 3, 2)
    sns.scatterplot(
        x=global_data["price"],
        y=global_data["frequency"], hue=pred, palette="deep")

    plt.subplot(nb_rows, 3, 3)
    sns.scatterplot(
        x=global_data["price"],
        y=global_data["recency"], hue=pred, palette="deep")

    plt.subplot(nb_rows, 3, 4)
    sns.scatterplot(
        x=global_data["frequency"],
        y=global_data["price"], hue=pred, palette="deep")

    plt.subplot(nb_rows, 3, 5)
    sns.scatterplot(
        x=global_data["frequency"],
        y=global_data["frequency"], hue=pred, palette="deep")

    plt.subplot(nb_rows, 3, 6)
    sns.scatterplot(
        x=global_data["frequency"],
        y=global_data["recency"], hue=pred, palette="deep")

    plt.subplot(nb_rows, 3, 7)
    sns.scatterplot(
        x=global_data["recency"],
        y=global_data["price"], hue=pred, palette="deep")

    plt.subplot(nb_rows, 3, 8)
    sns.scatterplot(
        x=global_data["recency"],
        y=global_data["frequency"], hue=pred, palette="deep")

    plt.subplot(nb_rows, 3, 9)
    sns.scatterplot(
        x=global_data["recency"],
        y=global_data["recency"], hue=pred, palette="deep")

    plt.subplot(nb_rows, 3, 10)
    sns.scatterplot(
        x=global_data["frequency"],
        y=global_data["product_volume_cm3"], hue=pred, palette="deep")

    plt.subplot(nb_rows, 3, 11)
    sns.scatterplot(
        x=global_data["frequency"],
        y=global_data["freight_value"], hue=pred, palette="deep")

# %%
if config["global"]["cluster_plot"]:
    display_clusters(global_data_preproc, 6)

# %% [markdown]
# ***
# ## 5.3 Geospatial analysis

# %% [markdown]
# ### 5.3.1 Geoloc dataset preprocessing

# %%


import geopandas as gpd

# %%
re_write = False
if config["geo"]["preproc"]:
    geo_data = gpd.read_file("data/olist_geolocation_dataset.csv")

    geo_data["geolocation_lat"] = geo_data["geolocation_lat"].astype("float64")
    geo_data["geolocation_lng"] = geo_data["geolocation_lng"].astype("float64")

    geo_data["geolocation_lat"] = geo_data["geolocation_lat"].where(
        lambda x: x < 10, np.nan
        )
    geo_data.dropna(subset=["geolocation_lat"], inplace=True)

    geo_data = gpd.GeoDataFrame(
        geo_data,
        geometry=gpd.points_from_xy(
            geo_data["geolocation_lng"],
            geo_data["geolocation_lat"]))
    geo_data.to_file("data/olist_geolocation_dataset.json", driver="GeoJSON")

# %% [markdown]
# ***
# ### 5.3.2 Base geoloc arrangment

# %%
if config["geo"]["arange"]:
    geo_data = gpd.read_file("data/olist_geolocation_dataset.json")

# %%
if config["geo"]["arange"]:
    geo_data.info()

# %%
if config["geo"]["arange"]:
    geo_data["geolocation_zip_code_prefix"] = geo_data[
        "geolocation_zip_code_prefix"
        ].astype("int64")

# %%
if config["geo"]["arange"]:
    geo_data.head()

# %%
if config["geo"]["arange"]:
    brazil_geo = gpd.read_file("data/brazil_geo.json")

# %%
if config["geo"]["arange"]:
    ax = brazil_geo.plot(
        figsize=(15, 15),
        color="none",
        edgecolor="gainsboro",
        zorder=3
        )
    geo_data.plot(color="blue", markersize=1, ax=ax)

# %% [markdown]
# ***
# ### 5.3.2 Clustered geoloc points

# %%
if config["geo"]["all"]:
    global_data.loc[:, "Cluster"] = pred

# %%
if config["geo"]["all"]:
    global_data.info()

# %%
if config["geo"]["all"]:
    global_data.dropna(inplace=True)

# %%
if config["geo"]["all"]:
    global_data.loc[:, "customer_zip_code_prefix"] = global_data[
        "customer_zip_code_prefix"
        ].astype("int64")

# %%
if config["geo"]["all"]:
    global_data.head()

# %%
if config["geo"]["all"]:
    global_data["customer_zip_code_prefix"].value_counts().count()

# %%
if config["geo"]["all"]:
    by_customer_zip_code_prefix = global_data.groupby("Cluster")

# %%
if config["geo"]["all"]:
    by_customer_zip_code_prefix.head()

# %%
if config["geo"]["all"]:
    global_data_cleaned = global_data.loc[
        :,
        [
            "customer_zip_code_prefix", "seller_zip_code_prefix",
            "customer_id", "Cluster"
        ]]
    geo_merged = geo_data.merge(
        global_data_cleaned,
        left_on="geolocation_zip_code_prefix",
        right_on="customer_zip_code_prefix")

# %%
if config["geo"]["all"]:
    geo_merged.head()

# %%
if config["geo"]["all"]:
    geo_merged.info()

# %%
if config["geo"]["all"]:
    geo_merged.drop_duplicates(
        subset=["geolocation_lng", "geolocation_lat"], inplace=True)

# %%
if config["geo"]["all"]:
    ax = brazil_geo.plot(
        figsize=(15, 15), color="none", edgecolor="gainsboro", zorder=3)
    kws = {"s": 10, "facecolor": "none", "linewidth": 0.2}
    sns.scatterplot(
        data=geo_merged,
        x="geolocation_lng",
        y="geolocation_lat", ax=ax, hue="Cluster", palette="deep", **kws)

# %% [markdown]
# ***
# ### 5.3.3 Per cluster visualization

# %%
if config["geo"]["per_cluster"]:
    cluster_0 = geo_merged[geo_merged["Cluster"] == 0]
    cluster_1 = geo_merged[geo_merged["Cluster"] == 1]
    cluster_2 = geo_merged[geo_merged["Cluster"] == 2]
    cluster_3 = geo_merged[geo_merged["Cluster"] == 3]
    cluster_4 = geo_merged[geo_merged["Cluster"] == 4]
    cluster_5 = geo_merged[geo_merged["Cluster"] == 5]

    cluster_1_3 = geo_merged[(
        geo_merged["Cluster"] == 1) | (geo_merged["Cluster"] == 3)]

# %%
if config["geo"]["per_cluster"]:
    plt.subplots(3, 2, figsize=(30, 30))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    ax = plt.subplot(3, 2, 1)
    brazil_geo.plot(color="none", edgecolor="gainsboro", zorder=3, ax=ax)
    kws = {"s": 10, "facecolor": "none", "linewidth": 0.2}
    sns.scatterplot(
        data=cluster_0,
        x="geolocation_lng",
        y="geolocation_lat", ax=ax, hue="Cluster", palette="deep", **kws)

    ax = plt.subplot(3, 2, 2)
    brazil_geo.plot(color="none", edgecolor="gainsboro", zorder=3, ax=ax)
    kws = {"s": 10, "facecolor": "none", "linewidth": 0.2}
    sns.scatterplot(
        data=cluster_2,
        x="geolocation_lng",
        y="geolocation_lat", ax=ax, hue="Cluster", palette="deep", **kws)

    ax = plt.subplot(3, 2, 3)
    brazil_geo.plot(color="none", edgecolor="gainsboro", zorder=3, ax=ax)
    kws = {"s": 10, "facecolor": "none", "linewidth": 0.2}
    sns.scatterplot(
        data=cluster_4,
        x="geolocation_lng",
        y="geolocation_lat", ax=ax, hue="Cluster", palette="deep", **kws)

    ax = plt.subplot(3, 2, 4)
    brazil_geo.plot(color="none", edgecolor="gainsboro", zorder=3, ax=ax)
    kws = {"s": 10, "facecolor": "none", "linewidth": 0.2}
    sns.scatterplot(
        data=cluster_5,
        x="geolocation_lng",
        y="geolocation_lat", ax=ax, hue="Cluster", palette="deep", **kws)

    ax = plt.subplot(3, 2, 5)
    brazil_geo.plot(color="none", edgecolor="gainsboro", zorder=3, ax=ax)
    kws = {"s": 10, "facecolor": "none", "linewidth": 0.2}
    sns.scatterplot(
        data=cluster_1_3,
        x="geolocation_lng",
        y="geolocation_lat", ax=ax, hue="Cluster", palette="deep", **kws)

# %% [markdown]
# ***
# ## 5.4 Maintenance

# %%


from sklearn.metrics.cluster import adjusted_rand_score

# %%
if config["maintenance"]["do"]:
    global_data.loc[:, "order_purchase_timestamp"] = data.loc[
        :,
        "order_purchase_timestamp"]
    global_data.loc[:, "order_purchase_timestamp"] = global_data.loc[
        :,
        "order_purchase_timestamp"].apply(pd.to_datetime)

# %%
if config["maintenance"]["do"]:
    global_data.info()

# %%
if config["maintenance"]["do"]:
    global_data.head()

# %%


def get_entire_period(dataset, temporal_index):
    return ceil(
        (dataset[temporal_index].max() - dataset[temporal_index].min()) /
        np.timedelta64(1, 'M'))

# %%
def get_frame_(dataset, temporal_index, start, period):
    temporal_index = dataset[temporal_index]
    end = start + pd.DateOffset(months=period)

    frame = dataset[(start <= temporal_index) & (temporal_index < end)]

    return frame

# %%


def kmeans_eval(dataset, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=1)
    kmeans.fit(dataset)
    labels = kmeans.labels_
    return labels

# %%


def rolling_eval_(
    dataset, features,
    temporal_index,
    n_clusters,
    initial_period,
    offset_month,
    key_index
):
    entire_period = get_entire_period(dataset, temporal_index)
    begin = dataset[temporal_index].min()
    ari_scores = []

    ref_period = get_frame_(
        dataset=dataset.copy(),
        temporal_index=temporal_index,
        start=begin,
        period=initial_period)
    ref_period.drop_duplicates(subset=["customer_id"], inplace=True)

    ref_labels = kmeans_eval(
        dataset=ref_period[features],
        n_clusters=n_clusters)

    for offset in np.arange(0, entire_period, offset_month):
        current_period = get_frame_(
            dataset=dataset,
            temporal_index=temporal_index,
            start=begin,
            period=initial_period + offset)
        current_period = current_period[
            current_period[key_index].isin(ref_period[key_index])
            ]
        current_period.drop_duplicates(subset=["customer_id"], inplace=True)

        current_labels = kmeans_eval(
            dataset=current_period[features], n_clusters=n_clusters)

        current_ari = adjusted_rand_score(ref_labels, current_labels)
        ari_scores.append((offset, current_ari))

    return ari_scores

# %%


if config["maintenance"]["do"]:
    global_data_preproc = preprocess_data(global_data)
    global_data_preproc["customer_id"] = global_data["customer_id"]
    global_data_preproc["order_purchase_timestamp"] = global_data[
        "order_purchase_timestamp"
        ]
    temp = global_data_preproc.drop_duplicates(
        subset=["customer_id"], inplace=False)

# %%
scores = None
if config["maintenance"]["do"]:
    scores = rolling_eval_(
        global_data_preproc,
        features=[
            "price", "freight_value", "product_weight_g",
            "product_volume_cm3", "price_sum", "frequency", "recency"],
        temporal_index="order_purchase_timestamp",
        n_clusters=6,
        initial_period=12, offset_month=2, key_index="customer_id")
scores

# %%
scores = pd.DataFrame(data=scores, columns=["offset", "ARI"])


plt.figure(figsize=(20, 7))
plt.title("Cluster evaluation", fontsize=22)
ax = sns.lineplot(data=scores, x="offset", y="ARI")

# minima = scores.iloc[scores["ARI"].idxmin()]["offset"]
# plt.axvline(minima, 0, scores["ARI"].max(), linestyle="--", linewidth=1)
plt.axvline(6, linestyle="--", linewidth=1)

plt.xlabel("Offset", fontsize=18)
plt.ylabel("ARI", fontsize=18)