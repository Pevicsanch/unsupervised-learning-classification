# function to plot and determinate pca.n_componets for 95% explained variance

def sulhouette_k_means(data, max_k):
    from sklearn.cluster import KMeans
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import silhouette_score

    # A list holds the silhouette coefficients for each k
    silhouette_coefficients = []
    means = []

    for k in range(2, max_k):
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init=50)
        kmeans.fit(data)
        score = silhouette_score(data, kmeans.labels_)
        silhouette_coefficients.append(score)
        means.append(k)

    
    
    sns.set_context("talk")
    plt.style.use('ggplot')

    fig = plt.subplots(figsize=(10, 5))
    plt.plot(means, silhouette_coefficients, 'o-')
    plt.xticks(range(2, max_k))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Coefficient")
    
    plt.tight_layout()
    plt.grid(True)
    plt.show()


def create_cluster(df,title,desti):
    from sklearn.cluster import KMeans
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import silhouette_score


    for k in range(3,7): # + 1
        kmeans= KMeans(n_clusters=k,init="k-means++", n_init=50,random_state=42)
        kmeans.fit(df)
        desti[f'kmeans_{k}_{title}'] = kmeans.labels_
        print(f"cluster with {k} elements PCA with {title} - silhouette score: ", silhouette_score(df, kmeans.labels_))
        

def create_dendogram(data,title):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import scipy.cluster.hierarchy as shc
    from scipy.cluster import hierarchy

    sns.set_context("talk")
    plt.style.use('ggplot')

    mergings = shc.linkage(data.values, method = "ward", metric = 'euclidean')
    plt.figure(figsize=(12, 6))
    
    shc.dendrogram(mergings)
    plt.title(f'Hierarchical Clustering Dendrogram\n(Using 6 PCA comp. and {title}')
    plt.ylabel('distance (Ward)')           
    plt.tight_layout()
    plt.show()   



def pca_var(data, title):
    """ function to plot and determinate pca.n_componets for 95% explained variance.,"""

    # Create a PCA instance: pca
    from sklearn.decomposition import PCA
    import numpy as np
    import seaborn as sns 
    import seaborn as sns
    import matplotlib.pyplot as plt


    pca = PCA()
    pca.fit(data)
    prop_varianza_acum = pca.explained_variance_ratio_.cumsum()

    features = np.arange(pca.n_components_) + 1

    pca2 = PCA(.95)
    pca2.fit(data)

    # Plot the explained variances
    sns.set_context("talk")
    plt.style.use('ggplot')

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))

    sns.barplot(x=features, y=pca.explained_variance_ratio_,
                ax=axes[0], color="pink")
    axes[0].set_xlabel('PCA feature')
    axes[0].set_ylabel('variance')
    axes[0].set_title(
        f'perc. variance explained by each component({title}).\n {pca2.n_components_} PCA components explained 95% variance.')

    axes[1].plot(
        np.arange(len(data.columns)) + 1,
        prop_varianza_acum,
        marker='o'
    )

    for x, y in zip(np.arange(len(data.columns)) + 1, prop_varianza_acum):
        label = round(y, 2)
        axes[1].annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center'
        )

    axes[1].set_ylim(0, 1.1)
    axes[1].set_xticks(np.arange(pca.n_components_) + 1)
    axes[1].set_title(f'Acumulative explained variance perc({title}).\n\n')
    axes[1].set_xlabel('PCA feature')
    axes[1].set_ylabel('Accumulated variance perc.')

    plt.xticks(features)
    plt.tight_layout()
    plt.grid(True)
    plt.show()


def create_pca(data, title,df):
    from sklearn.decomposition import PCA
    
    """ #create a funcion to performance PCA with 6 components"""
    PCA_columns = [f'PC1_{title}', f'PC2_{title}', f'PC3_{title}',
        f'PC4_{title}', f'PC5_{title}', f'PC6_{title}']
    pca = PCA(n_components=6)
    pca.fit(data)
    df[PCA_columns] = pca.transform(data)
    return PCA_columns

# create function to find optimum number of clusters


def optimise_k_means(data, max_k):
    """function to find optimum number of clusters, generate the elbow plot and show the best number of cluster"""
    from sklearn.cluster import KMeans
    from kneed import KneeLocator
    import seaborn as sns
    import matplotlib.pyplot as plt


    means = []
    inertias = []
    
    for k in range(1, max_k):
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init=50,
                            max_iter=500,
                            random_state=42)
        kmeans.fit(data)

        means.append(k)
        inertias.append(kmeans.inertia_)
    kl = KneeLocator(
            range(1, max_k), inertias, curve="convex", direction="decreasing")


# generate the elbow plot
    sns.set_context("talk")
    plt.style.use('ggplot')

    fig = plt.subplots(figsize=(10, 5))
    plt.plot(means, inertias, 'o-')
    plt.xlabel("number of clusters")
    plt.ylabel("inertia score")
    plt.title(f"elbow point k: {kl.elbow}")
    plt.xticks(range(1, max_k))

    plt.tight_layout()
    plt.grid(True)
    plt.show()


def create_dendogram(data,title):

    import seaborn as sns
    import matplotlib.pyplot as plt
    import scipy.cluster.hierarchy as shc
    from scipy.cluster import hierarchy
    
    sns.set_context("talk")
    plt.style.use('ggplot')

    mergings = shc.linkage(data.values, method = "ward", metric = 'euclidean')
    plt.figure(figsize=(12, 6))
    
    shc.dendrogram(mergings)
    plt.title(f'Hierarchical Clustering Dendrogram\n(Using PCA comp. and {title}')
    plt.ylabel('distance (Ward)')           
    plt.tight_layout()
    plt.show()  

def clean_data(df):
    import pandas as pd
    import numpy as np
    import math
    from sklearn.preprocessing import LabelEncoder


# wihtout arrival time is imposible to know if a flight arrived or not
    df.dropna(subset=["ArrTime"], inplace=True)
# as the column "year" have just one value, we are going to drop it.
    df.drop(["Year"], axis=1, inplace=True)
# changing types:
    df[['DepTime', 'ArrTime']] = df[['DepTime', 'ArrTime']].astype(int)

# to_change changing type from "objet" to "category"
    to_change = ["Month", "DayofMonth", "DayOfWeek", "FlightNum", "Origin",
                "Dest", "TailNum", "UniqueCarrier", "CancellationCode", "Cancelled", "Diverted"]

    for x in to_change:
        df[x] = df[x].astype("category")

    to_date = ["CRSArrTime", "CRSDepTime", "DepTime", "ArrTime"]

    # to_date changing type to date

    for x in to_date:
        df[x] = df[x].astype(str).str.zfill(4)
        df[x] = df[x].astype(str).str[:2] + ':' + df[x].astype(str).str[2:] + ':00'
        df.loc[df[x] == "24:00:00", x] = "00:00:00"
        df[x] = pd.to_datetime(df[x], format="%H:%M:%S").dt.time

    # where on time or delay new column
    # where delayed = 1
    # and on time = 0

    df['with_delay'] = np.where(df['ArrDelay'] > 0, 1, 0)
    df['with_delay'] = df['with_delay'].astype("category")

    # let's drop the columns of Delays that are not ArrDelay, because ArrDelay is the sum of all others
    df = df.drop(['CarrierDelay', 'WeatherDelay', 'NASDelay',
                'SecurityDelay', 'LateAircraftDelay'], axis=1)

    # keep values cancelled ==1 (flights cancelled) in a diferent dataframe.
    df_cancelled = df.loc[df['Cancelled'] == 1]
    # keep values diverted ==1 (flights diverted) in a diferent dataframe.
    df_diverted = df.loc[df['Diverted'] == 1]

    # New dataframe with out cancelled diverted and ontime values
    df_analisys = df.loc[(df['Cancelled'] == 0) & (
        df['Diverted'] == 0) & (df['with_delay'] == 1)].copy()
    # clean memory
    del df

    # As we aren't going to use it, we can drop "Cacelled","CancellationCode","Diverted","with_delayt" columns from our new dataframe
    df_analisys.drop(columns=["Cancelled", "CancellationCode",
                    "Diverted", "with_delay"], inplace=True)

    # filter numerical features
    num_features = df_analisys.select_dtypes(exclude=["category"])

    # Cyclical features encoding: converting the original time to their corresponding cosine and sine values
    to_transform = ["DepTime", "CRSDepTime", "ArrTime", "CRSArrTime"]

    for column in to_transform:
        num_features[[f'h_{column}', f'm_{column}', f's_{column}']] = num_features[column].astype(
            str).str.split(':', expand=True).astype(int)
        num_features.drop([f's_{column}'], axis=1, inplace=True)
        num_features[f'x_{column}'] = num_features[f'h_{column}'] * \
            60 + num_features[f'm_{column}']
        num_features.drop([f'h_{column}'], axis=1, inplace=True)
        num_features.drop([f'm_{column}'], axis=1, inplace=True)
        num_features[f'x_{column}'] = 2 * math.pi * \
            num_features[f'x_{column}'] / num_features[f'x_{column}'].max()
        num_features[f"cos_{column}"] = np.cos(num_features[f'x_{column}'])
        num_features[f"sin_{column}"] = np.sin(num_features[f'x_{column}'])
        num_features.drop([f'x_{column}'], axis=1, inplace=True)
        num_features.drop([column], axis=1, inplace=True)
    # keep label of the airlines
    num_features["true_label_names"] = df_analisys["UniqueCarrier"]

    # create a label encoder obj-
    label_encoder = LabelEncoder()
    true_labels = label_encoder.fit_transform(num_features["true_label_names"])
    # keep true labels
    num_features["true_labels"] = true_labels

    return num_features

