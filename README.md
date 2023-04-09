# Marketing-Campaign-Analysis - Yuri Velkis

<h2>Project Goal</h2>

<p>The dataset used for this project is based on a classic dataset available on Kaggle, which is designed to help students improve their skills in data cleaning, visualization, and machine learning. The goal of this project is to gain a better understanding of the customer profile, their spending patterns on the platform, and the effectiveness of the company's marketing campaigns.

More specifically, the project aims to achieve the following objectives: </p>

<ul>
  <li>Analyzing the Marketing Campaign Data Set.</li>
  <li>Explore the data set to understand the company's customer profile and behavior.</li>
  <li>Do a customer segmentation with cluster algorithms.</li>
  <li>Extract meaningful insights and provide actionable recommendations for the company.</li>

</ul>

<h2>Technology & Libraries</h2>
<ul>
  <li>Python</li>
  <li>Pandas</li>
  <li>Matplotlip</li>
  <li>Seaborn</li>
  <li>Numpy</li>
  <li>Sklearn</li>

</ul>

<h2>The Dataset</h2>

<table>
  <thead>
    <tr>
      <th>Feature</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>AcceptedCmp1</td>
      <td>1 if the customer accepted the offer in the 1st campaign, 0 otherwise</td>
    </tr>
    <tr>
      <td>AcceptedCmp2</td>
      <td>1 if the customer accepted the offer in the 2nd campaign, 0 otherwise</td>
    </tr>
    <tr>
      <td>AcceptedCmp3</td>
      <td>1 if the customer accepted the offer in the 3rd campaign, 0 otherwise</td>
    </tr>
    <tr>
      <td>AcceptedCmp4</td>
      <td>1 if the customer accepted the offer in the 4th campaign, 0 otherwise</td>
    </tr>
    <tr>
      <td>AcceptedCmp5</td>
      <td>1 if the customer accepted the offer in the 5th campaign, 0 otherwise</td>
    </tr>
    <tr>
      <td>Response (target)</td>
      <td>1 if the customer accepted the offer in the last campaign, 0 otherwise</td>
    </tr>
    <tr>
      <td>Complain</td>
      <td>1 if the customer complained in the last 2 years</td>
    </tr>
    <tr>
      <td>DtCustomer</td>
      <td>data on customer's enrollment with the company</td>
    </tr>
    <tr>
      <td>Education</td>
      <td>customer's level of education</td>
    </tr>
    <tr>
      <td>Marital</td>
      <td>customer's marital status</td>
    </tr>
    <tr>
      <td>Kidhome</td>
      <td>number of small children in the customer's household</td>
    </tr>
    <tr>
      <td>Teenhome</td>
      <td>number of teenagers in the customer's household</td>
    </tr>
    <tr>
      <td>Income</td>
      <td>customer's yearly household income</td>
    </tr>
    <tr>
      <td>MntFishProducts</td>
      <td>amount spent on fish products in the last 2 years</td>
    </tr>
    <tr>
      <td>MntMeatProducts</td>
      <td>amount spent on meat products in the last 2 years</td>
    </tr>
    <tr>
      <td>MntFruits</td>
      <td>amount spent on fruits products in the last 2 years</td>
    </tr>
    <tr>
      <td>MntSweetProducts</td>
      <td>amount spent on sweet products in the last 2 years</td>
    </tr>
    <tr>
      <td>MntWines</td>
      <td>amount spent on wines products in the last 2 years</td>
    </tr>
    <tr>
      <td>MntGoldProds</td>
      <td>amount spent on gold products in the last 2 years</td>
    </tr>
    <tr>
      <td>NumDealsPurchases</td>
      <td>number of purchases made with discount</td>
    </tr>
     <tr>
    <td>NunCatalogPurchases</td>
    <td>number of purchases made using catalog</td>
  </tr>
  <tr>
    <td>NunStorePurchases</td>
    <td>number of purchases made directly in stores</td>
  </tr>
  <tr>
    <td>NumWebPurchases</td>
    <td>number of purchases made through the company's website</td>
  </tr>
  <tr>
    <td>NumWebVisitsMonth</td>
    <td>number of visits to the company's website in the last month</td>
  </tr>
  <tr>
    <td>Recency</td>
    <td>number of days since the last purchase</td>
  </tr>
</table>
    
<h2>Data Cleaning</h2>
    
<p> The dataset doesn't need too much data cleaning, except some outliers that must to be treat.</p>
  
<h3> Numerical Features Box Plot</h3>

![Texto Alternativo](https://github.com/yurivlk/Marketing-Campaign-Analysis/blob/main/Box_plt_NumericalFeatures.png?raw=true)

<p>
<b>Let's treat the outliers</b>

<li><b>Income</b>: Very high values. As we have a normal distribution, we will keep 3 standard deviations.</li>

<li><b>Year_birth</b>: Very low values. We will use 100 years as the maximum age.</li>

</p>

<h3>Box Plot after the treatment</h3>
<table>
  <tr>
    <td><img src="https://github.com/yurivlk/Marketing-Campaign-Analysis/blob/main/Income_afterOut.png?raw=true" alt="Imagem 1"></td>
    <td><img src="https://github.com/yurivlk/Marketing-Campaign-Analysis/blob/main/Year_Birth_afterOut.png?raw=true" alt="Imagem 2"></td>
  </tr>
</table>

<h2>Exploratory Data Analysis</h2>

<h3>Let's take a look at our numerical features distributions</h3>

![Texto Alternativo](https://github.com/yurivlk/Marketing-Campaign-Analysis/blob/main/Numerical_features_Distribution.png?raw=true)

<h3>Let's zoom in our products distribution as some interisting insights can be taken from the above overview</h3>

![Texto Alternativo](https://github.com/yurivlk/Marketing-Campaign-Analysis/blob/main/Product_categories_distributions.png?raw=true)

<h3>Insights</h3>

These plots shows us interesting things:
<ul>
    <li>All of our products have a similar right-skewed distribution, which leads us to think that there is a long room for the company to increase sales in each product category.</li>
    <li>As the count of costumer who spend 50 or less is very close for each category, it can means thst this is a group of customers that are not spending to much and we need to investigate this to ensure that is a customer profile and try to come up something to increase their activities. </li>
    
</ul>
<p>
We will talk about this later...

Keep in mind that one of our goals is to improve the company's promotion acceptance rate and find room for revenue increase.

</p>

<b>Let's check if is there any correlation between the products category</b>

![Texto Alternativo](https://github.com/yurivlk/Marketing-Campaign-Analysis/blob/main/Products_Corr_M.png?raw=true)

<h3>Insights</h3>

<b>Medium to High Correlations</b>

<ul>
    <li>Wines & MeatProducts</li>
    <li>Fruits & MeatProducts</li>
    <li>Fruits & FishProducts</li>
    <li>Fruits & SweetProducts</li>
    <li>MeatProducts & SweetProducts</li>
    <li>MeatProducts & FishProducts</li>
        
</ul>

  <p>
  Here we can see a medium to high correlation between some products, which can helps the marketing team to build up strategies based in those relationship aiming to incrise the revenue, customer retention and campaigns performance.
  </p>

<h3>Feature Engineering</h3>

<ul>
<li><b>Age</b> : Customer's Age.</li>
<li><b>Education_lvl</b> : Group education levels into High, Medium, and Low.</li>
<li><b>Total_spent</b> : Total amount spent per customer.</li>
<li><b>Acc_age</b> : We will create this column to know how long the customer has had their account.</li>
<li><b>Marital_Status_grouped</b> : Group marital status as singles, couples, and widows.</li>
<li><b>Age group</b> : Group age range of consumers.</li>
<li><b>Total_campaignAcc</b> : Total campaign accepted per costumer.</li>
<li><b>Number of purchase</b> : Total purchase done by the costumer.</li>
</ul>

<h3>Let's check these new distributions to have a better understand of the Customers' profile</h3>

![Texto Alternativo](https://github.com/yurivlk/Marketing-Campaign-Analysis/blob/main/Categorical%20Features%20Distribution.png?raw=true)

<h3>Bi variate analysis</h3> 

![Texto Alternativo](https://github.com/yurivlk/Marketing-Campaign-Analysis/blob/main/Income_Distribution_by%20_Category.png?raw=true)

![Texto Alternativo](https://github.com/yurivlk/Marketing-Campaign-Analysis/blob/main/Spending_distributions%20by%20category.png?raw=true)

<h3>Insights</h3>

**Customer Profile**
<ul>
<li>Age Group: The database is composed mostly of middle-aged adults, followed by the elderly and young adults.</li>

<li>Marital Status: The database is composed mostly of couples.</li>

<li>Education Level: The database is composed mostly of individuals with a high education level.</li>

</ul>

**Income Profile**

The average salary does not vary much among different age groups, marital status, and individuals with or without teenage children. However, this does not apply to customers with young children at home, who tend to earn more.

**Spending Profile**

The average spending of our customers does not vary much among different age groups, marital status, and individuals with teenage children. However, customers with young children tend to spend less, while customers with a higher education level tend to spend more.

From the insights above, it can be observed that categorical variables mainly help us understand our customer profiles, except for the fact that customers with young children tend to spend less on average.


<h3>Where do the customers spend the most, and via which channel?</h3>
<table>
  <tr>
    <td><img src="https://github.com/yurivlk/Marketing-Campaign-Analysis/blob/main/Total_Sales_by_SC.png?raw=true"></td>
    <td><img src="https://github.com/yurivlk/Marketing-Campaign-Analysis/blob/main/Total_Spent_by_PC.png?raw=true"></td>
  </tr>
</table>

<h3>Is there a significant correlation between the products and the sales channels?</h3>

![Texto Alternativo](https://github.com/yurivlk/Marketing-Campaign-Analysis/blob/main/Products&SalesC_CorrM.png?raw=true)

<h3>Insights</h3>

We can see a significant correlation between wines/meats products and the sales channels such as catalog and in-store purchases, which provide insights to help the company increase revenue.

There are several actions that can be taken based on this information:

   `*Store organization: Presenting these products in a way that makes them more noticeable.`

   `*Promotions: Creating linked promotions between meat products and wines for these sales channels.`

   `*Marketing: Launching marketing campaigns aimed at promoting these products through the catalog and in-store sales channels.`
   
An interesting thing to look at is that the number of web visits is not correlated with the number of web purchases... It makes us raise questions like why our web traffic is not being converted to purchases?

<h3>Let's try to find if is there any pattern on the customers who have accepted our campaigns..</h3>

<h3>Do our categorical variables have a significant influence on accepting the campaigns?</h3>

<table>
  <tr>
    <td><img src="https://github.com/yurivlk/Marketing-Campaign-Analysis/blob/main/Promo_Acc_by_Marital_S.png?raw=true"></td>
    <td><img src="https://github.com/yurivlk/Marketing-Campaign-Analysis/blob/main/Promo_Acc_by_Education_L.png?raw=true"></td>
  </tr>
</table>

<h3>Insights</h3>

<p>Based on the plots below, we can conclude that the campaigns were more accepted by couples and customers with higher levels of education. However, these observations do not provide any actionable insights as they simply reflect the underlying data distribution.</p>

![Texto Alternativo](https://github.com/yurivlk/Marketing-Campaign-Analysis/blob/main/MKT_C_Analysis.png?raw=true)

<h3>Insights</h3>

Customers who accepted promotions have higher incomes.

Customers who spend more on wine and meat tend to have a higher acceptance rate.

Promotions were more accepted by clients who prefer catalog and web purchases.

Despite the poor performance of Campaign 3, this promotion seems to have attracted the largest customer profile in our dataset. This promotion is worth reviewing, as it could attract even more customers based on the profile of those who accepted it.

Customers who accept our promotions tend to spend more. The company can consider developing a loyalty program to encourage customers who already participate in promotions to continue, and to encourage those who do not to start.

___________________________________________________________________________________________________________________________________________________

<h3>Machine Learning & Model Evaluation</h3>

<p>Now let's apply the K-means algorithm to segment our customer database to help the marketing team make more efficient decisions based on the customers' profiles.</p>

<h3>K Means Clustering without Dimensionality Reduction</h3>

**Normalazing the Data**
```python
from sklearn.preprocessing import StandardScaler

df_enc_norm = StandardScaler().fit_transform(df_enc)

df_enc_scaled= pd.DataFrame(df_enc_norm, columns= df_enc.columns)

```

**Identifying the ideal number of clusters**

```python
from yellowbrick.cluster import KElbowVisualizer

# Instantiate the clustering model and visualizer
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,10))

visualizer.fit(df_enc_scaled)        # Fit the data to the visualizer
visualizer.show()
```
![Texto Alternativo](https://github.com/yurivlk/Marketing-Campaign-Analysis/blob/main/ML_Elbow.png?raw=true)

**Fitting The Data**

```python
km= KMeans(n_clusters=3, init='k-means++',
            n_init=10, max_iter=100, random_state=0)

km.fit(df_enc_scaled)

df["cluster_km_orig"] = km.labels_
```
**Evaluating the model**

```python
from sklearn.metrics import silhouette_score
sc= silhouette_score(df_enc_scaled, km.labels_)
print (f"Silhoutte Score is: {sc}")

Silhoutte Score is: 0.22149767133501388
```

**Cluster without Dimensionality Visuzalization**
![Texto Alternativo](https://github.com/yurivlk/Marketing-Campaign-Analysis/blob/main/Cluster1_Vis.png?raw=true)


**K Means Clustering with Dimensionality Reduction**

```python
from sklearn.decomposition import PCA

pca = PCA()
_ = pca.fit_transform(df_enc_scaled)
PC_components = np.arange(pca.n_components_) + 1
```
**Understanding the variance of our features**

```python
_ = sns.set(style='whitegrid', font_scale=1.2)
fig, ax = plt.subplots(figsize=(10, 7))
_ = sns.barplot(x=PC_components, y=pca.explained_variance_ratio_, color='b')
_ = sns.lineplot(x=PC_components-1, y=np.cumsum(pca.explained_variance_ratio_), color='black', linestyle='-', linewidth=2, marker='o', markersize=8)

plt.title('Scree Plot')
plt.xlabel('N-th Principal Component')
plt.ylabel('Variance Explained')
plt.ylim(0, 1)
plt.show()
```
![Texto_alternativo](https://github.com/yurivlk/Marketing-Campaign-Analysis/blob/main/Variance_PCA.png?raw=true)

**Reducing to 3 Components**

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=3)

pca.fit(df_enc_scaled)
PCA_df = pd.DataFrame(pca.transform(df_enc_scaled), columns=(["col1","col2",'col3']))
```

**Fitting in The Data**

```python
#Clustering the new reduced DF

km= KMeans(n_clusters=3, init='k-means++',
            n_init=10, max_iter=100, random_state=0)

km.fit(PCA_df)

from mpl_toolkits.mplot3d import Axes3D

#Plotting the clusters
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(PCA_df['col1'], PCA_df['col2'], PCA_df['col3'], s=40, c= km.labels_, marker='o' ,cmap = 'plasma')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.show()
```

![image](https://user-images.githubusercontent.com/97385851/230797086-31c7911b-ef5e-405c-8b52-f2c47a36175a.png)


**Evaluating the model**

```python
from sklearn.metrics import silhouette_score
#Evaluating the cluster
sc= silhouette_score(PCA_df, km.labels_)
print (f"Silhoutte Score is: {sc}")

Silhoutte Score is: 0.4478819732216168
```

**Reducing to 2 Components**

```python
pca = PCA(n_components=2)

pca.fit(df_enc_scaled)
PCA_df = pd.DataFrame(pca.transform(df_enc_scaled), columns=(["col1","col2"]))
PCA_df.describe().T
```

**Fitting in The Data**
```python
km= KMeans(n_clusters=3, init='k-means++',
            n_init=10, max_iter=100, random_state=42)

km.fit(PCA_df)

df['cluster_km_orig_pca2'] = km.labels_
```

![image](https://user-images.githubusercontent.com/97385851/230797215-31e0b981-b087-408d-9324-2a068db0b53d.png)

**Evaluating the model**
```python
sc= silhouette_score(PCA_df, km.labels_)
print (f"Silhoutte Score is: {sc}")

Silhoutte Score is: 0.5476329733619794
```

<h3>Insights</h3>

<p>We could notice that with the dimensionality redution for two componentes raised our Silhouete score significantly, therefore let's use this to visualize the clusters and try to understand each cluster caracteristics</p>

**Let's explore our customer segmentation...**

**Clusters Distribution**

![image](https://user-images.githubusercontent.com/97385851/230798039-1e36c426-20c1-46cc-b282-3b8e4b217c7b.png)

**Let's vizualise the spending profile of each cluster**

![image](https://github.com/yurivlk/Marketing-Campaign-Analysis/blob/main/Spending_P_Clusters.png?raw=true)

**Let's vizualise the Sales channel Preference**

![image](https://github.com/yurivlk/Marketing-Campaign-Analysis/blob/main/SalesC_P_Clusters.png?raw=true)
