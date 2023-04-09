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

