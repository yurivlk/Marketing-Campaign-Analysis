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

<style>
  table {
    width: 100%;
  }
  td {
    width: 50%;
  }
  img {
    max-width: 100%;
    height: auto;
  }
</style>

<table>
  <tr>
    <td><img src="imagem1.png" alt="Imagem 1"></td>
    <td><img src="imagem2.png" alt="Imagem 2"></td>
  </tr>
</table>
