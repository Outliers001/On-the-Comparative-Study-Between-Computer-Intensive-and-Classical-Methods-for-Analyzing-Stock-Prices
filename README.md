# On-the-Comparative-Study-Between-Computer-Intensive-and-Classical-Methods-for-Analyzing-Stock-Prices

## Abstruct
The stock market plays a pivotal role in the growth of the economy of a country. Stock
market analysis enables investors to identify the intrinsic worth of a security even before
investing in it. Factors like supply and demand in the market, market sentiment and
investor’s expectations, economic and political shocks can affect stock prices. All these
factors make stock prices volatile and noisy. Machine learning techniques have the
capacity to insights the unseen and derive hidden patterns from features of data like the
latest reports about a company, their opening and closing price, day’s high and day’s
low etc.
Several methods including statistics and artificial intelligence are being developed
in order to turn prediction more accurate and reliable. In this research, we conduct a
comparative study between computer intensive and classical methods on the problem of
stock market forecasting in case of Dhaka Stock Exchange. Two machine learning models,
Linear Support Vector Regression(SVR) and K-Nearest Neighbor (KNN) Regression
be treated as computer intensive method and a statistical method, Auto-regressive
Integrated Moving Average (ARIMA) be treated as classical method . The purpose of
this research is to examine the feasibility and performance of SVR, KNN and ARIMA
model in forecasting closing price of selected company from Dhaka Stock Exchange
(DSE). We optimize the models by testing different hyper-parameters, i.e., the values
of " and C in case of Linear SVR, the values of k in case of KNN and the values of p, d
q in case of ARIMA. For each model, Root Mean Square Errors(RMSE) are calculated
by analyzing historical data of three stocks that are based on Dhaka Stock Exchange
(DSE). Our study shows that, Linear SVR is more effective and efficient technique to
predict the prices of Dhaka Stock Exchange (DSE).

## Introduction

Stock markets are today a large part of world’s financial system and the stock markets
have a large impact on economy of a country. Stock prices change every day by market
forces. By this we mean that share prices change because of supply and demand. If
more people want to buy a stock (demand) than sell it (supply), then the price moves
up. Conversely, if more people wanted to sell a stock than buy it, there would be greater
supply than demand, and the price would fall. There are many features that are act on
increasing or decreasing the supply and demand of a stock, which are make the pattern
of the prices noisy. In this epoch of fourth industrial revolution, researchers take this
job as a challenging assignment.
In this era of Artificial Intelligence, wherever we see, we find applications of Machine
Learning and Deep Learning. Our smartphones are getting more powerful with
limited resources, self-driving cars are already on the streets, and not surprisingly, multinational
investment banks like JPMorgan Chase and Morgan Stanley now have their
own Machine Learning department to assist them to make investment decisions using
AI.
Correctly predicting of stock price carries obvious economic benefits. But the most
accurate way to predict the outcome of the stock market is a frequently discussed matter.
It is extremely difficult to take into consideration all those factors that can influence a
stock (See for example: Ida Vainionpaa, 2014). For example, internal development,
world events, inflation and interest rates, exchange rates and lastly hype etc. (See for
example: Wolski, 2014) can affect a stock.
However, the most important variable that are highly correlated with next day’s
stock prices are A day’s avg. price, Day’s high, Day’s low, Year high, YTD change, P/E
ratio etc. (See for example: Mbeledogu.N. N, 2012) and Open, High, Low and Close
prices that are recorded on daily basis have a higher informational content that other

intraday prices (See for example: N.M Fiess, R. MacDonald, 2002). Using computational
intelligence, machine learning and data mining to find correlations in large data
sets that humans are not capable of finding, which can be used as a prediction method
in stock prices as well as others uncertain sector in finance (See for example: Alexander,
1998). There are currently several methods used in predicting stock prices. These
are including various statistical method; machine learning algorithms and deep learning
algorithms often used by traders. This project follows mainly K-Nearest Neighbor
(KNN) and Support Vector Regression(SVR) and a comparison with Autoregressive Integrated
Moving Average (ARIMA), in which algorithms are presented with historical
stock data. The algorithms use information from historical data to train a model that is
expected to infer future prices given recent price information. The models are trained
on daily stock exchange data, to make short-term predictions for one day ahead.

### 1.1 Problem and Motivation
Forecasting is a procedure of making a statement about the way things will happen
in the future in the light of available information. The more precise the forecast, the
simpler it could be to reach an agreement about the future. Stock market prediction is
the act of trying to determine the future value of a company stock or other financial
instrument traded on an exchange. The successful prediction of a stock’s future price
could yield significant profit.
Predicting stock market accurately is a difficult task to do because of the dynamic nature
of the stock market. Machine learning algorithms can use historical data, to predict
what will happen to the stock market over the given period of time. There are several
studies on predicting closing price of Dhaka Stock Exchange (DSE). In our study, we
inquire the best among them.
The problem statement we have chosen to work with in this project follows from the
hypothesis that the SVR algorithm is a more precise way of predicting closing prices
than the KNN and ARIMA to predict the stock prices of DSE.
### 1.2 Objectives
Prediction of financial markets requires an analysis of the price trend and the determination
of their inherent value. The foremost purpose of this study is to find an optimum
algorithm to forecast closing price of selected company of Dhaka Stock Exchange. On
this purpose, we compared specific machine learning algorithms with ARIMA; the most
familiar time-series forecasting method and came up with best and comparatively easy
solution for forecasting and implementing on a system. The main objectives of this
study are as follow:
• to give an overview of time series forecasting in stock market analysis;
• to extract the inter-dependency among the variables;
• to illustrate an intelligible outline of the algorithms;
• to investigate the optimum values of the hyper parameters and make prediction;
9
• to evaluate the efficient method for policy making which provides helpful information
regarding all the tests we allow them to perform.
1.5 Research Data
To conduct the study we collect the data set from Dhaka stock exchange (DSE), Bangladesh.
The dataset is available on the archive of official website of Dhaka Stock Exchange
(www.dsebd.org) and Stock Bangladesh (www.stockbangladesh.com). Stock Bangladesh
is the first and oldest financial portal based on share markets of Bangladesh. We collect
adjusted data from data resource of Stock Bangladesh.
In our study, we have selected 2 pharmaceuticals company and a telecommunications
company as listed to DSE - BXPHARMA( Beximco Pharmaceuticals Ltd.),
SQPHARMA (Square Pharmaceuticals Ltd.) and GP(Grameenphone Ltd.) over the
period January 1, 2016 to December 31,2018.
1.6 Project Orientation
The rest of the book is organized as follow:
• Chapter 2 Stock Market: A brief discussion about stock markets and role stock
markets in economic development.
• Chapter 3 Machine Learning: Steps and techniques that are used in machine learning
modeling.
• Chapter 4 Time Series Analysis: Steps and techniques that are used in any time
series analysis.
• Chapter 5 System Implementation and Result: Applying those steps and techniques
on the stock market data-sets and discussion about the findings. Comparison
between results are also a major issue for this paper.
• Chapter 6 Conclusion and Future Work: Summary of the computer intensive and
classical methods for analysing stock prices of DSE and discussion about the issues
that can added in future work to make the system more efficient and accurate.
10
# Chapter 2
## Stock Market
### 2.1 Introduction
Stock markets are home to extensive trade in shares and other company securities, and
they have a crucial role to play in the success of commerce and the overall health of
an economy. There are stock markets across the world’s global financial centres which
regulate markets and run indices on which traders can take exposure with a view to
delivering capital growth on the size of their position. Some of the biggest and most
famous stock exchanges include the London Stock Exchange (LSE), New York Stock
Exchange (NYSE) and Japan Exchange Group(JPX).The main participants in the equity
market include individual investors, institutional investors and finally public companies.
2.2 Capital market
Capital market is a market where buyers and sellers engage in trade of financial securities
like bonds, stocks, etc. The buying/selling is undertaken by participants such as
individuals and institutions. Capital market consists of primary markets and secondary
markets. Primary markets deal with trade of new issues of stocks and other securities,
whereas secondary market deals with the exchange of existing or previously-issued securities.
Another important division in the capital market is made on the basis of the
nature of security traded, i.e. stock market and bond market.
The instruments issued in capital markets are listed below:
• Shares: Share is the share in the share capital of the company.Share is one of the
units into which the capital of company is divided. A person having the shares of
the company is called as shareholder of that company, He is regarded as the part
of owner of the company.
• Debentures: Debentures are long term borrowed funds of the company. They
11
have fixed maturity period as well as fixed interest rate. These are the certificates
issued under common seal of the company.
• Bonds: Bonds are the long term borrowed funds of the government and also
companies. Like debentures have fixed maturity and fixed interest rate even bonds
have. Here interest charged on bonds termed as coupon rate.
• Derivatives: These are instruments that derive from other securities, which are
referred to as underlying assets. The price, riskiness and function of the derivative
depend on the underlying assets since whatever affects the underlying asset must
affect the derivative.
2.3 Stock Market and Its History
It was in 12th century France that the ‘Courretiers de change’ took on the task of managing
and regulating agricultural communities debt. This was on behalf of the french
banks of that time. This group of men traded the debts and became known as first
brokers. In the thirteenth century a common misconception arose in Bruges. Commodity
traders started to meet in a house that was owned by a gentleman named Van der
Beurze, and so later, in 1409l, this group became known as ‘Brugse Beurse’. This was
the formalised, and even institutionalised a meeting that had previously been known as
an informal gathering. In actual fact, those meetings took place in a building in the city
of Antwerp that was owned by Van der Beurze. Most merchants of that time undertook
their trading in the city of Antwerp. This idea spread across the region of Flanders very
quickly, and also in neighbouring countries, and soon in the cites of Ghent and Rotterdam
‘Beurzen’ soon opened.
It was in the middle of the thirteenth century in Venice, that bankers started to trade
in government securities. It was soon outlawed in 1351, by the government of Venice
spreading stories. The idea behind this was to decreased the cost of Venetian government
funds. In the large cities of Italy, including Pisa, Genoa, Florence and Verona,
bankers started to trade, again with government securities. This happened around the
fourteenth century. What allowed this trading to happen was that these cities were not
ruled by a duke, but they were independent states in their own right. These states were
ruled by an elected citizen council. Businesses and organisations in Italy were also the
the first to sell shares on the stock market. It was not until the sixteenth century that
businesses in the United Kingdom were able to sell shares. Many other countries then
followed.
12
The first joint-stock company was the Dutch East India country that was formed in
1602. A joint-stock company is an organisation in which stakes can be purchased and
sold by shareholders. A shareholder may own a proportion of the business depending on
how many shares they purchase. A certificate of ownership is issued to the shareholder
detailing the proportion of shares held in the company. The Dutch East India company
was the first to get fixed capital stock, and so company stock was continuously traded
on Amsterdam Stock Exchange. It was not long after this happened that other trade followed,
in different derivatives, came into being on the Amsterdam market. Something
known as short selling also occurred. This is where securities that are not currently
owned by anyone are attempted to be sold. This, however, was soon seen as illegal and
was banned in 1610 by the authorities.
Today, there are stock markets all over the world, in both those countries that are
both developed, and developing. The most well known stock markets, and the worlds
largest are in the United States of America, as well as in the United Kingdom, China,
Canada, India, Japan, Germany, France, Netherlands and South Korea.
2.4 Importance of Stock Market
It is one of the main ways that businesses and other organisations raise money. Stock
markets enable companies to be traded publicly, and so raise capital to expand their
business. The liquid nature of the stock exchange allows those who invest to easily buy
and sell their securities. This is a particularly attractive feature to investors.
Over time, the cost of shares have been studied, and it has been realised that the price
are an important economic indicator, and can indicate current social mood. Usually, the
stock market is seen to be the most important indicator of a nations economic development
and current strength. When share prices are rising, for example, this is seen as
increased investment in business and indicates a rising economy
2.5 Relation of The Stock Market to The Modern Financial
System
The modern economy cannot exist without the efficient financial system that is defined
as the collection of markets, institutions, instruments and regulations through which
the financial securities are traded, interest rates are determined and financial services
are produced and delivered around the world [See for example: Pietrzak, Polański and
Woźniak, 2008] The financial system is regarded as one of the most important creations
of the modern society and it is described as an integrated part of the economic system
13
and by this – a significant part of the social system [See for example: Pietrzak et al.,
2008, p. 15].
The financial system in most western countries has undergone a remarkable transformation.
One feature of this development is disintermediation. A portion of the funds
involved in saving and financing, flows directly to the financial markets instead of being
routed via the traditional bank lending and deposit operations. The general public
interest in investing in the stock market, either directly or through mutual funds, has
been an important component of this process.
Statistics show that in recent decades shares have made up an increasingly large
proportion of households’ financial assets in many countries. In the 1970s, in Sweden,
deposit accounts and other very liquid assets with little risk made up almost 60 percent
of households’ financial wealth, compared to less than 20 percent in the 2000s. The
major part of this adjustment is that financial portfolios have gone directly to shares but
a good deal now takes the form of various kinds of institutional investment for groups
of individuals, e.g., pension funds, mutual funds, hedge funds, insurance investment of
premiums, etc.
The trend towards forms of saving with a higher risk has been accentuated by new
rules for most funds and insurance, permitting a higher proportion of shares to bonds.
Similar tendencies are to be found in other industrialized countries. In all developed
economic systems, such as the European Union, the United States, Japan and other developed
nations, the trend has been the same: saving has moved away from traditional
(government insured) bank deposits to more risky securities of one sort or another
2.6 Bangladesh Securities and Exchange Commission:
Capital market plays a significant role in the economy as a source of long term financing.
A fair, efficient and transparent capital market is essential for a country for its industrialization
and economic development. To develop such a fair, efficient and transparent
capital market, the Bangladesh Securities and Exchange Commission was established
as the regulator through enactment of the Bangladesh Securities and Exchange Commission
Act. 1993 in June 08, 1993, with the following mission:
• Protecting the interest of investors in securities;
• Developing the capital and securities markets; and
• Framing of securities rules.
In Bangladesh, there are two stock exchanges, the Dhaka Stock Exchange (DSE) and
the Chittagong Stock Exchange (CSE), DSE was setup on April 28, 1954 that started
formal trading in 1956. In 1995, CSE was setup.
14
2.6.1 The Dhaka Stock Exchange (DSE)
Dhaka Stock Exchange Ltd (DSE) is the oldest and largest stock exchange in Bangladesh.
Though DSE was established in 28 April 1954 but its commercial operation started in
1956.The board of directors consisting of 25 members, which directs the activities of
DSE. Out of the 12 directors are elected by direct votes of DSE members and other 12
directors are nominated by the elected members from non-DSE members upon approval
of the Commission and their CEO is a member by chair. At present, there are 250 members
in DSE of which 229 members are registered by the Commission for conducting
securities business. DSE has expanded its on-line trading network to many district towns
like Gazipur, Narayanganj, Comilla, Feni, Habiganj, Maulvibazar, Mymenssngh, Chittagong,
Khulna, Syihet, Kushtia, Barisal, Rajshahi and Bogra including the divisional
towns.
***2.6.2 The Chittagong Stock Exchange (CSE)

The Chittagong Stock Exchange ltd (CSE), the second stock exchange, was established
in 1995.The board of directors consisting of 25 members, which directs the activities of
CSE. Out of them, 12 directors are elected by direct votes of CSE members and other 12
directors are nominated by the elected members from non-CSE members upon approval
of the Commission and their CEO is a member by chair. Now there are 148 members
in CSE of which 138 members are registered by the Commission for conducting securities
business. CSE has expanded its on-line trading network to many district towns
like Chittagong, Dhaka, Narayanganj, Feni, Syihet, Noakhali, Coxs Bazar, Khulna, Jessore,
Harisal and Rajshahi. Besides, CSE introduced internet trading system by which
investors can trade from anywhere.
***2.7 Dhaka Stock Exchange(DSE): A Brief Description

The Dhaka Stock Exchange (DSE), located in Motijheel, Dhaka, is one of two financial
marketplaces in Bangladesh (the other is the Chittagong Stock Exchange). The DSE was
incorporated in 1954, and formal trading began in 1956. Originally, the exchange was
called the East Pakistan Stock Exchange Association Ltd.; in 1962 the name was revised
to: East Pakistan Stock Exchange Ltd.; and two years later, the name again changed to
the current, Dhaka Stock Exchange Ltd. The Dhaka Stock Exchange is registered as a
Public Limited Company (PLC) and is regulated by the Bangladesh Securities and Exchange
Ordinance of 1969, the Companies Act of 1994 (Bangladesh) and the Securities
and Exchange Commission (SEC) Act of 1993 (Bangladesh)1.
1https://www.investopedia.com/terms/d/dse.asp
15
*** 2.7.1 History and Formation of DSE
Dhaka Stock Exchange named was “Stock Exchange Association Ltd” in 1954 and
started it’s journey from East Pakistan. First formal trading started in 1956. After six
years of trading it’s name was renamed to ” East Pakistan Stock Exchange Ltd”, remember
it was in 23 june, 1962. Two years passed and in 13th may 1964 again it was renamed
to “Dacca Stock Exchange Ltd”. We all know that our liberation year took place
in 1971, a nasty war killed many muslim Bangladeshis as well as Hindu Bangladeshis,
so the market collapse at that time and it didn’t operate for five years. In 1976, share
market got it’s life back and started it’s journey in a free country named “Bangladesh”.
On september 16th 1986.
2.7.2 Major Functions of DSE
The major functions of DSE are2:
• Listing of Companies (As per Listing Regulations).
• Providing the screen based automated trading of listed Securities.
• Settlement of trading (As per Settlement of Transaction Regulations).
• Gifting of share / granting approval to the transaction/transfer of share outside the
• trading system of the exchange (As per Listing Regulations 47).
• Market Administration Control.
• Market Surveillance.
• Publication of Monthly Review.
• Monitoring the activities of listed companies (As per Listing Regulations).
• Investors grievance Cell (Disposal of complaint bye laws 1997).
• Investors Protection Fund (As per investor protection fund Regulations 1999).
• Announcement of Price sensitive or other information about listed companies
through online.
2https://www.dsebd.org/ilf.php
16
*** 2.7.3 Role of Dhaka Stock Exchange in Economic Development
To achieve the desired objective for growth and prosperity the world economy always
changed to integrate itself in the parts of the world. Both developed and underdeveloped
countries are supposed to move the wheel of socioeconomic development by making resources
to facilitate the economic growth through appropriate allocation of the same. In
Bangladesh the idle money is not properly canalizes due to some non-availability of
investment arena with a safe return of both principal and interest thereof. Most of the
potential entrepreneurs often gather in the capital markets to meet the capital requirements.
The surplus units of the society are not supposed to invest their money. So, they
are often ready to supply their money to purchase securities from the capital markets.
As a result stock exchange plays a crucial role to mobilize capital for the development
of a capital market.
After opening of the market to the foreign investors in 1992 the trading has been
increasing day by day. Foreign investors are encouraged because it was found that without
substantial foreign investment, domestic private investment would be inadequate to
achieve economic development. The role of DSE in promoting securities of Bangladesh,
firstly it is essential to evaluate the objectives of DSE. Being a self-regulated non-profit
organization Dhaka Stock Exchange was established to perform the following objects:
• To bring together buyers and sellers to bargain for fixing the security-price providing
them with a safer market place.
• To reflect the current market price of securities.
• To protect the interest of investors.
• To analyze and review the fiscal measures that affect the investors.
• To reflect impartial and detailed disclosures about listed companies through publishing
journals and brochures.
2.8 Conclusion
In this chapter, we have discussed about bout stock markets and role stock markets in
economic development.
In the next chapter of machine learning, we will discuss about the computer intensive
methods that are frequently used in stock prices prediction.

# Chapter 3
## Machine Learning
### 3.1 Introduction
Machine learning is a field of computer science that aims to teach computers how to
learn and act without being explicitly programmed. More specifically, machine learning
is an approach to data analysis that involves building and adapting models, which allow
programs to ”learn” through experience. Machine learning involves the construction of
algorithms that adapt their models to improve their ability to make predictions.Consider
a game of Chess, where E = Experience of playing many games of Chess, T = Task of
playing Chess. P = Probability that the program will win the next game. A computer
program is said to learn from experience E with respect to some class of tasks T and
performance measure P if its performance at tasks in T, as measured by P, improves
with experience E. [See for example: Mitchell, 1997]
In simpler terms, Machine learning is a sub-domain of
