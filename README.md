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
5
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
