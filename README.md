# **Investment Report** #


**Welcome and Introduction**

Dear user, 

We have created this code to allow everyone to be able to find an investment portfolio that would fit them. Through this tool, you are able to find out multiple information about the companies in S&P 500, read the description of what the company is doing, see how it has been performing financially and finally find out what are the optimal weights for the investment in the selected company. 

The only input that is required from you is the list of tickers from the companies you are interested in, your maximum loss aversion for the investment and the certainty (significance level) at which you want your results. 

Please do not hesitate to contact us if any questions arise.


**Main usage**

The goal of this code is to offer the user a range of portfolio allocation strategies in terms of expected return and risk as well as to give him a better understanding of the companies included in his final portfolio. 
 

**User input**

The user is presented with a total of 400 tickers of the S&P 500 Index and has the option to select the stocks of his portfolio. His maximum accepted loss, as a percentage, and the significance level are also considered in order to calculate the optimal portfolio. 


**Functions created**

The code runs based mainly on the following functions: 
1.	 Get_risk_free, calculates the risk-free rate (10 years treasury yield).
2.	 Get_stock_data, calculates daily returns for each stock that has been selected by the user.
3.	 Pick, in combination with “tkinter”, stores the tickers that have been selected by the user
4.	 F_alpha & F_var, store the maximum accepted loss and the significance level that have been selected by the user
5.	 Closewindow, finishes the first stage of the code as soon as the user has finished choosing his portfolio preferences.
6.	 Frontierplot, creates the Efficient Frontier including various portfolios of interest (see below). Furthermore, Utility of the user is derived considering his risk aversion (maximum accepted loss).
7.	 Hist_returns, creates a graph representing past and future returns of the selected stocks as well as of the Maximum Utility Portfolio
8.	 Get_stock_description, gathers information for the stocks that have been selected by the user

Additional processes supplement the output with summary statistics for each company, performance of the different available portfolios, and weights for each portfolio allocation strategy. Furthermore, the code ensures that the user inputs are correct and if not, it opens again the first stage window. All the data are extracted into a pdf file, available to the user.


**Output**

As soon as the code runs, a pdf file is created, which contains the following information:

1.	 Portfolio components, as per user’s stocks selection
2.	 Maximum loss acceptable and significance level, as per user’s preferences
3.	 Company description, of each stock selected by the user
4.	 Summary statistics, of each stock selected by the user
5.	 Portfolio analysis, which includes 3 main portfolios with their respective expected return and risk:
	
	a.	Optimal Risky Portfolio, which maximizes the risk-adjusted return for the investor. 
		It is expressed as the slope of the CAL and is located on the tangency of the Efficient Frontier and the CAL. 
		Use of leverage gives the user the opportunity to increase or decrease his expected return (along with risk).
	
	b.	Minimum Variance Portfolio, which as the name suggests offers the least possible risk to the investor.
	
	c.	Maximum Utility Portfolio, which maximizes the user’s expected utility and should be the preferred 
		portfolio allocation strategy according to the user’s risk preferences.
	
6.	 Portfolio weights, for each portfolio as presented above.
7.	 Portfolio performance, of the individual stocks as well as of the Maximum Utility Portfolio

