
"""" Initial Setup """

# Importing necessary libraries
import pandas as pd
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf
import datetime as date
from fpdf import FPDF
from tkinter import *
import numpy as np
import scipy.stats as st
from matplotlib.lines import Line2D
from pylab import rcParams
import matplotlib.ticker as mtick
import os

# Set figure size
rcParams['figure.figsize'] = 7, 4

# Override yfinance with pandas
yf.pdr_override()

"""" Functions to retrieve necessary data """


# Function to get the risk-free rate
def get_risk_free():
    # We download the data from yahoo finance
    risk_free = pdr.get_data_yahoo("^TNX", start=date.date.today() - date.timedelta(days=1),
                                   end=date.date.today())['Adj Close'][0]
    # We divide by 100 to have it in percentage form
    risk_free = float(risk_free / 100)
    return round(risk_free, 3)


# Function to download daily returns data for a list of stock tickers
def get_stock_data(tickers):
    # Create an empty dataframe to store the data
    df = pd.DataFrame()

    # Loop through the list of tickers
    for ticker in tickers:
        # Download the data for the current ticker from Yahoo Finance
        data = pdr.get_data_yahoo(ticker, start=date.date.today() - date.timedelta(days=5 * 365), end=date.date.today())
        # Calculate the daily returns
        data['returns'] = data['Adj Close'].pct_change()
        # Add the returns data to the dataframe
        df[ticker] = data['returns']

    # Drop first row as it contains NAs as the returns cannot be calculated for it
    df = df.iloc[1:, :]

    # Drop rows that have any missing values
    df = df.dropna()

    return df


"""" Interface to ask user for input """

# Load 400 tickers of S&P 500
tickersTmp = ["AAPL", "MSFT", "AMZN", "GOOGL", "UNH", "GOOG", "JNJ", "XOM", "NVDA", "JPM", "TSLA", "PG", "V",
              "HD", "CVX", "MA", "PFE", "ABBV", "LLY", "MRK", "META", "PEP", "KO", "BAC", "AVGO", "TMO", "WMT", "COST",
              "MCD", "CSCO", "ABT", "DHR", "NEE", "ACN", "LIN", "DIS", "ADBE", "PM", "WFC", "VZ", "BMY", "TXN", "CMCSA",
              "RTX", "HON", "AMGN", "COP", "NKE", "T", "CRM", "NFLX", "IBM", "UPS", "UNP", "QCOM", "CVS", "LOW", "ORCL",
              "CAT", "SCHW", "ELV", "DE", "GS", "LMT", "SBUX", "MS", "SPGI", "INTC", "INTU", "GILD", "BA", "AMD", "BLK",
              "PLD", "MDT", "ADP", "CI", "AMT", "ISRG", "TJX", "MDLZ", "CB", "AMAT", "GE", "AXP", "ADI", "C", "TMUS",
              "MO", "MMC", "SYK", "PYPL", "REGN", "NOW", "DUK", "NOC", "BKNG", "SO", "VRTX", "PGR", "EOG", "SLB", "BDX",
              "APD", "MMM", "MRNA", "ZTS", "TGT", "BSX", "CL", "CSX", "HUM", "FISV", "ETN", "AON", "ITW", "PNC", "CME",
              "EQIX", "LRCX", "WM", "CCI", "USB", "NSC", "SHW", "EMR", "ICE", "GD", "MU", "EL", "TFC", "KLAC", "DG",
              "FCX", "ATVI", "MCK", "MPC", "PXD", "ORLY", "HCA", "ADM", "GM", "D", "GIS", "SNPS", "SRE", "PSX", "AEP",
              "MET", "VLO", "AIG", "CNC", "KMB", "EW", "APH", "OXY", "AZO", "F", "ROP", "PSA", "CDNS", "DXCM", "A",
              "TRV", "JCI", "MCO", "MSI", "CTVA", "NXPI", "EXC", "BIIB", "ENPH", "ADSK", "AFL", "MAR", "FIS", "ROST",
              "SYY", "O", "CMG", "FDX", "WMB", "AJG", "LHX", "MCHP", "TT", "MNST", "DVN", "CTAS", "XEL", "NEM", "SPG",
              "HES", "IQV", "STZ", "MSCI", "PAYX", "TEL", "PH", "YUM", "PRU", "ALL", "DOW", "HLT", "ECL", "NUE", "KMI",
              "CARR", "PCAR", "HSY", "HAL", "COF", "DD", "IDXX", "ED", "CMI", "CHTR", "AMP", "FTNT", "OTIS", "BK", "EA",
              "VICI", "KHC", "MTD", "AME", "TDG", "CSGP", "KEYS", "KDP", "RMD", "ILMN", "WELL", "ANET", "SBAC", "WEC",
              "PEG", "DLTR", "PPG", "BKR", "ROK", "KR", "CEG", "STT", "ES", "OKE", "WBA", "CTSH", "DLR", "ON", "DHI",
              "ALB", "AWK", "ABC", "FAST", "VRSK", "RSG", "IT", "DFS", "ZBH", "WTW", "IFF", "GPN", "ODFL", "CPRT",
              "PCG", "GWW", "APTV", "BAX", "GPC", "EIX", "MTB", "TROW", "URI", "HPQ", "HIG", "CDW", "CBRE", "FANG",
              "GLW", "VMC", "EFX", "TSCO", "ACGL", "WY", "ULTA", "ETR", "LEN", "FTV", "AVB", "AEE", "EBAY", "FE", "DTE",
              "FITB", "FRC", "LUV", "ARE", "PPL", "MLM", "CAH", "MKC", "DAL", "IR", "LYB", "LH", "RJF", "ANSS", "NDAQ",
              "WAT", "HPE", "PWR", "EQR", "WBD", "HBAN", "PFG", "RF", "CTRA", "EXR", "XYL", "CHD", "MOH", "EPAM", "DOV",
              "VRSN", "AES", "CFG", "CNP", "CF", "HOLX", "CAG", "K", "WAB", "TDY", "AMCR", "NTRS", "STE", "TSN", "CMS",
              "VTR", "MAA", "DGX", "EXPD", "PKI", "CLX", "MRO", "IEX", "DRI", "ABMD", "INVH", "SEDG", "WST", "SJM",
              "CINF", "ETSY", "BBY", "OMC", "BALL", "MPWR", "TRGP", "BR", "ATO", "MOS", "FMC", "COO", "FSLR", "TTWO",
              "KEY", "J", "LVS", "INCY", "FDS", "WRB", "PAYC", "SYF", "TXT", "JBHT", "AVY", "ALGN", "IRM", "APA",
              "SWKS", "NVR", "LDOS", "HWM", "EVRG", "GRMN", "LKQ", "TER", "LNT", "FLT", "AKAM", "ESS", "TYL", "PEAK",
              "NTAP", "EQT", "HRL", "CBOE", "RE"]

# Convert the tickers list into a 20x20 matrix
tickersReshaped = np.reshape(tickersTmp, (len(tickersTmp) // 20, len(tickersTmp) // 20))

# Function to save the user's tickers selection
tickers = []


def pick(picked):

    if picked not in tickers:
        tickers.append(picked)


# Functions to save the user's inputs on Alpha and max loss
# Set initial values that will NOT pass the input check if the user does not press enter (see below)
entry1 = -1


def f_alpha(entry):
    global entry1
    entry1 = entry.widget.get()


entry2 = -1


def f_var(entry):
    global entry2
    entry2 = entry.widget.get()


# Function to exit the selection stage
def closewindow(window):
    window.destroy()


# Loop function to ensure that user's inputs are correct
checking = 0
while checking == 0:
    # Window creation for user interaction
    root = Tk()
    root.title('In case the window opens again, please check that you followed the directions correctly')

    # Customize window size and create the frame inside it, in which the widgets will be
    root.geometry('1500x700')
    frame = Frame(root)
    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)
    frame.grid(row=0, column=0, sticky="news")
    grid = Frame(frame)
    grid.grid(sticky="news", column=0, row=7, columnspan=2)
    frame.rowconfigure(7, weight=1)
    frame.columnconfigure(0, weight=1)

    # Widgets creation
    name_var1 = StringVar()
    name_var2 = StringVar()

    # Create label widget for informing user about tickers selection
    lbl = Label(frame, width=15, height=3, text='Please pick stocks by\nclicking on the respective tickets')

    # Position the widget inside the window
    lbl.grid(column=0, row=0, columnspan=3, sticky="news")

    # Create label widget for informing user about significance request
    lbl = Label(frame, width=15, height=3,
                text='Please enter the significance level alpha for the loss\n'
                     'function as a number from 0 to 100 (%) and press'
                     ' Enter.\n'
                     '(e.g. 5 corresponds to a max loss with probability of 5%)')

    # Position the widget inside the window
    lbl.grid(column=3, row=0, columnspan=5, sticky="news")

    # Create entry widget to input the significance level as entered by the user
    entry = Entry(frame, textvariable=name_var1, font=('calibre', 10, 'normal'))

    # Position the widget inside the window
    entry.grid(column=8, row=0, columnspan=2, sticky="news")

    # Store the significance level entered in the widget
    entry.bind("<Return>", f_alpha)

    # Create label widget for informing user about maximum loss request
    lbl = Label(frame, width=15, height=3,
                text='Please enter the maximum loss that you\n'
                     'can accept as an integer from 0 to 100 (%)\n '
                     'and press Enter')

    # Position the widget inside the window
    lbl.grid(column=10, row=0, columnspan=5, sticky="news")

    # Create entry widget to input the maximum loss as entered by the user
    entry = Entry(frame, textvariable=name_var2, font=('calibre', 10, 'normal'))

    # Position the widget inside the window
    entry.grid(column=15, row=0, columnspan=2, sticky="news")

    # Store the maximum loss entered in the widget
    entry.bind("<Return>", f_var)

    # Create button widget for proceeding to next stage of the code
    btn = Button(frame, text='Next Step', bg="green", command=lambda: closewindow(root))

    # Position the widget inside the window
    btn.grid(column=17, row=0, columnspan=3, sticky="news")

    # Create 400 Tickers buttons, one for each available stock
    for x in range(len(tickersReshaped)):
        for y in range(len(tickersReshaped[0])):
            btn = Button(frame, text=tickersReshaped[x][y], command=lambda j=tickersReshaped[x][y]: pick(j))

            # Position the widget inside the window
            btn.grid(column=x, row=y + 1, sticky="news")
    frame.columnconfigure(tuple(range(len(tickersReshaped))), weight=1)
    frame.rowconfigure(tuple(range(len(tickersReshaped) + 1)), weight=1)

    root.mainloop()

    # Check for correctness of input variables
    if 0 < int(entry1) <= 100 and 0 <= int(entry2) <= 100 and len(tickers) > 0:
        checking = 1

significance = int(entry1)
max_loss = int(entry2)

# Download the daily returns data for the stocks
returns = get_stock_data(tickers)

""" Mean-Variance Module """

# Change Return Data to Numpy Array and Define Dimensionality
R = np.array(returns)
T, K = R.shape

# Get Risk-free rate for user input and calculate Value-at-Risk and Z-statistic
RisklessRate = get_risk_free()
VaR = max_loss / 100
Confidence = 1 - (significance / 100)
Zstat = st.norm.ppf(1 - Confidence)

# Calculate necessary first moments and covariance matrix
Mu = returns.mean() * 252
Mu = np.array(Mu) - RisklessRate
Sigma = returns.cov() * np.sqrt(252)
Sigma = np.array(Sigma)

# Get volatilities from covariance matrix
Volatilities = np.sqrt(np.diag(Sigma))

# Solve for inverse covariance matrix and define a vector of ones
SigmaInv = np.linalg.inv(Sigma)
Ones = np.ones((K, 1))

# Apply analytical solution for Global Minimum Variance Portfolio (GMVP)
GMVP = np.array(np.matmul(SigmaInv, Ones) / (np.matmul(Ones.T, np.matmul(SigmaInv, Ones))))

# Apply analytical solution for Maximum Sharpe Ratio Portfolio (Tangency)
Tangency = np.array(np.matmul(SigmaInv, Mu) / (np.matmul(Ones.T, np.matmul(SigmaInv, Mu))))
Tangency = Tangency.reshape((K, 1))

# Excess expected return and variance of Global Minimum Variance Portfolio
ExcessGMVPReturn = np.matmul(Mu, GMVP)
GMVPVariance = np.matmul(GMVP.T, np.matmul(Sigma, GMVP))

# Excess expected return, variance, and sharpe ratio of Tangency Portfolio
ExcessTangencyReturn = np.matmul(Mu, Tangency)
TangencyVariance = np.matmul(Tangency.T, np.matmul(Sigma, Tangency))
TangencySharpe = ExcessTangencyReturn / np.sqrt(TangencyVariance)

# Apply analytical solution for weights in Tangency portfolio (and risk-free asset) based on VaR and Z-statistic
RiskyWeight = (VaR - RisklessRate) / (ExcessTangencyReturn - Zstat * np.sqrt(TangencyVariance))

# Apply analytical solution for implied risk-aversion given RiskyWeight
RiskAversion = ExcessTangencyReturn / (TangencyVariance * RiskyWeight)

# Derive Returns and volatility of maximum utility portfolio
UtilityReturn = RisklessRate + RiskyWeight[0] * ExcessTangencyReturn
UtilityVolatility = RiskyWeight[0] * np.sqrt(TangencyVariance)

# Apply the analytical solution for maximum utility
MaximumUtility = UtilityReturn - 0.5 * RiskAversion * UtilityVolatility ** 2

# Calculate the weights of risky assets in maximum utility portfolio 
OptimalPortfolio = (RiskyWeight * Tangency).tolist()
RiskfreeWeight = (1 - RiskyWeight).tolist()

# Append weights in  risk-free asset to get maximum utility portfolio
OptimalPortfolio.append(RiskfreeWeight[0])
OptimalPortfolio = np.array(OptimalPortfolio).reshape((K + 1, 1))

# Extend tickers by adding 'risk-free rate' to columns of tickers
ticks = tickers[:]
ticks.append("risk-free rate")
ticks = np.array(ticks).reshape((K + 1, 1))

# Illustrate Maximum Utility Portfolio as dataframe with tickers and weights
UtilityPortfolio = pd.DataFrame(ticks, columns=["Tickers"])
UtilityPortfolio["Weights"] = OptimalPortfolio
UtilityPortfolio = UtilityPortfolio.set_index("Tickers")

# Plot Efficient Frontier
x_axis, y_axis = [], []
weight_range = np.linspace(-10, 10, num=1000)

# Generate Markowitz Bullet Frontier
for x in weight_range:
    # Use fact that Markowitz bullet is closed under linear transformations
    Portfolio = x * GMVP + (1 - x) * Tangency

    # Calcualte portfolio returns and Variance
    PortfolioReturns = np.matmul((Mu + RisklessRate), Portfolio)
    PortfolioVariance = np.matmul(Portfolio.T, np.matmul(Sigma, Portfolio))

    # Define the (x,y)-coordinates of Markowitz bullet as list 
    y_axis.append(PortfolioReturns.tolist())
    x_axis.append(np.sqrt(PortfolioVariance).tolist())

# Flatten lists
mean_axis = [item for sublist in y_axis for item in sublist]
variance_axis = [item for sublist in x_axis for item in sublist]

"""" Frontier Plot """


# Generate plot of Efficient Frontier, Maximal Utility Function, and Capital Allocation Line
def frontierplot(color_mv, color_tobin, nums):
    # Format axes as percentages
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1))

    # Set attributes of the plot
    plt.xlabel('Volatility', fontsize=13)
    plt.ylabel('Mean return', fontsize=13)
    plt.title(label="Mean-Volatility Frontiers", fontsize=13)

    # Delimit scale to 2 times Tangency portfolio return and volatility
    plt.xlim(0, 2 * np.sqrt(TangencyVariance))
    plt.ylim(0, 2 * (ExcessTangencyReturn + RisklessRate))
    plt.plot(variance_axis, mean_axis, color=str(color_mv), label='MV Frontier')

    # Apply Tobin Separation theorem to plot the Capital Allocation Line
    tobinX = np.array(np.linspace(0, 5, num=nums)).reshape((1, nums))
    tobinX = [item for sublist in tobinX for item in sublist]
    tobinY = RisklessRate + tobinX * TangencySharpe
    tobinY = [item for sublist in tobinY for item in sublist]
    plt.plot(tobinX, tobinY, color=str(color_tobin))

    # Plot Global Minimum Variance Portfolio
    plt.plot(np.sqrt(GMVPVariance), ExcessGMVPReturn + RisklessRate, 'o', color="red")

    # Plot Tangency Portfolio
    plt.plot(np.sqrt(TangencyVariance), ExcessTangencyReturn + RisklessRate, 'o', color="red")

    # Plot Maximum Utility Portfolio 
    plt.plot(UtilityVolatility, UtilityReturn, 'o', color="blue")

    # Calculate and plot the Maximum Utility Function
    retU = 0.5 * RiskAversion[0] * np.square(tobinX) + MaximumUtility[0]
    plt.plot(tobinX, retU, '--', color=str(color_mv))

    # Plot individual stocks
    plt.plot(Volatilities, Mu + RisklessRate, 'o', color="purple")

    # Add Legend and put it in the upper right 
    legend_elements = [Line2D([0], [0], color=str(color_mv), lw=3,
                              label='Efficient Frontier'),
                       Line2D([0], [0], color=str(color_tobin), lw=3,
                              label='Capital Allocation Line'),
                       Line2D([0], [0], color=str(color_mv), lw=1, linestyle='--',
                              label='Utility Function')]
    # Set legend to top-right
    plt.legend(handles=legend_elements, loc='upper right')
    return plt.savefig('Mean_Variance.png')


frontierplot('black', 'tab:blue', nums=4000)

"""" Historical returns plot """


# Plot historical return performance
def Hist_Returns(rets, riskfree, port):
    # Convert riskfree rate to daily, copy return dataframe, and add risk-free returns to data frame
    rf = (1 + riskfree) ** (1 / 252) - 1
    returns2 = rets
    returns2['Riskfree Rate'] = rf

    # Calculate Portfolio Performance and add to return dataframe
    PortfolioPerformance = np.matmul(np.array(returns2), np.array(port))
    returns2['Utility Portfolio'] = PortfolioPerformance

    # Plot cumulative returns of all time-series
    ((returns2 + 1).cumprod()).plot()

    # Set attributes of the plot
    plt.ylabel('Cumulative return', fontsize=13)
    plt.xlabel('Date', fontsize=13)
    return plt.savefig('Historical_Returns.png')


Hist_Returns(returns, RisklessRate, UtilityPortfolio)

"""" Stock Description """


# Function to scrape the description of a stock ticker from Yahoo Finance
def get_stock_description(ticker):
    # Build the URL for the Yahoo Finance page for the ticker
    url = f'https://finance.yahoo.com/quote/{ticker}'

    # Make a request to the URL
    response = requests.get(url)

    # Parse the HTML of the page
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the element containing the description
    description_element = soup.find('p', {'class': 'businessSummary Mt(10px) Ov(h) Tov(e)'})

    # Extract the text of the description
    description = description_element.text
    return description


"""" PDF Creation """

# Reshape tickers for Cover Page Entry
tickers_for_cover_page = ', '.join(tickers)

# Create pdf to present results
pdf = FPDF("P", "cm", "A4")
pdf.add_page()

# Write on the Cover Page
# Set font size and create cell for main header
pdf.set_font("Helvetica", size=30)
pdf.cell(ln=1, h=12, align='C', w=0, txt="Your Personal Investment Report", border=0, fill=False)

# List the portfolio components the user selected
pdf.set_font("Helvetica", size=18)
pdf.write(1.0, "Portfolio Components:\n")
pdf.set_font("Helvetica", size=18)
pdf.write(1.0, f"{tickers_for_cover_page}")
pdf.ln(2)

# List the significance level and acceptable loss
pdf.set_font("Helvetica", size=18)
pdf.write(1.0, f"Your significance level: {significance}%\n")
pdf.write(1.0, f"Your maximum loss acceptable: {max_loss}%")
pdf.ln(6)

# List the names of the people who created the file
pdf.set_font("Helvetica", size=16)
pdf.write(1.5, "© Michal Glinicki, Andreas Karlsson, Nikolaos Karavasilis, Stefan Koehler")

# Print the HSG-Logo on the page
pdf.image('Uni_logo.png', x=11, y=26, w=8, h=2, type='', link='')

# Write on the second page
# Set Cell of Header
pdf.set_left_margin(0)
pdf.add_page()
pdf.set_font("Helvetica", size=22)
pdf.set_fill_color(52, 138, 85)
pdf.set_text_color(255, 255, 255)
pdf.cell(ln=1, h=1.5, align='', w=0, txt="   Company Descriptions", border=0, fill=True)

# Write the description of the information on the page
pdf.set_text_color(0)
pdf.set_font("Helvetica", size=11)
pdf.set_left_margin(0.7)
pdf.ln(0.5)
pdf.write(0.5,
          "The following pages contain descriptions of the companies you are interested in, "
          "giving you a proper understanding of the related industries and business models.")
pdf.ln(1)

# Loop over the tickers to get the description for each company and write them on the page
for ii in range(len(tickers)):
    pdf.set_font("Helvetica", "B", 12)
    pdf.write(1, f"{ii + 1}) {tickers[ii]}'s Company Profile:\n")
    pdf.set_font("Helvetica", "", 11)
    pdf.write(0.5, txt=f"{get_stock_description(tickers[ii])}")
    pdf.ln(1)

# Write on the third page
# Set Cell of Header
pdf.set_left_margin(0)
pdf.add_page()
pdf.set_font("Helvetica", size=22)
pdf.set_fill_color(52, 138, 85)
pdf.set_text_color(255, 255, 255)
pdf.cell(ln=1, h=1.5, align='', w=0, txt="   Summary Statistics", border=0, fill=True)

# Set back to body text
pdf.set_text_color(0)
pdf.set_font("Helvetica", size=11)
pdf.set_left_margin(0.7)
pdf.ln(0.5)
pdf.write(0.5, "The following page contains summary statistics for the different stocks.")
pdf.ln(0.5)

# Loop over tickers and for each ticker create a table with descriptive statistics:
for i in range(len(tickers)):
    pdf.set_font("Helvetica", "B", 12)

    # Condition so that the table does not go out of page bounds
    if pdf.get_y() >= 24:
        pdf.add_page()

    # Subheader with the company's ticker
    pdf.write(1, f"\n{tickers[i]}'s Company Summary Statistics:\n")
    pdf.set_font("Helvetica", "", 11)

    # Calculate descriptive statistics for each stock
    daily_return = R[:, i].mean() * 100  # Times 100 to turn it into percentage
    daily_volatility = R[:, i].std() * 100  # Times 100 to turn it into percentage
    annual_return = daily_return * 252  # Multiplying by 252 for number of trading days
    annual_volatility = daily_volatility * np.sqrt(252)  # Multiplying by square root of 252 for number of trading days
    sharpe = (annual_return - get_risk_free()) / annual_volatility
    max_ret = max(R[:, i]) * 100
    min_ret = min(R[:, i]) * 100

    # Create the table
    y = pdf.get_y()
    x = pdf.get_x()

    # Set width and heights of the columns
    w = 2.5
    h = 0.75

    # Create the table with setting each next cell one width further
    # Title cells
    pdf.set_y(y)
    pdf.multi_cell(w=w, h=h, txt="Daily \nReturns (%)", border=1)
    pdf.set_y(y)
    pdf.set_x(x + w)
    pdf.multi_cell(w=w, h=h, txt="Daily \nVolatility (%)", border=1)
    pdf.set_y(y)
    pdf.set_x(x + 2 * w)
    pdf.multi_cell(w=w, h=h, txt="Annual \nReturns (%)", border=1)
    pdf.set_y(y)
    pdf.set_x(x + 3 * w)
    pdf.multi_cell(w=w, h=h, txt="Annual \nVolatility (%)", border=1)
    pdf.set_y(y)
    pdf.set_x(x + 4 * w)
    pdf.multi_cell(w=w, h=h, txt="Max \nReturns (%)", border=1)
    pdf.set_y(y)
    pdf.set_x(x + 5 * w)
    pdf.multi_cell(w=w, h=h, txt="Min \nReturns (%)", border=1)
    pdf.set_y(y)
    pdf.set_x(x + 6 * w)
    pdf.multi_cell(w=w, h=h, txt="Sharpe \nRatio", border=1)

    # Output cells
    pdf.set_xy(x, y + 2 * h)
    pdf.cell(w=w, h=h, txt=str(round(daily_return, 3)), border=1)
    pdf.cell(w=w, h=h, txt=str(round(daily_volatility, 3)), border=1)
    pdf.cell(w=w, h=h, txt=str(round(annual_return, 3)), border=1)
    pdf.cell(w=w, h=h, txt=str(round(annual_volatility, 3)), border=1)
    pdf.cell(w=w, h=h, txt=str(round(max_ret, 3)), border=1)
    pdf.cell(w=w, h=h, txt=str(round(min_ret, 3)), border=1)
    pdf.cell(w=w, h=h, txt=str(round(sharpe, 3)), border=1)

# Write on fourth page
# Create cell for header
pdf.set_left_margin(0)
pdf.add_page()
pdf.set_font("Helvetica", size=22)
pdf.set_fill_color(52, 138, 85)
pdf.set_text_color(255, 255, 255)
pdf.cell(ln=1, h=1.5, align='', w=0, txt="   Portfolio Analysis", border=0, fill=True)

# Insert description on what is displayed on the page
pdf.set_text_color(0)
pdf.set_font("Helvetica", size=11)
pdf.set_left_margin(0.7)
pdf.ln(0.5)
pdf.write(0.5,
          "The following page contains information about various portfolios that can be generated"
          " with the stocks you choose. This includes an Optimal Risky Portfolio which maximizes "
          "the risk adjusted return for the investor without any leverage. For less risk-seeking investors "
          "there is the Minimum Variance Portfolio. This portfolio represents the least volatile combination"
          " of risky assets. The most important one, however, is the Maximum Utility Portfolio which "
          "maximizes your utility according to your inputs.")
pdf.ln(1)

# Insert the Graph
pdf.image('Mean_Variance.png', x=0.5, y=6, w=21, h=12, type='', link='')
pdf.ln(14)

# Insert information on the three portfolios
# Optimal Risky Portfolio
pdf.set_font("Helvetica", "B", 12)
pdf.write(0.5, "Optimal Risky Portfolio:\n")
pdf.set_font("Helvetica", size=11)
pdf.write(0.5, f"Return: {np.round(ExcessTangencyReturn[0] * 100, decimals=2)}%\n")
pdf.write(0.5, f"Volatility: {np.round(np.sqrt(TangencyVariance[0, 0]) * 100, decimals=2)}%\n")
pdf.ln(1)

# Global Minimum Variance Portfolio
pdf.set_font("Helvetica", "B", 12)
pdf.write(0.5, "Minimum Variance Portfolio:\n")
pdf.set_font("Helvetica", size=11)
pdf.write(0.5, f"Return: {np.round(ExcessGMVPReturn[0] * 100, decimals=2)}%\n")
pdf.write(0.5, f"Volatility: {np.round(np.sqrt(GMVPVariance[0, 0]) * 100, decimals=2)}%\n")
pdf.ln(1)

# Maximum Utility Portfolio
pdf.set_font("Helvetica", "B", 12)
pdf.write(0.5, "Maximum Utility Portfolio:\n")
pdf.set_font("Helvetica", size=11)
pdf.write(0.5, f"Return: {np.round(UtilityReturn[0] * 100, decimals=2)}%\n")
pdf.write(0.5, f"Volatility: {np.round(UtilityVolatility[0, 0] * 100, decimals=2)}%\n")

# Write on fifth page
# Create Cell for header
pdf.set_left_margin(0)
pdf.add_page()
pdf.set_font("Helvetica", size=22)
pdf.set_fill_color(52, 138, 85)
pdf.set_text_color(255, 255, 255)
pdf.cell(ln=1, h=1.5, align='', w=0, txt="   Portfolio Weights", border=0, fill=True)

# Insert description for what the page contains
pdf.set_text_color(0)
pdf.set_font("Helvetica", size=11)
pdf.set_left_margin(0.7)
pdf.ln(0.5)
pdf.write(0.5,
          "The following page contains the weighting of the different "
          "stocks in order to realize the portfolio exposure displayed on the previous page. "
          "Negative values mean short positions in the respective companies "
          "or borrowing for the risk-free rate.")
pdf.ln(1)

# Format Input for Table of weights
GMVP = GMVP.tolist()
GMVP2 = []
Tangency = Tangency.tolist()
Tangency2 = []
OptimalPortfolio = OptimalPortfolio.tolist()
OptimalPortfolio2 = []

# Loop over vectors to remove the []
for zz in range(len(tickers)):
    GMVP2 = GMVP2 + GMVP[zz]
    Tangency2 = Tangency2 + Tangency[zz]
    OptimalPortfolio2 = OptimalPortfolio2 + OptimalPortfolio[zz]

# Add the risk-free rate weight since this is not covered by the loop
OptimalPortfolio2 = OptimalPortfolio2 + OptimalPortfolio[len(OptimalPortfolio) - 1]

# Create the table
y = pdf.get_y()
x = pdf.get_x()

w1 = 2
w = 5.5
h = 0.75

# Create the first row which includes the headers
pdf.set_font("Helvetica", size=10)
pdf.set_y(y)
pdf.multi_cell(w=w1, h=h, txt="Stock", border=1)
pdf.set_y(y)
pdf.set_x(x + w1)
pdf.multi_cell(w=w, h=h, txt="Optimal Risky Portfolio" + " (%)", border=1)
pdf.set_y(y)
pdf.set_x(x + w1 + w)
pdf.multi_cell(w=w, h=h, txt="Minimum Variance Portfolio" + " (%)", border=1)
pdf.set_y(y)
pdf.set_x(x + w1 + 2 * w)
pdf.multi_cell(w=w, h=h, txt="Maximum Utility Portfolio" + " (%)", border=1)
pdf.set_font("Helvetica", size=11)

# Loop over the tickers to get the data on the specific stock and create additional cells for the table
for jj in range(len(tickers)):
    h1 = h * (jj + 1)
    pdf.set_xy(x, y + h1)
    pdf.cell(w=w1, h=h, txt=str(tickers[jj]), border=1)
    pdf.set_xy(x + w1, y + h1)
    pdf.cell(w=w, h=h, txt=str(round(Tangency2[jj] * 100, 2)), border=1)
    pdf.set_xy(x + w1 + w, y + h1)
    pdf.cell(w=w, h=h, txt=str(round(GMVP2[jj] * 100, 2)), border=1)
    pdf.set_xy(x + w1 + 2 * w, y + h1)
    pdf.cell(w=w, h=h, txt=str(round(OptimalPortfolio2[jj] * 100, 2)), border=1)

# Add a last row for the risk-free rate
h1 = h1 + h
pdf.set_xy(x, y + h1)
pdf.set_font("Helvetica", size=9)
pdf.cell(w=w1, h=h, txt="Risk free", border=1)
pdf.set_font("Helvetica", size=11)
pdf.set_xy(x + w1, y + h1)
pdf.cell(w=w, h=h, txt=str("0.00"), border=1)
pdf.set_xy(x + w1 + w, y + h1)
pdf.cell(w=w, h=h, txt=str("0.00"), border=1)
pdf.set_xy(x + w1 + 2 * w, y + h1)
pdf.cell(w=w, h=h, txt=str(round(OptimalPortfolio2[len(OptimalPortfolio2) - 1] * 100, 2)), border=1)

# Write on the sixth page
# Create Cell for Header
pdf.set_left_margin(0)
pdf.add_page()
pdf.set_font("Helvetica", size=22)
pdf.set_fill_color(52, 138, 85)
pdf.set_text_color(255, 255, 255)
pdf.cell(ln=1, h=1.5, align='', w=0, txt="   Portfolio Performance", border=0, fill=True)

# Write the description on what is displayed on the page
pdf.set_text_color(0)
pdf.set_font("Helvetica", size=11)
pdf.set_left_margin(0.7)
pdf.ln(0.5)
pdf.write(0.5,
          "The following graph shows the performance of the individual stocks as well as overall "
          "performance of the chosen market portfolio.")
pdf.ln(1)

# Insert the graph
pdf.image('Historical_Returns.png', x=0.5, y=4, w=21, h=12, type='', link='')

# Safe the PDF file
pdf.output("Investment_Report.pdf")

# Open the PDF file
path = "Investment_Report.pdf"
os.system(path)
