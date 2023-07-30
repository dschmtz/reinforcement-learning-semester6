import pandas as pd
import numpy as np


class StockMarket:
    """
    A class representing a simulated stock market that keeps track of various stocks.

    Attributes:
        stocks (dict): A dictionary containing stocks with their symbols as keys.

    Methods:
        __init__(): Initializes an empty stock market with an empty dictionary for stocks.
        add_stock(stock): Adds a new stock to the market.
        remove_stock(symbol): Removes a stock from the market using its symbol.
        get_stock(symbol): Retrieves a stock from the market using its symbol.
        update_prices(): Updates the market prices of all stocks.
    """
    
    def __init__(self):
        self.stocks = {}
        self.time_offset = 0

    def add_stock(self, stock):
        self.stocks[stock.symbol] = stock

    def remove_stock(self, symbol):
        if symbol not in self.stocks:
            raise ValueError("Stock not found in the market.")
        del self.stocks[symbol]

    def get_stock(self, symbol):
        if symbol not in self.stocks:
            raise ValueError("Stock not found in the market.")
        return self.stocks[symbol]
    
    def update_prices(self):
        self.time_offset += 1


class Portfolio:
    """
    A class representing a simulated portfolio that tracks the balance and stocks.
    This class acts as an environment for Reinforcement Learning.

    Attributes:
        init_balance (float): The initial balance of the portfolio.
        balance (float): The current balance of the portfolio.
        stocks (dict): A dictionary containing the stocks held in the portfolio with their symbols as keys.

    Methods:
        __init__(balance=50_000): Initializes a portfolio with an initial balance (default is 50,000).
        reset(): Resets the portfolio to its initial state, setting the balance and stocks to their initial values.
        buy(stock, quantity): Buys a certain quantity of a stock and adds it to the portfolio.
        sell(stock, quantity): Sells a certain quantity of a stock from the portfolio.
        get_value(): Calculates the total value of the portfolio, including balance and stock holdings.
        plot_history(): Plots the history of the portfolio.
        __str__(): Returns a string representation of the portfolio with details of stocks and their quantities.
    """

    def __init__(self, market, balance=100_000):
        self.init_balance = balance
        self.balance = balance
        self.market = market
        self.stocks = {}

    def reset(self, reset_market=True):
        self.balance = self.init_balance
        self.stocks = {}

        if reset_market:
            self.market.time_offset = 0

    def buy(self, stock, quantity):
        stock = self.market.get_stock(stock)

        if quantity < 1:
            raise ValueError("Invalid quantity.")

        price = stock.get_current_price() * quantity

        if self.balance < price:
            raise ValueError("Not enough money.")
        
        self.balance -= price

        if stock.symbol in self.stocks:
            self.stocks[stock.symbol] += quantity
        else:
            self.stocks[stock.symbol] = quantity

    def sell(self, stock, quantity):
        stock = self.market.get_stock(stock)

        if quantity < 1:
            raise ValueError("Invalid quantity.")
        
        if not stock.symbol in self.stocks:
            raise ValueError("Stock not found in portfolio.")
        
        if self.stocks[stock.symbol] < quantity:
            raise ValueError("Not enough shares to sell.")
        
        self.stocks[stock.symbol] -= quantity
        if self.stocks[stock.symbol] == 0:
            del self.stocks[stock.symbol]

        portfolio += stock.get_current_price() * quantity

    def get_value(self):
        total_value = self.balance

        for symbol, quantity in self.stocks.items():
            stock = self.market.get_stock(symbol)
            total_value += stock.get_current_price() * quantity

        return total_value
    
    def get_state(self, window_size=90):
        # Balance, quantity of stocks
        portfolio_state = np.zeros(2)
        portfolio_state[0] = self.balance
        portfolio_state[1] = self.stocks.get("AAPL", 0)

        # Num Stock 1, 7 Day Price
        price_window = np.zeros((window_size))
        price_window = self.market.get_stock("AAPL").get_current_price(n=90)

        min_prices = np.min(price_window, keepdims=True)
        max_prices = np.max(price_window, keepdims=True)
        price_range = max_prices - min_prices

        # Normalising each stock indendently
        price_window = (price_window - min_prices) / price_range
        return portfolio_state, price_window

    def __str__(self):
        string = "Portfolio:\n" + "*" * 16 + "\n"

        for symbol, quantity in self.stocks.items():
            stock = self.market.get_stock(symbol)
            string += f"{stock.name} #{quantity} * {stock.get_current_price()}\n"

        string += "*" * 16
        return string


class Stock:
    """
    A class representing a stock.

    Attributes:
        symbol (str): The symbol of the stock.
        price_history (list): A list containing the historical prices of the stock.

    Methods:
        __init__(symbol, initial_price): Initializes a stock with a symbol and an initial price.
        update_price(new_price): Updates the stock's price history with a new price.
        get_current_price(n=1): Retrieves the current price of the stock (by default, the latest price).
        __str__(): Returns a string representation of the stock, showing its symbol and current price.
    """
    
    def __init__(self, symbol, name, market, initial_price=None, price_history=[]):
        self.symbol = symbol
        self.name = name
        self.market = market

        if initial_price:
            self.price_history = [initial_price]
        else:
            self.price_history = price_history

    def update_price(self, price_list):
        self.price_history += price_list

    def get_current_price(self, offset=None, n=1):
        if not offset:
            offset = self.market.time_offset
        
        if n < 2:
            return self.price_history[offset]
        return self.price_history[offset-n:offset] 

    def __str__(self):
        return f"{self.symbol}: {self.get_current_price()}"



def setup_environment():
    """
    This method sets the environment up by loading historical 
    stock data and initialzing all important classes.
    """
    market = StockMarket()
    stocks = {
        "AAPL": "Apple",
        "LUMN": "Lumen Technologies",
        "PG": "Procter & Gamble"
    }

    # Load historical stock data
    loaded_stocks = pd.read_csv("./data/historically_stocks.csv", index_col=0, header=[0, 1])

    for symbol, name in stocks.items():
        prices = loaded_stocks["Adj Close"][symbol]

        stock = Stock(symbol, name, market, price_history=prices.tolist())
        market.add_stock(stock)

    return Portfolio(market)


if __name__ == "__main__":
    # Testing portfolio creation
    portfolio = setup_environment()

    #portfolio.buy(portfolio.market.get_stock("AAPL"), 1)
    #portfolio.buy(portfolio.market.get_stock("LUMN"), 1)
    #portfolio.buy(portfolio.market.get_stock("PG"), 1)

    portfolio.market.time_offset = 90
    print(portfolio.get_state())
    
    #print(portfolio.get_state()[0].shape)
    #print(portfolio.get_state()[1].shape)

    aapl = portfolio.market.get_stock("AAPL")
    #df = pd.DataFrame(aapl)
    #df.plot()

    print(portfolio, portfolio.balance)