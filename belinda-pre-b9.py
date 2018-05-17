#extracting orders dictionary to multiple columns
#record EL, ES
#with empyrical module added
#long and short trailing stops
#long and short positions
#long positions only ok
#0.2% comission on maker orders, long orders close update
#trailing stop update
#with context commission, slippage and Ta-lib EMA calculatons, extra saving results to csv file (not pickle-only)
#with trailing stop from https://www.quantopian.com/posts/trailing-stop-loss
#with sizes
#1. catalyst ingest-exchange -x bitfinex -i btc_usd -f minute
#2. catalyst run -f belinda-pre-b1.py -x bitfinex -s 2018-4-28 -e 2018-4-30 --capital-base 1000 --base-currency usd --data-frequency minute -o out.pickle
#3. python print_results.py (or process csv file instead of the pickle)



#import some essential Python libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from logbook import Logger
from empyrical import max_drawdown, alpha_beta, tail_ratio, sharpe_ratio

from catalyst import run_algorithm
from catalyst.api import (record, symbol, order_target_percent, order, )
from catalyst.exchange.utils.stats_utils import extract_transactions
import talib



#define namespace for proper work according to Python scripting rules
NAMESPACE = 'belinda-pre'
log = Logger(NAMESPACE)

#calls the initialize() function and passes in a context variable. context is a persistent namespace to store variables to access from one algorithm iteration to the next
def initialize(context):    
    context.i = 0
    context.asset = symbol('btc_usd') #set btc_usd trading pair
    context.base_price = None
   
    
    #0.2% commission on maker orders, and 0.1% taker orders and slippage
    context.set_commission(maker=.001, taker=.002)
    #context.set_slippage(spread=0.0000001)
    context.set_slippage(spread=0)
    
    context.stop_price = 0
    context.stop_price_short = 0
    context.risk = 0.01

def set_trailing_stop(context, data, stop):
    if (context.portfolio.positions[context.asset].amount >0): #find evaluation. boolean in python
        #price = data[context.asset].price

        price = data.current(context.asset, 'price')        
        context.stop_price = max(context.stop_price, (price - stop)) 
        #context.stop_price_short = min(context.stop_price_short, (price + stop)) 
        #print("price = ", price)

def set_trailing_stop_short(context, data, stop):
    if (context.portfolio.positions[context.asset].amount < 0): #find evaluation. boolean in python //<= 0 before
        #price = data[context.asset].price
        price = data.current(context.asset, 'price')        
        context.stop_price_short = min(context.stop_price_short, (price + stop)) 
        #print("price = ", price)
         
    
#catalyst calls the handle_data() function on each iteration, for example, once every minute. On every iteration, handle_data() passes the same context variable and an event-frame called data containing the current trading bar with open, high, low, and close (OHLC) prices and volume
def handle_data(context, data):
    short_window = 5 #value obtained from Belinda script
    long_window = 24 #value obtained from Belinda script
    
    
    context.i += 1
    if context.i < long_window:
        return
    
    #calculate EMA with Talib
    my_short_stock_series = data.history(context.asset, 'price', bar_count = short_window, frequency = "1T",)
    ema_short_result = talib.EMA(my_short_stock_series, timeperiod=short_window)
    short_ema = ema_short_result[-1]
    
    #long ema calculations is corrected based on upwork developer comment
    my_long_stock_series = data.history(context.asset, 'price', bar_count = long_window+1, frequency = "1T",)
    ema_long_result = talib.EMA(my_long_stock_series, timeperiod=long_window)
    long_ema = ema_long_result[-1]
            
    
    price = data.current(context.asset, 'price')
    

    
    if context.base_price is None:
        context.base_price = price
    price_change = (price - context.base_price)/context.base_price
    
    
    record(price = price, cash = context.portfolio.cash, price_change = price_change, short_ema = short_ema, long_ema = long_ema)
    

    pos_amount = context.portfolio.positions[context.asset].amount
    
    
  
    orders = context.blotter.open_orders
    if len(orders) > 0:
        return      
    if not data.can_trade(context.asset):    
        return

    
    
    
    
    
    #sizing calculations   

    #getting high, low, close value from history to talib.ATR function
    highs = data.history(context.asset, 'high', bar_count = 300, frequency = "1T",).dropna()
    lows = data.history(context.asset, 'low', bar_count = 300, frequency = "1T",).dropna()
    closes = data.history(context.asset, 'close', bar_count = 300, frequency = "1T",).dropna()

    
    atr = talib.ATR(highs, lows, closes) #talib.ATR output is an array
    stop_loss_atr = atr[-1] #talib.ATR output is an array
    stop = stop_loss_atr
    #print("stop_loss_atr=", stop_loss_atr)
    #position size calculation based on current equity, risk and trailing stop    
    size = context.portfolio.portfolio_value*context.risk/stop
    
    trading_avaliable = context.portfolio.portfolio_value/0.3/price
    size_aval = size if size<trading_avaliable else trading_avaliable     #equal to PineScript statement, size_aval = size<trading_avaliable?size:trading_avaliable
   
    #print("size_aval=", size_aval)
     
    #trailing stop call. obtaining context.stop_price
    set_trailing_stop(context, data, stop) #maybe trailing stop should be calculated when order start
    set_trailing_stop_short(context, data, stop)
    #print("price=", price, "context.stop_price=", context.stop_price,"context.stop_price_short=", context.stop_price_short)

    #check for the long stop price and long orders are open
    if (pos_amount > 0):
        if price < context.stop_price:        
            order_target_percent(context.asset, 0)
            context.stop_price = 0
            #print("trailing stop")

    if (pos_amount < 0):
        if price > context.stop_price_short:        
            order_target_percent(context.asset, 0)
            context.stop_price_short = 0        
            #print("trailing stop short")
        
    EL = short_ema > long_ema and pos_amount <=0
    ES = short_ema < long_ema and pos_amount > 0
    
    
    record(EL=EL, ES=ES)
    
    if EL:
        order_target_percent(context.asset, 0) #close short asset
        order(context.asset, size_aval) #open long asset
        #print("long")
    elif ES:
        order_target_percent(context.asset, 0) #close long asset
        order(context.asset, -size_aval) #open short asset
        #print("short")

def unpack(df, column):
    ret = None

    tmp = pd.DataFrame((d for idx, d in df[column].iteritems()))
    ret = pd.concat([df.drop(column,axis=1), tmp], axis=1)

    return ret

        
def analyze(context, perf): 
    #print("perf.max_drawdown=", perf.max_drawdown)
    empyrical_max_drawdown = max_drawdown(perf.algorithm_period_return)
    print("empyrical_max_drawdown = ", empyrical_max_drawdown)
    
    empyrical_tail_ratio = tail_ratio(perf.algorithm_period_return)
    print("empyrical_tail_ratio = ", empyrical_tail_ratio)
    
    empyrical_sharpe_ratio = sharpe_ratio(perf.algorithm_period_return)
    print("empyrical_sharpe_ratio = ", empyrical_sharpe_ratio)
    
    empyrical_alpha_beta = alpha_beta(perf.algorithm_period_return, perf.benchmark_period_return)
    print("empyrical_alpha_beta = ", empyrical_alpha_beta)    
    
    
    
    #cum_returns(perf)
    # Save results in CSV file
    filename = "csvoutput"
    perf.to_csv(filename + '.csv')      


    filename_orders = "orders_output"    

    perf0=perf[['orders']]
    perf1 = perf0[perf0['orders'].apply(len) > 0]
      
    perf2 = pd.DataFrame(perf1['orders'])
    #perf2 = pd.DataFrame([x for x in perf1['orders']])
    #print(perf1[['orders',-1]].head(n=5))
    #convert list of dictionaries to dictionary
    perf2["ordersd"] = pd.Series(perf2["orders"].str[0])
    print( perf2["ordersd"].head(70))
    
    #extracting orders dictionary to multiple columns
    perf2 = pd.DataFrame([x for x in perf2['ordersd']])
    
    #unpack(perf2, 'ordersd')
    perf2.to_csv(filename_orders + '.csv')    
    
    exchange = list(context.exchanges.values())[0]
    base_currency = exchange.base_currency.upper()
    
    axl = plt.subplot(411)
    perf.loc[:, ['portfolio_value']].plot(ax = axl)
    axl.legend_.remove()
    axl.set_ylabel('Portfolio Value\n({})'.format(base_currency))
    start, end = axl.get_ylim()
    axl.yaxis.set_ticks(np.arange(start, end, (end-start) / 5))
    
    ax2 = plt.subplot(412, sharex = axl)
    perf.loc[:,['price','short_ema','long_ema']].plot(ax = ax2, label = 'Price')
    ax2.legend_.remove()
    ax2.set_ylabel('{asset}\n({base})'.format(asset = context.asset.symbol, base = base_currency))
    start, end = ax2.get_ylim()
    ax2.yaxis.set_ticks(np.arange(start, end, (end-start) / 5))
    
    transaction_df = extract_transactions(perf)
    if not transaction_df.empty:
        buy_df = transaction_df[transaction_df['amount'] > 0]
        sell_df = transaction_df[transaction_df['amount'] < 0]
        ax2.scatter(buy_df.index.to_pydatetime(), perf.loc[buy_df.index, 'price'], marker = '^', s = 100, c = 'green', label = '')
        ax2.scatter(sell_df.index.to_pydatetime(), perf.loc[sell_df.index, 'price'], marker = 'v', s = 100, c = 'red', label = '')    
    
    
    ax3 = plt.subplot(413, sharex = axl)
    perf.loc[:,['algorithm_period_return', 'price_change']].plot(ax = ax3)
    ax3.legend_.remove()
    ax3.set_ylabel('Percent Change')
    start, end = ax3.get_ylim()
    ax3.yaxis.set_ticks(np.arange(0, end, (end-start) / 5))
    
    
    
    
    ax4 = plt.subplot(414, sharex = axl)
    perf.cash.plot(ax = ax4)
    ax4.set_ylabel('Cash\n({})'.format(base_currency))
    start, end = ax4.get_ylim()
    ax4.yaxis.set_ticks(np.arange(0, end, end / 5))
    
    
    
    plt.show()
    
if __name__ == '__main__':
    

    run_algorithm(
            capital_base = 10000,
            data_frequency = 'minute',
            initialize = initialize,
            handle_data = handle_data,
            analyze = analyze,
            exchange_name = 'bitfinex',
            algo_namespace = NAMESPACE,
            base_currency = 'usd',
            start = pd.to_datetime('2018-5-11', utc = True),
            end = pd.to_datetime('2018-5-11', utc = True),        
        )
    
    
    
    
    
    
    
    
    
