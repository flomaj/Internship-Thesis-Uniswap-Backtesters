import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set variables
decimals = 10**18
q96 = 2**96
q128 = 2**128
starting_base_amount = 1000 * decimals

    # Compute liquidity added to the pool
def compute_liquidity(tL, tH, price0, amount_0, amount_1):
    sqrtp_low = int(math.sqrt(tL) * q96) 
    sqrtp_cur = int(math.sqrt(price0) * q96) 
    sqrtp_upp = int(math.sqrt(tH) * q96) 

    def liquidity0(amount, pa, pb):
        if pa > pb: #if "pa" is greater than "pb"
            pa, pb = pb, pa #the values of "pa" and "pb" will be swapped, and "pa" will now have the original value of "pb" and "pb" will have the original value of "pa".
        return (amount * (pa * pb) / q96) / (pb - pa)
        
    def liquidity1(amount, pa, pb):
        if pa > pb:
            pa, pb = pb, pa
            return amount * q96 / (pb - pa)

    liq0 = liquidity0(amount_0, sqrtp_cur, sqrtp_upp)
    liq1 = liquidity1(amount_1, sqrtp_cur, sqrtp_low)
    myLiq = int(min(liq0, liq1))
    return myLiq

    # Compute fees (the fees generated)
def fees(i, df):
    df.deltaFee0[i] = df.feeGrowth0[i] - df.feeGrowth0[i-1]
    df.deltaFee1[i] = df.feeGrowth1[i] - df.feeGrowth1[i-1]
    df.myFee0[i] = df.myFee0[i-1] + (df.deltaFee0[i] / decimals / q128) * df.myUnbLiq[i]
    df.myFee0in1[i] = df.myFee0[i] * df.price[i]
    df.myFee1[i] = df.myFee1[i-1] + (df.deltaFee1[i] / decimals / q128) * df.myUnbLiq[i]
    df.totalFeeIn1[i] = df.myFee0in1[i] + df.myFee1[i]

def fees_out(i, df):
    df.deltaFee0[i] = 0
    df.deltaFee1[i] = 0
    df.myFee0[i] = df.myFee0[i-1] + (df.deltaFee0[i] / decimals / q128) * df.myUnbLiq[i]
    df.myFee0in1[i] = df.myFee0[i] * df.price[i]
    df.myFee1[i] = df.myFee1[i-1] + (df.deltaFee1[i] / decimals / q128) * df.myUnbLiq[i]
    df.totalFeeIn1[i] = df.myFee0in1[i] + df.myFee1[i]

def output(df):
    # Create the figure and subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))

    # First graph - Top left subplot
    axs[0, 0].plot(df.price, color='blue', label='Price', linestyle='-', linewidth=1.5)
    within_thresholds = np.logical_and(df.price >= df.tL, df.price <= df.tH)
    outside_thresholds = np.logical_not(within_thresholds)
    axs[0, 0].fill_between(range(len(df.price)), df.price, df.tL, where=within_thresholds, color='limegreen', alpha=0.4)
    axs[0, 0].fill_between(range(len(df.price)), df.price, df.tH, where=outside_thresholds, color='red', alpha=0.4)
    axs[0, 0].plot(df.tL, color='grey', label='Threshold Low', linestyle='--', linewidth=1.5)
    axs[0, 0].plot(df.tH, color='grey', label='Threshold High', linestyle='--', linewidth=1.5)
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('Price')
    axs[0, 0].set_title('Price Variation over Time')
    axs[0, 0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    axs[0, 0].legend()

    # Second graph - Top right subplot
    x = range(len(df))
    y1 = np.array(df.positionValue1).astype(np.float)
    y2 = np.array(df.totalFeeIn1)
    colors = ['steelblue', 'limegreen']
    labels = ['Position Value', 'Total Fee']
    axs[0, 1].stackplot(x, y1, y2, colors=colors, labels=labels, alpha=0.5)
    axs[0, 1].plot(df.amountInvested1, label='Total Investment', color='blue')
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('Position Value / Total Fee / Total Investment')
    axs[0, 1].set_title('Position Value, Total Fee, and Total Investment Over Time', fontsize=12)
    axs[0, 1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    axs[0, 1].legend(loc='lower right')

    # Third graph - Bottom subplot
    strategy = np.array(df.positionValue1).astype(np.float) + np.array(df.totalFeeIn1)
    hold = np.array(df.hold1)

    strategy_percentage = (strategy / strategy[0] * 100) - 100
    hold_percentage = (hold / hold[0] * 100) - 100
    axs[1, 0].plot(strategy_percentage, label='Strategy', color='blue', linewidth=2)
    axs[1, 0].plot(hold_percentage, label='Holding', color='grey', linewidth=2)
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Return (%)')
    axs[1, 0].set_title('Comparison of Holding vs. Strategy')
    axs[1, 0].legend()
    axs[1, 0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # Hide unused subplot
    axs[1, 1].axis('off')

    plt.tight_layout()  # Ensures the subplots are properly spaced
    plt.show()


def strategy2(dataset):
    
    # Load dataset
    df = pd.read_csv(dataset)

    # Set initial values for columns in the DataFrame 
    df['tL'] = df.price[0] * 0.5
    df['tH'] = df.price[0] * 1.5
    df['amount0'] = starting_base_amount
    df['amount1'] = (df.amount0[0] / ((1/math.sqrt(df.price[0])) - (1/math.sqrt(df.tH[0])))) * (math.sqrt(df.price[0]) - math.sqrt(df.tL[0]))
    df['amountInvested1'] = df.amount1[0] / decimals
    df['positionValue1'] = ((df.amount0[0] / decimals) * df.price[0]) + (df.amount1[0] / decimals) #value of the LP at each point
    df['deltaE'] = df.amount0[0] * math.sqrt(df.price[0] / df.tL[0]) * (math.sqrt(df.tH[0]) - math.sqrt(df.tL[0])) / (math.sqrt(df.tH[0]) - math.sqrt(df.price[0]))
    df['myUnbLiq'] = compute_liquidity(df.tL[0], df.tH[0], df.price[0], df.amount0[0], df.amount1[0])
    df['deltaFee0'] = 0 
    df['deltaFee1'] = 0 
    df['myFee0'] = 0 
    df['myFee0in1'] = 0 
    df['myFee1'] = 0 
    df['totalFeeIn1'] = 0 # total amount of fees in terms of quote at current price. 
    df['hold1'] = (starting_base_amount / decimals) * df.price + (df.amount1[0] / decimals)

    # Iterate over each row of the DataFrame to run our backtester
    for i in range(1, len(df) - 1):
        if df.price[i] < df.tL[i]:
            df.amount0[i] = df.deltaE[i]
            df.amount1[i] = 0
            df.positionValue1[i] = ((df.amount0[i] / decimals) * df.price[i]) + (df.amount1[i] / decimals)
            fees_out(i, df)

        elif df.price[i] > df.tH[i]:
            df.amount0[i] = 0
            df.amount1[i] = df.deltaE[i] * math.sqrt(df.tH[i]*df.tL[i])
            df.positionValue1[i] = ((df.amount0[i] / decimals) * df.price[i]) + (df.amount1[i] / decimals)
            fees_out(i, df)

        else:
            df.amount0[i] = df.deltaE[i] * math.sqrt(df.tL[i] / df.price[i]) * (math.sqrt(df.tH[i]) - math.sqrt(df.price[i])) / (math.sqrt(df.tH[i]) - math.sqrt(df.tL[i]))
            df.amount1[i] = df.deltaE[i] * math.sqrt(df.tL[i]*df.tH[i]) * (math.sqrt(df.price[i]) - math.sqrt(df.tL[i])) / (math.sqrt(df.tH[i]) - math.sqrt(df.tL[i]))
            df.positionValue1[i] = ((df.amount0[i] / decimals) * df.price[i]) + (df.amount1[i] / decimals)
            fees(i, df)

    # Remove the last row from the DataFrame
    df = df.drop(df.index[-1])

    # Set the initial fee to 0.
    df.totalFeeIn1 = df.totalFeeIn1 - df.totalFeeIn1[1]
    df.myFee0 = df.myFee0 - df.myFee0[1]
    df.myFee1 = df.myFee1 - df.myFee1[1]

    # Remove the first row from the DataFrame
    df = df.drop(df.index[0])

    # Remove useless columns

    df = df.drop(['feeGrowth0', 'feeGrowth1', 'amount0', 'amount1', 'deltaE', 'deltaFee0', 'deltaFee1', 'myFee0in1', 'myUnbLiq'], axis=1)



    ## IMPORTANT VARIABLES ##

    # When looking at the results, the important columns are: 

    # tL: the lower boundary of the range
    # tH: the upper boundary of the range
    # amountInvested1: the total amount invested (in ETH)
    # positionValue1: the value of the liquidity provision position at the end of the backtest (excluding fees)
    # myFee0: amount of base assets collected in fees
    # myFee1: amount of ETH collected in fees
    # totalFeeIn1: the total amount of fees collected (base + ETH) converted in ETH
    # hold1: value of the position if the 1,000 aidropped tokens and the amount of ETH invested would have been held

    output(df)
