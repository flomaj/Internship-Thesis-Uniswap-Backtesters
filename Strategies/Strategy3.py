import math
import pandas as pd

# Load dataset
df = pd.read_csv('Data/dataPSP.csv')

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
def fees(i):
    df.deltaFee0[i] = df.feeGrowth0[i] - df.feeGrowth0[i-1]
    df.deltaFee1[i] = df.feeGrowth1[i] - df.feeGrowth1[i-1]
    df.myFee0[i] = df.myFee0[i-1] + (df.deltaFee0[i] / decimals / q128) * df.myUnbLiq[i]
    df.myFee0in1[i] = df.myFee0[i] * df.price[i]
    df.myFee1[i] = df.myFee1[i-1] + (df.deltaFee1[i] / decimals / q128) * df.myUnbLiq[i]
    df.totalFeeIn1[i] = df.myFee0in1[i] + df.myFee1[i]

# Define rebalance (this function will be applied when the position has to be rebalanced and redeployed to keep the price in the range)
def rebalance(i, base_amount):
    df.tH[i:] = df.price[i] * 1.3
    df.tL[i:] = df.price[i] * 0.7
    df.amount0[i:] = base_amount
    df.deltaE[i:] = df.amount0[i] * math.sqrt(df.price[i] / df.tL[i]) * (math.sqrt(df.tH[i]) - math.sqrt(df.tL[i])) / (math.sqrt(df.tH[i]) - math.sqrt(df.price[i]))
    df.amount1[i:] = df.deltaE[i] * math.sqrt(df.tL[i]*df.tH[i]) * (math.sqrt(df.price[i]) - math.sqrt(df.tL[i])) / (math.sqrt(df.tH[i]) - math.sqrt(df.tL[i]))
    df.myUnbLiq[i:] = compute_liquidity(df.tL[i], df.tH[i], df.price[i], df.amount0[i], df.amount1[i])
    df.injection0[i] = (df.amount0[i] - df.amount0[i-1]) / decimals
    df.injection0in1[i] = df.injection0[i] * df.price[i]
    df.injection1[i] = (df.amount1[i] - df.amount1[i-1]) / decimals
    df.injectionTotal1[i:] = df.injectionTotal1[i-1] + df.injection1[i] + df.injection0in1[i]
    df.totalInvest0[i:] = df.totalInvest0[i-1] + df.injection0[i]
    df.totalInvest1[i:] = df.totalInvest1[i-1] + df.injection1[i] + df.injection0in1[i]

# Set initial values for columns in the DataFrame 
df['tL'] = df.price[0] * 0.70
df['tH'] = df.price[0] * 1.3
df['amount0'] = starting_base_amount
df['amount1'] = (df.amount0[0] / ((1/math.sqrt(df.price[0])) - (1/math.sqrt(df.tH[0])))) * (math.sqrt(df.price[0]) - math.sqrt(df.tL[0]))
df['positionValue1'] = (df.amount0 / decimals) * df.price + (df.amount1 / decimals) #value of the LP at each point
df['injection0'] = 0 #Amount of base that you add when rebalancing
df['injection0in1'] = 0 #Amount of base that you add when rebalancing in terms of quote 
df['injection1'] = 0 #Amount of quote that you add when rebalancing
df['injectionTotal1'] = 0
df['totalInvest0'] = starting_base_amount / decimals #total amount of base invested (including the airdrop)
df['totalInvest1'] = df.amount1[0] / decimals #total of quote invested in TOTAL => amount of money that you had to invest including rebalacing
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
    if df.price[i] > 0.9*df.tH[i]:
        rebalance(i, starting_base_amount)
        fees(i)

    elif df.price[i] < 1.1*df.tL[i]:
        rebalance(i, starting_base_amount)
        fees(i)

    else:
        df.amount0[i:] = df.deltaE[i] * math.sqrt(df.tL[i] / df.price[i]) * (math.sqrt(df.tH[i]) - math.sqrt(df.price[i])) / (math.sqrt(df.tH[i]) - math.sqrt(df.tL[i]))
        df.amount1[i:] = df.deltaE[i] * math.sqrt(df.tL[i]*df.tH[i]) * (math.sqrt(df.price[i]) - math.sqrt(df.tL[i])) / (math.sqrt(df.tH[i]) - math.sqrt(df.tL[i]))
        fees(i)

# Remove the last row from the DataFrame
df = df.drop(df.index[-1])

# Set the initial fee to 0.
df.totalFeeIn1 = df.totalFeeIn1 - df.totalFeeIn1[1]
df.myFee0 = df.myFee0 - df.myFee0[1]
df.myFee1 = df.myFee1 - df.myFee1[1]

# Remove the first row from the DataFrame
df = df.drop(df.index[0])

# Remove useless columns

df = df.drop(['feeGrowth0', 'feeGrowth1', 'amount0', 'amount1', 'deltaE', 'deltaFee0', 'deltaFee1', 'myFee0in1', 'totalInvest0', 'myUnbLiq', 'injection0', 'injection1', 'injection0in1'], axis=1)



## IMPORTANT VARIABLES ##

# When looking at the results, the important columns are: 

# tL: the lower boundary of the range
# tH: the upper boundary of the range
# amountInvested1: the total amount invested (in ETH)
# positionValue1: the value of the liquidity provision position at the end of the backtest (excluding fees)
# injectionTotal1: The total amount of capital withdrew or injected from the position when rebalancing
# myFee0: amount of base assets collected in fees
# myFee1: amount of ETH collected in fees
# totalFeeIn1: the total amount of fees collected (base + ETH) converted in ETH
# hold1: value of the position if the 1,000 aidropped tokens and the amount of ETH invested would have been held 
