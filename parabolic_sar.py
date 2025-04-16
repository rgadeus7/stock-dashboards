import pandas as pd
import matplotlib.pyplot as plt

def parabolic_sar(df, acceleration=0.02, maximum=0.2):
    direction = [1]
    sar = [df['Low'].iloc[0]]
    af = acceleration
    ep = df['High'].iloc[0]
    for i in range(1, len(df)):
        sar_prev = sar[-1]
        direction.append(direction[-1])
        sar.append(sar_prev + af * (ep - sar_prev))
        if direction[-1] == 1:
            if df['Low'].iloc[i] < sar[-1]:
                direction[-1] = -1
                sar[-1] = ep
                ep = df['Low'].iloc[i]
                af = acceleration
        else:
            if df['High'].iloc[i] > sar[-1]:
                direction[-1] = 1
                sar[-1] = ep
                ep = df['High'].iloc[i]
                af = acceleration
        if direction[-1] == 1 and df['High'].iloc[i] > ep:
            ep = df['High'].iloc[i]
            af = min(af + acceleration, maximum)
        elif direction[-1] == -1 and df['Low'].iloc[i] < ep:
            ep = df['Low'].iloc[i]
            af = min(af + acceleration, maximum)
    return pd.Series(sar, index=df.index)

def plot_parabolic_sar(df, title='Parabolic SAR Trailing Stop'):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Closing Price', color='blue')
    plt.plot(df.index, df['Parabolic_SAR'], label='Parabolic SAR Trailing Stop', color='red')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(title)
    plt.legend()
    plt.show() 