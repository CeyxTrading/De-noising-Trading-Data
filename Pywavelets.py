import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import pywt
import numpy as np
from datetime import datetime, timedelta
import os


#  Define output directory
RESULTS_DIR = "results"


def fetch_prices(symbol, start_date, end_date):
    try:
        data_df = yf.download('QQQ', start=start_date, end=end_date)
        return data_df
    except Exception as ex:
        print(f"Failed to fetch prices for {symbol}")


def denoise_prices(prices_series, wavelet, scale):
    #  Perform DWT: Decompose prices into wavelet coefficients
    coefficients = pywt.wavedec(prices_series, wavelet, mode='per')
    #  Calculate threshold level based on scale
    threshold = scale * np.max(prices_series)
    #  Apply soft thresholding to all coefficient except the first approximation coefficient
    coefficients[1:] = [pywt.threshold(i, value=threshold, mode='soft') for i in coefficients[1:]]
    #  Reconstruct prices from the thresholded wavelet coefficients
    denoised_prices = pywt.waverec(coefficients, wavelet, mode='per')

    # Create a DataFrame to store original and denoised prices
    combined_prices_df = pd.DataFrame({
        'Adj Close': prices_series,
        'Denoised Close': denoised_prices[:len(prices_series)]  # Making sure both series have the same length
    }, index=prices_series.index)
    return combined_prices_df


def plot_wavelet_function(wavelet):
    phi, psi, x = pywt.Wavelet(wavelet).wavefun(level=5)

    plt.figure(figsize=(10, 5))
    plt.plot(x, psi, label=f'{wavelet} Wavelet')
    plt.title(f'{wavelet} Wavelet Function')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()

    # Save the plot
    file_name = f"{wavelet}-function.png"
    path = os.path.join(RESULTS_DIR, file_name)
    plt.savefig(path)


def plot_denoised_prices(denoised_prices_df, symbol, wavelet, scale):
    plt.figure(figsize=(10, 5))
    plt.plot(denoised_prices_df['Adj Close'], label='Original Prices', alpha=0.6, linewidth=1)
    plt.plot(denoised_prices_df['Denoised Close'], label='Denoised Prices', linewidth=1)
    plt.title(f"{symbol}, Wavelet: {wavelet}, Scale: {scale}: Original vs Denoised Prices")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)

    # Save the plot
    file_name = f"{symbol}-{wavelet}-{scale}-denoised.png"
    path = os.path.join(RESULTS_DIR, file_name)
    plt.savefig(path)


if __name__ == "__main__":
    #  Create output directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # Set start- and end dates
    num_years = 3
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=num_years * 365)).strftime('%Y-%m-%d')

    # Download prices
    symbol = "QQQ"
    prices_df = yf.download('QQQ', start=start_date, end=end_date)
    if prices_df is None or len(prices_df) == 0:
        print(f"No prices available for {symbol}")
        exit(0)

    # Get adjusted close prices
    prices_series = prices_df['Adj Close']

    # List of wavelets to use
    wavelets = ["db6", "haar", "sym5"]
    # List of thresholds
    scales = [0.01, 0.1, 0.5]

    for wavelet in wavelets:
        #  Plot the wavelet function
        plot_wavelet_function(wavelet)
        for scale in scales:
            print(f"Processing {symbol}, wavelet: {wavelet}, scale: {scale}")
            #  Denoise prices
            denoised_prices_df = denoise_prices(prices_series, wavelet, scale)

            #  Plot prices
            plot_denoised_prices(denoised_prices_df, symbol, wavelet, scale)

    print("Done!")



