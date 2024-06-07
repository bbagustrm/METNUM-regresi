import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os

def prepare_data():
    # Define the path to the CSV file
    file_path = os.path.join(os.path.dirname(__file__), 'Student_Performance.csv') # Update this path

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    # Baca data dari CSV
    data = pd.read_csv(file_path)

    # Ambil kolom yang relevan
    data = data[['Hours Studied', 'Sample Question Papers Practiced', 'Performance Index']]
    data.columns = ['Durasi Waktu Belajar(TB)', 'Jumlah Latihan Soal(NL)', 'Nilai Ujian Siswa (NL)']
    data.to_csv('hasil_process.csv', index=False)
    print(data.head())

def load_data():
    # Baca data yang sudah diproses
    file_path = 'hasil_process.csv'

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    data = pd.read_csv(file_path)
    return data

def linear_regression(X_train, y_train, X_test):
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred_linear = linear_model.predict(X_test)
    return y_pred_linear

def exponential_regression(X_train, y_train, X_test):
    # Transformasi data menggunakan log
    X_train_exp = X_train.copy()
    X_train_exp['Durasi Waktu Belajar(TB)'] = np.log(X_train_exp['Durasi Waktu Belajar(TB)'] + 1)
    X_train_exp['Jumlah Latihan Soal(NL)'] = np.log(X_train_exp['Jumlah Latihan Soal(NL)'] + 1)

    X_test_exp = X_test.copy()
    X_test_exp['Durasi Waktu Belajar(TB)'] = np.log(X_test_exp['Durasi Waktu Belajar(TB)'] + 1)
    X_test_exp['Jumlah Latihan Soal(NL)'] = np.log(X_test_exp['Jumlah Latihan Soal(NL)'] + 1)

    exp_model = LinearRegression()
    exp_model.fit(X_train_exp, y_train)
    y_pred_exp = exp_model.predict(X_test_exp)
    return y_pred_exp

def calculate_rms(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

def visualize_results(y_test, y_pred_linear, y_pred_exp):
    # Visualisasi hasil regresi linear
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred_linear, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Observed')
    plt.ylabel('Predicted')
    plt.title('Linear Regression')

    # Visualisasi hasil regresi eksponensial
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred_exp, color='red')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Observed')
    plt.ylabel('Predicted')
    plt.title('Exponential Regression')

    plt.tight_layout()
    plt.show()

def visualize_single_result(y_test, y_pred, color, title):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, color=color)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Observed')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.show()


def main():
    prepare_data()
    data = load_data()
    if data is None:
        print("Data not found. Exiting program.")
        return

    # Memisahkan fitur dan target
    X = data[['Durasi Waktu Belajar(TB)', 'Jumlah Latihan Soal(NL)']]
    y = data['Nilai Ujian Siswa (NL)']

    # Membagi data menjadi training dan testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fungsi untuk menjalankan analisis berdasarkan metode yang dipilih
    def run_analysis(method):
        y_pred_linear = None
        y_pred_exp = None

        if method == '1':
            y_pred_linear = linear_regression(X_train, y_train, X_test)
            rms_linear = calculate_rms(y_test, y_pred_linear)
            print(f'RMS Error for Linear Model: {rms_linear}')

        if method == '2':
            y_pred_exp = exponential_regression(X_train, y_train, X_test)
            rms_exp = calculate_rms(y_test, y_pred_exp)
            print(f'RMS Error for Exponential Model: {rms_exp}')

        if method == '1' and y_pred_linear is not None:
            visualize_single_result(y_test, y_pred_linear, 'blue', 'Linear Regression')
        elif method == '2' and y_pred_exp is not None:
            visualize_single_result(y_test, y_pred_exp, 'red', 'Exponential Regression')
        else:
            print("Silahkan pilih 1/2")

    # Loop untuk meminta input metode dari pengguna
    while True:
        method = input("Pilih metode regresi: \n1.linear \n2.eksponensial\n ketik 'exit' untuk keluar: ").strip().lower()
        if method == 'exit':
            print("Program dihentikan.")
            break
        elif method in ['1', '2']:
            run_analysis(method)
        else:
            print("Silahkan pilih 1/2")

if __name__ == "__main__":
    main()