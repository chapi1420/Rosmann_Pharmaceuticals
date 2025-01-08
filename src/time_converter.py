import pandas as pd
trainer = pd.read_csv('C:\\Users\\nadew\\10x\\week4\\Rosmann\\rossmann-store-sales\\processed_train_data.csv')
tester = pd.read_csv('C:\\Users\\nadew\\10x\\week4\\Rosmann\\rossmann-store-sales\\processed_test_data.csv')
def preprocess_data(df):
    # Convert 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Extract features from 'Date'
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df.drop(['Date'], axis=1, inplace=True)  # Drop the original 'Date' column if not needed

    # One-hot encode 'StateHoliday'
    df = pd.get_dummies(df, columns=['StateHoliday'], drop_first=True)

    # Handle missing values (if any)
    df.fillna(0, inplace=True)  # Replace NaNs with 0 or use other imputation strategies

    return df
train1 = preprocess_data(trainer)
train1.to_csv('C:\\Users\\nadew\\10x\\week4\\Rosmann\\rossmann-store-sales\\processed_train_data.csv')
test1 = preprocess_data(tester)

# print(tester.dtypes)
