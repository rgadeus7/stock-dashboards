import pandas as pd

# Load your SPX_daily.csv file
file_path = "your_path/SPX_daily.csv"

# Step 1: Read the file correctly, skipping initial metadata lines
raw_df = pd.read_csv(file_path, skiprows=4)
raw_df = raw_df.dropna(how='all')  # drop completely blank rows

# Step 2: Rename columns for clarity
raw_df = raw_df.rename(columns={
    'Unnamed: 0': 'Date',
    '^SPX': 'Close',
    '^SPX.1': 'High',
    '^SPX.2': 'Low',
    '^SPX.3': 'Open',
    '^SPX.4': 'Volume'
})

# Step 3: Preprocessing
raw_df['Date'] = pd.to_datetime(raw_df['Date'])
raw_df = raw_df.sort_values('Date').reset_index(drop=True)
raw_df['DayOfWeek'] = raw_df['Date'].dt.dayofweek

# Step 4: Focus on data from 2000 onward
df = raw_df[raw_df['Date'] >= '2000-01-01'].copy()

# Step 5: Build Friday data
df['Thursday_High'] = df['High'].shift(1)
df['Thursday_Low'] = df['Low'].shift(1)
df['Thursday_Close'] = df['Close'].shift(1)
df['Thursday_Open'] = df['Open'].shift(1)

df['Friday_Close'] = df['Close']
df['Friday_Low'] = df['Low']

fridays = df[df['DayOfWeek'] == 4].copy()

# Step 6: Condition: Friday High < Thursday High
qualified_fridays = fridays[fridays['High'] < fridays['Thursday_High']].copy()
qualified_fridays['Monday_Date'] = qualified_fridays['Date'] + pd.Timedelta(days=3)

# Step 7: Get Monday Data
mondays = df[['Date', 'High', 'Low', 'Close']]
mondays = mondays.rename(columns={'Date': 'Monday_Date', 'High': 'Monday_High', 'Low': 'Monday_Low', 'Close': 'Monday_Close'})

# Step 8: Merge
merged = qualified_fridays.merge(mondays, on='Monday_Date', how='inner')

# Step 9: Perform checks
merged['Monday_Close_Below_Thursday_Low'] = merged['Monday_Close'] <= merged['Thursday_Low']
merged['Monday_Close_Below_Friday_Low'] = merged['Monday_Close'] <= merged['Friday_Low']

merged['Touched_Thursday_High'] = merged['Monday_High'] >= merged['Thursday_High']
merged['Touched_Thursday_Close'] = merged['Monday_High'] >= merged['Thursday_Close']
merged['Touched_Thursday_Open'] = merged['Monday_High'] >= merged['Thursday_Open']
merged['Touched_Thursday_Low_Pullback'] = merged['Monday_High'] >= merged['Thursday_Low']

# Step 10: Calculate statistics
total_cases = len(merged)

results = {
    'Monday Close below Thursday Low (%)': merged['Monday_Close_Below_Thursday_Low'].sum() / total_cases * 100,
    'Monday Close below Friday Low (%)': merged['Monday_Close_Below_Friday_Low'].sum() / total_cases * 100,
    'Monday Touched Thursday High (%)': merged['Touched_Thursday_High'].sum() / total_cases * 100,
    'Monday Touched Thursday Close (%)': merged['Touched_Thursday_Close'].sum() / total_cases * 100,
    'Monday Touched Thursday Open (%)': merged['Touched_Thursday_Open'].sum() / total_cases * 100,
    'Monday Touched Thursday Low (%)': merged['Touched_Thursday_Low_Pullback'].sum() / total_cases * 100
}

# Output results
for key, value in results.items():
    print(f"{key}: {value:.2f}%")
