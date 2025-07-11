import pandas as pd

df = pd.read_csv('./wnba_game_data/wnba_stats_20250710_083447.csv', encoding='utf-8')
# Strip whitespace and print unique values for Paige
paige = df[df['Player'].str.lower().str.strip() == 'paige bueckers']


print(df['MP'].dtype)

