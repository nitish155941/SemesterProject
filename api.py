import requests

API_KEY = "OAE7RRQZBF5DSMP7"

url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=IBM&outputsize=full&apikey={API_KEY}'
r = requests.get(url)

if r.status_code == 200:
    data = r.json()
    print(data)
else:
    print(f"Error: Received status code {r.status_code}. Check API key or try again later.")