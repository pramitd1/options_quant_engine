from kiteconnect import KiteConnect

api_key = "YOUR_API_KEY"
api_secret = "YOUR_API_SECRET"
request_token = "PASTE_REQUEST_TOKEN"

kite = KiteConnect(api_key=api_key)

data = kite.generate_session(request_token, api_secret=api_secret)

print("ACCESS TOKEN:")
print(data["access_token"])