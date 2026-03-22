import requests

try:
    response = requests.get("https://api.tavily.com", timeout=5)
    print("Status code:", response.status_code)
    print("Connection successful!")
except Exception as e:
    print("Connection failed:", e)