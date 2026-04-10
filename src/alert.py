import requests

BOT_TOKEN = "8641286925:AAHaNSPI04RI9HvawPmbg6UTnoQCGCm6wYQ"
CHAT_ID = "882317843"

def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message
    }

    response = requests.post(url, data=payload)
    print(response.json())