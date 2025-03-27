from pyngrok import ngrok

# Define the local URL
network_url = "192.168.0.147:8501"

# Start ngrok tunnel
public_url = ngrok.connect(address=network_url, proto="http")

print(f"Ngrok Public URL: {public_url}")
