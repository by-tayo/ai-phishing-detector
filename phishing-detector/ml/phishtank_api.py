import requests

def check_url_live(url: str):
    try:
        # Fetch the OpenPhish live feed
        response = requests.get("https://openphish.com/feed.txt", timeout=10)
        phishing_urls = response.text.splitlines()
        
        in_database = url in phishing_urls
        
        return {
            "url": url,
            "in_database": in_database,
            "verified": in_database,
            "source": "OpenPhish"
        }
    except Exception as e:
        return {
            "url": url,
            "in_database": False,
            "verified": False,
            "source": "OpenPhish",
            "error": str(e)
        }

if __name__ == "__main__":
    result = check_url_live("http://suspicious-login.com")
    print(result)
