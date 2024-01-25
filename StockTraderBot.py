from bs4 import BeautifulSoup

def scrape_last_price(html):
    soup = BeautifulSoup(html, 'html.parser')
    last_price = soup.find('fin-streamer').text
    return last_price


#Tesla Price Calculator to the second. Exact price.
# Example usage
html = '<fin-streamer data-test="colorChange" class="" data-symbol="TSLA" data-field="regularMarketPrice" data-trend="none" data-pricehint="2" value="207.83">207.83</fin-streamer>'
last_price = scrape_last_price(html)
print("Last Price:", last_price)
