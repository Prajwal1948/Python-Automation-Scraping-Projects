import requests
from bs4 import BeautifulSoup

import pandas as pd

Names = []
Prices = []
Desc = []
Reviews = []

for i in range(1, 6):

    url = "https://www.flipkart.com/search?q=mobiles&otracker=search&otracker1=search&marketplace=FLIPKART&as-show=off&as=off" + str(
        i)
    r = requests.get(url)
    print(r)

    soup = BeautifulSoup(r.text, "lxml")
    boxes = soup.find_all("div", class_="tUxRFH")

    for box in boxes:
        # Product Name
        name = box.find("div", class_="KzDlHZ")
        Names.append(name.text if name else "N/A")

        # Price
        price = box.find("div", class_="Nx9bqj _4b5DiR")
        if price:
            clean_price = ''.join(ch for ch in price.text if ch.isdigit() or ch == ',')
            Prices.append(clean_price)
        else:
            Prices.append("N/A")

        # Description
        desc = box.find("ul", class_="G4BRas")
        Desc.append(desc.text if desc else "N/A")

        # Reviews
        review = box.find("div", class_="XQDdHH")
        Reviews.append(review.text if review else "No Reviews")

df = pd.DataFrame(
    {"Product Name": Names, "Prices of Product": Prices, "Product Description": Desc, "Product Reviews": Reviews})

print(df)

df.to_csv("Flipkart.csv")