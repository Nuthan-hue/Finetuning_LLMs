from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import os

def scrape_data_with_selenium(competition_name, output_dir=None):
    """
    Scrape competition overview from Kaggle using Selenium.

    Args:
        competition_name: Name of the Kaggle competition
        output_dir: Directory to save the overview file (optional, for caching)

    Returns:
        Combined description and evaluation text
    """
    url = f"https://www.kaggle.com/competitions/{competition_name}/overview"

    # Setup headless browser
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)

    try:
        driver.get(url)
        time.sleep(5)  # Wait for JS to load content

        # Find description and evaluation sections
        description = ""
        evaluation = ""

        try:
            desc_elem = driver.find_element(By.XPATH, '//div[contains(@class, "sc-iNHxRi iA-dloL")]')
            Kaggle_Data = desc_elem.text.strip()
        except:
            print("❌ Description section not found")

        combined_text = f"--- Data ---{Kaggle_Data}"

        # Save to file if output_dir provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{competition_name}_overview.txt")

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(combined_text)

            print(f"✅ Scraped and saved to: {output_file}")

        return combined_text

    finally:
        driver.quit()
