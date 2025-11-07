from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import os

def scrape_kaggle_with_selenium(competition_name):
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
            desc_elem = driver.find_element(By.XPATH, '//*[@id="site-content"]/div[2]/div//div[contains(@id, "description")]')
            description = desc_elem.text.strip()
        except:
            print("❌ Description section not found")

        try:
            eval_elem = driver.find_element(By.XPATH, '//*[@id="site-content"]/div[2]/div//div[contains(@id, "evaluation")]')
            evaluation = eval_elem.text.strip()
        except:
            print("❌ Evaluation section not found")

        combined_text = f"--- Description ---\n{description}\n\n--- Evaluation ---\n{evaluation}"

        # Save to file
        output_dir = "/Volumes/SD_Card/Finetuning_LLMs/data/raw/titanic"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{competition_name}_overview.txt")

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(combined_text)

        print(f"✅ Scraped and saved to: {output_file}")

        return combined_text

    finally:
        driver.quit()
