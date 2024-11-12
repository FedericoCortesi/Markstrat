from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.keys import Keys
import pandas as pd
import random
import time
from tqdm import tqdm
import os
import numpy as np


class Scraper:
    def __init__(self) -> None:

        # Create Driver
        self._create_driver()

        # Store login link
        self.login_link = "https://markstrat7.stratxsimulations.com/Home/IndexPAK?PAK=PMW-PHLWW"

        # Store Password
        self.password = '8583'

    def _create_driver(self):
        # Define absolute path for download directory
        download_dir = os.path.abspath("./Exports")  # Converts to an absolute path

        # Set up Chrome options
        self.chrome_options = Options()
        self.chrome_options.add_argument("--headless")
        self.chrome_options.add_argument("--disable-gpu")
        self.chrome_options.add_experimental_option("prefs", {
            "download.default_directory": download_dir,  # Specify your desired download location
            "download.prompt_for_download": False,  # Disable the download prompt
            "download.directory_upgrade": True,     # Automatically upgrade to the specified download directory
            "safebrowsing.enabled": True,           # Enable safe browsing to avoid download warnings
            "safebrowsing.disable_download_protection": True,  # Disable download protection for certain file types
            "profile.default_content_setting_values.automatic_downloads": 1,  # Allow automatic downloads
            "profile.default_content_setting_values.notifications": 2         # Disable notifications
        })
        self.chrome_options.add_argument('--no-sandbox')  # Optional, for certain environments
        self.chrome_options.add_argument('--disable-dev-shm-usage')  # Optional, for performance

        # Initialize Chrome with these options
        self.driver = webdriver.Chrome(ChromeDriverManager().install(), options=self.chrome_options)
        return 


    def _login(self):
        # Open URL
        self.driver.get(self.login_link)

        # Locate the input field by its ID and enter a number
        input_field = self.driver.find_element(By.ID, 'txtPassword')
        input_field.send_keys(self.password)  # Replace '123456' with the actual number you want to enter

        # Locate send button and click
        submit_button = self.driver.find_element(By.ID, 'btnSubmitPassword')
        submit_button.click()

        print("-"*10, "Logged In", "-"*10)
        return 

    def download_exports(self):
        self._login()
        
        # click on nav resources first
        nav_resources = WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//li[@data-id="navResources"]/a'))
        )
        nav_resources.click()


        # Wait until the tile is clickable by class name, then click it
        excel_export_link = WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//li[@data-id="navExcelExport"]/a'))
        )
        excel_export_link.click()

        # Wait for the page section containing export links to load
        export_section = WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "analyse-view"))
        )

        # Find the first export link, which corresponds to the highest period
        latest_period_link = export_section.find_element(By.TAG_NAME, "a")

        # Click the link to initiate the download
        latest_period_link.click()

        return
    

    def _access_semantic_regression(self, type:str="Create"):
        self._login()

        # Wait until the tile is clickable by class name, then click it
        tile = WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable((By.CLASS_NAME, 'ANALYZE_4_3'))
        )
        tile.click()
            
        # Wait until the dynamic radio button is clickable based on `type`, then click it
        create_radio_button = WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, f"//input[@type='radio' and @value='{type}']"))
        )
        create_radio_button.click() 

        print("-"*10, "Accessed Regression Page", "-"*10)
        return

    def _access_mds_regression(self):
        self._login()

        # Wait until the tile is clickable by class name, then click it
        tile = WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable((By.CLASS_NAME, 'ANALYZE_4_5'))
        )
        tile.click()

        # Wait until the "Create" radio button is clickable and then click it
        create_radio_button = WebDriverWait(self.driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//input[@type='radio' and @value='Create']"))
        )
        create_radio_button.click()        

        print("-"*10, "Accessed Regression Page", "-"*10)
        return
    

    def run_regression_semantic(self, inputs):
        self._access_semantic_regression()

        results = {}
        for i, input in tqdm(enumerate(inputs),desc="Running Regressions..."):
            results_inner = []
            for n in range(1,7):
                # Define IDs
                id_input = f'dimPerception_{n}'
                id_output = f'newAttValue_{n}'

                # Insert Random Number and store it
                input_value = float(input[n-1])

                # Locate the input field by its ID and enter a number
                input_field = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.ID, id_input))
                )            

                # Use JavaScript to set the value directly
                self.driver.execute_script("arguments[0].value = arguments[1]; arguments[0].onchange();", input_field, input_value)

                # Locate the element by its ID and retrieve the text content
                element = self.driver.find_element(By.ID, id_output)
                result_value = element.text  # Retrieve the inner text of the <td> element
                results_inner.append(result_value)
            
            # Store in Dict
            results[i] = results_inner

        # Build Df 
        df_inputs = pd.DataFrame(inputs)
        index = [f'dimPerception_{n}' for n in range(1, 6 + 1)]
        df_inputs.columns = index

        df_results = pd.DataFrame(results).T
        index = [f'newAttValue_{n}' for n in range(1, 6 + 1)]
        df_results.columns = index
            
        return df_inputs, df_results


    def run_simulations_semantic(self, iterations:int = 2):
        # Define random inputs
        inputs = np.random.rand(iterations, 6)*6+1
        
        # Compute the values 
        res = self.run_regression_semantic(inputs=inputs)

        return res

    def run_simulations_mds(self, iterations: int = 2):
        self._access_mds_regression()

        inputs = {}
        results = {}
        achieved = {}

        # Add tqdm for iteration progress tracking
        for iter in tqdm(list(range(1, iterations + 1)), desc="Running simulations"):
            # Start timer for this iteration
            start_time = time.time()
            
            inputs_inner = []
            results_inner = []
            achieved_inner = []

            # Iterate over Inputs
            for n in range(12, 14 + 1):
                id_input = f'dimPerception_{n}'
                input_value = random.uniform(1, 7)
                inputs_inner.append(input_value)

                input_field = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.ID, id_input))
                )
                self.driver.execute_script("arguments[0].value = arguments[1]; arguments[0].onchange();", input_field, input_value)

            # Click the button
            button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.ID, 'CalcCharbtn'))
            )
            button.click()

            # Find Results
            for n in range(1, 6 + 1):
                id_output = f'newSemValue_{n}'
                element = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.ID, id_output))
                )
                result_value = element.text
                results_inner.append(result_value)

            # Find Achieved
            for n in range(12, 14 + 1):
                id_achieved = f'achivedDimValue_{n}'
                element = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.ID, id_achieved))
                )
                achieved_value = element.text
                achieved_inner.append(achieved_value)

            # Store in dictionaries
            inputs[iter] = inputs_inner
            results[iter] = results_inner
            achieved[iter] = achieved_inner

            # Calculate and print iteration time
            iter_duration = time.time() - start_time
            print(f"Iteration {iter} took {iter_duration:.2f} seconds")

        
        # Build dfs
        df_inputs = pd.DataFrame(inputs).T
        index = [f'dimPerception_{n}' for n in range(12, 14 + 1)]
        df_inputs.columns = index

        df_results = pd.DataFrame(results).T
        index = [f'newSemValue_{n}' for n in range(1, 6 + 1)]
        df_results.columns = index

        df_achieved = pd.DataFrame(achieved).T
        index = [f'achivedDimValue_{n}' for n in range(12, 14 + 1)] # Sic
        df_achieved.columns = index

        self.driver.quit()
        return df_inputs, df_results, df_achieved
