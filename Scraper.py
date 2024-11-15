from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.keys import Keys
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
from dotenv import load_dotenv
import os


class Scraper:
    def __init__(self, headless:bool=True) -> None:
        # Load environment variables
        load_dotenv('./variables.env')

        # Store headless
        self.headless = headless
        
        # Create Driver
        self._create_driver()

        # Store login link
        self.login_link = os.getenv('LOGIN_LINK')

        # Store Password
        self.password = str(os.getenv('PASSWORD'))


    def _create_driver(self):
        # Define absolute path for download directory
        download_dir = os.path.abspath("./Exports")  # Converts to an absolute path

        # Set up Chrome options
        self.chrome_options = Options()
        if self.headless:
            self.chrome_options.add_argument("--headless")
            self.chrome_options.add_argument("--disable-gpu")
        else:
            pass

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

    def _back_to_home(self):
        # Wait until the image with the specific src is clickable, then click it
        logo_image = WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "img[src='/Content/Img/MS7_Logo_2.png']"))
        )
        logo_image.click()
    
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
    

    def _access_regression_semantic(self, reg_type:str="Create", brand:str="MOVE"):
        self._login()

        # Wait until the tile is clickable by class name, then click it
        tile = WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable((By.CLASS_NAME, 'ANALYZE_4_3'))
        )
        tile.click()

        xpath = f"//input[@type='radio' and @value='{reg_type}']"

        # Wait until the dynamic radio button is clickable based on `type`, then click it
        create_radio_button = WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, xpath))
        )
        create_radio_button.click() 

        if reg_type == "CheckModified" or "Modify":
            # Wait for Dropdown
            dropdown = WebDriverWait(self.driver, 10).until(
                EC.visibility_of_element_located((By.ID, "ddlBrand"))
            )

            # Initialize the Select object
            select = Select(dropdown)

            # Wait until the option with the text of the brand is available and select it
            select.select_by_visible_text(brand)

        else:
            pass

        print("-"*10, "Accessed Regression Page", "-"*10)
        return

    def _insert_inputs_semantic(self, inputs_list: list = None, reg_type: str = None, iter: int = None):
        if iter is None:  # Check for None correctly
            input_prefix = 'brdAtt_' if reg_type in ("CheckModified", "CheckNewBrand") else 'dimPerception_'
            for n in range(1, 7):  # range(1, 7) generates 1 through 6
                id_input = f'{input_prefix}{n}' if reg_type in ("CheckModified", "CheckNewBrand") else f'dimPerception_{n}'
                input_value = float(inputs_list[n - 1])
    
            # Wait until input field is clickable and set its value
                input_field = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.ID, id_input))
                )
                self.driver.execute_script("arguments[0].value = arguments[1]; arguments[0].onchange();", input_field, input_value)
        else:  # Handling case when iter is provided
            if reg_type in ("CheckModified", "CheckNewBrand"):
                id_input = f'brdAtt_{iter+1}' if iter < 5 else "brdPrice"
            else:
                id_input = f'dimPerception_{iter}'
                    
            print(id_input)
            input_value = float(inputs_list)
            input_field = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.ID, id_input))
            )
            self.driver.execute_script("arguments[0].value = arguments[1]; arguments[0].onchange();", input_field, input_value)
        return


    def _obtain_results_semantic(self, reg_type:str=None):
        results = []
        
        # Define cases and store values 
        if reg_type == "Modify" or reg_type == "Create":
            for n in range(1, 6 + 1):
                id_output = f'newAttValue_{n}'
                element = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.ID, id_output))
                )
                result_value = element.text
                results.append(result_value)

        elif reg_type == "CheckModified" or reg_type == "CheckNewBrand":
            for n in range(1, 6 + 1):
                id_output = f'newPerceptionValue_{n}'
                element = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.ID, id_output))
                )
                result_value = element.text
                results.append(result_value)
        else:
            raise ValueError
            
        return results

    def _obtain_df_index_semantic(self, reg_type:str=None):
        index = []
        
        # Define cases and store values 
        if reg_type == "Modify" or reg_type == "Create":          
            index = [f'newAttValue_{n}' for n in range(1, 6 + 1)]

        elif reg_type == "CheckModified" or reg_type == "CheckNewBrand":
            index = [f'newPerceptionValue_{n}' for n in range(1, 6 + 1)]
        else:
            raise ValueError
            
        return index


   
    def run_regression_semantic(self, inputs_array:np.ndarray, reg_type:None, brand:str="MOVE"):
        self._access_regression_semantic(reg_type, brand=brand)

        results = {}
        for i, input_list in tqdm(enumerate(inputs_array),desc="Running Regressions..."):
            results_inner = []

            if inputs_array.ndim > 1:
                # Insert inputs with function
                self._insert_inputs_semantic(input_list, reg_type)
            else:
                self._insert_inputs_semantic(input_list, reg_type, i)

            # Obtain results with function
            results_inner = self._obtain_results_semantic(reg_type)
            
            results[i] = results_inner
            
        # Build Df 
        df_results = pd.DataFrame(results).T
        index = self._obtain_df_index_semantic(reg_type)
        df_results.columns = index
            
        return df_results


    def generate_samples_semantic(self, iterations:int = 2, reg_type:str=None):
        # Define random inputs
        inputs = np.random.rand(iterations, 6)*6+1
        
        # Compute the values 
        res = self.run_regression_semantic(inputs_array=inputs, reg_type=reg_type)

        return res
    
    def _access_regression_mds(self, reg_type:str="Create", brand:str="MOVE"):
        self._login()

        # Wait until the tile is clickable by class name, then click it
        tile = WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable((By.CLASS_NAME, 'ANALYZE_4_5'))
        )
        tile.click()

        xpath = f"//input[@type='radio' and @value='{reg_type}']"

        # Wait until the dynamic radio button is clickable based on `type`, then click it
        create_radio_button = WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, xpath))
        )
        create_radio_button.click() 

        if reg_type == "CheckModified" or reg_type == "Modify":
            # Wait for Dropdown
            dropdown = WebDriverWait(self.driver, 10).until(
                EC.visibility_of_element_located((By.ID, "ddlBrand"))
            )

            # Initialize the Select object
            select = Select(dropdown)

            # Wait until the option with the text of the brand is available and select it
            select.select_by_visible_text(brand)

        else:
            pass

        print("-"*10, "Accessed Regression Page", "-"*10)
        return
    
    def _insert_inputs_mds(self, inputs_list:list=None):
        # Find the lenght of an input 
        len_single_input = len(inputs_list)
    
        if len_single_input == 3:
            for i, n in enumerate(range(12, 14 + 1)):
                id_input = f'dimPerception_{n}'

                input_value = float(inputs_list[i])
 
                input_field = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.ID, id_input))
                )
                self.driver.execute_script("arguments[0].value = arguments[1]; arguments[0].onchange();", input_field, input_value)
        
        elif len_single_input == 6:
            for n in range(1, 6 + 1):
                if n <6:
                    id_input = f'brdAtt_{n}'
                else:
                    id_input = "brdPrice"

                input_value = float(inputs_list[n])
 
                input_field = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.ID, id_input))
                )
                self.driver.execute_script("arguments[0].value = arguments[1]; arguments[0].onchange();", input_field, input_value)
        
        return   
         

    def _obtain_results_mds(self, reg_type:str="Create"):
        results = []
        
        # Define cases and store values 
        if reg_type == "Modify" or reg_type == "Create":
            for n in range(1, 6 + 1):
                id_output = f'newSemValue_{n}'
                element = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.ID, id_output))
                )
                result_value = element.text
                results.append(result_value)

        elif reg_type == "CheckModified" or reg_type == "CheckNewBrand":
            for n in range(12, 14 + 1):
                id_output = f'newPerceptionValue_{n}'
                element = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.ID, id_output))
                )
                result_value = element.text
                results.append(result_value)
        else:
            raise ValueError
           
        return results
    
    def _obtain_df_indexes_mds(self, reg_type:str=None):
        inputs_index, outputs_index, achieved_index = [], [], []
        
        if reg_type == "Modify" or reg_type == "Create":
            for n in range(12, 14 + 1):
                id_input = f'dimPerception_{n}'
                id_achieved = f'achivedDimValue_{n}'
                # Append in list
                inputs_index.append(id_input)
                achieved_index.append(id_achieved)

            for n in range(1, 6+1):
                id_output = f'newSemValue_{n}'
                outputs_index.append(id_output)

            return inputs_index, outputs_index, achieved_index

        elif reg_type == "CheckModified" or reg_type == "CheckNewBrand":
            for n in range(1, 6 + 1):
                if n <6:
                    id_input = f'brdAtt_{n}'
                else:
                    id_input = "brdPrice"
                # Append in list
                inputs_index.apend(id_input)
            for n in range(12, 14+1):
                id_output = f'newPerceptionValue_{n}'
                outputs_index.append(id_output)

            return inputs_index, outputs_index, achieved_index

        else:
            raise ValueError


        
      
    def run_regression_mds(self, inputs_array:np.ndarray, reg_type:str="Create", brand:str="Move"):
        self._access_regression_mds(reg_type=reg_type)

        results = {}
        achieved = {}

        if inputs_array.ndim > 1:
            # Add tqdm for iteration progress tracking
            for i, input_array in tqdm(enumerate(inputs_array), desc="Running simulations"):          
                results_inner = []
                achieved_inner = []

                # Insert input with function
                self._insert_inputs_mds(input_array)

                if reg_type == "Create" or reg_type == "Modify": 
                    # Click the button
                    button = WebDriverWait(self.driver, 10).until(
                        EC.element_to_be_clickable((By.ID, 'CalcCharbtn'))
                    )
                    button.click()
                else:
                    pass

                # Find Results
                results_inner = self._obtain_results_mds(reg_type=reg_type)
                # Store in dict
                results[i] = results_inner

                if reg_type == "Create" or reg_type == "Modify": 
                    for n in range(12, 14 + 1):
                        id_achieved = f'achivedDimValue_{n}' #Sic
                        element = WebDriverWait(self.driver, 10).until(
                            EC.presence_of_element_located((By.ID, id_achieved))
                        )
                        achieved_value = element.text
                        achieved_inner.append(achieved_value)
                    # Store in dict
                    achieved[i] = achieved_inner

                else:
                    pass
        else:
            results_inner = []
            achieved_inner = []

            # Insert input with function
            self._insert_inputs_mds(input_array)

            if reg_type == "Create" or reg_type == "Modify": 
                # Click the button
                button = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.ID, 'CalcCharbtn'))
                )
                button.click()
            else:
                pass

            # Find Results
            results_inner = self._obtain_results_mds(reg_type=reg_type)
            # Store in dict
            results[i] = results_inner

            if reg_type == "Create" or reg_type == "Modify": 
                for n in range(12, 14 + 1):
                    id_achieved = f'achivedDimValue_{n}' #Sic
                    element = WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.ID, id_achieved))
                    )
                    achieved_value = element.text
                    achieved_inner.append(achieved_value)
                # Store in dict
                achieved[i] = achieved_inner

            else:
                pass
        
        # Obtain indexes 
        inputs_index, results_index, achieved_index = self._obtain_df_indexes_mds(reg_type=reg_type)

        # Build dfs
        print(inputs_array)
        df_inputs = pd.DataFrame(inputs_array)
        df_inputs.columns = inputs_index

        df_results = pd.DataFrame(results).T
        df_results.columns = results_index

        # Check to see if achieved can be created
        if reg_type == "Create" or reg_type == "Modify":
            df_achieved = pd.DataFrame(achieved).T
            df_achieved.columns = achieved_index
            return df_inputs, df_results, df_achieved
        else:
            return df_inputs, df_results, None
   

    def generate_samples_mds(self, iterations:int=2):
        # Define random inputs
        inputs = np.random.rand(iterations, 3)*40-20
        
        # Compute the values 
        res = self.run_regression_mds(inputs_array=inputs)

        return res
    
    