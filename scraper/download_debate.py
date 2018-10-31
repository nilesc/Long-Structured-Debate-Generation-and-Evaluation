from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import traceback
import pickle


# Init a Firefox driver
driver = webdriver.Firefox()


def save_cookies():
    """
    Must be run once you've logged in to kialo, to save your login cookie.
    Instructions:
    1.) Run this script with python in interactive mode in order to create a webdriver.Firefox instance:
        run python -i donwload_debate.py
    2.) When a browser opens, login to your Kialo account
    3.) In the shell, run:
        save_cookies()
    """
    driver.get("http://www.kialo.com")
    pickle.dump(driver.get_cookies(), open("cookies.pkl", "wb"))


def load_cookies(driver):
    """
    Call after running save_cookies to load your kialo user account (necessary for downloading debates).
    """
    driver.get("http://www.kialo.com")
    cookies = pickle.load(open("cookies.pkl", "rb"))
    for cookie in cookies:
        driver.add_cookie(cookie)


def find_by_class_and_click(class_name, timeout=5):
    """Wait for an element of the given class name to load, and click on it once it does."""
    try:
        pop_up_exit = WebDriverWait(driver, timeout).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, class_name))
        )[0]

        pop_up_exit.click()

    except Exception:
        traceback.print_exc()


def click_through_first_time_popups():
    """
    Currently unused, since it seems to be unnecessary once login cookie is loaded.
    """

    # Escape first time user pop up
    find_by_class_and_click("pop-up-template__close")

    # Escape Debate popup
    find_by_class_and_click("discussion-info-dialog__close")


def download_debate(url):
    driver.get(url)

    # Click top title banner
    find_by_class_and_click("topbar-title__discussion-title")

    # Click 'Export Discussion'
    find_by_class_and_click("discussion-settings-tab__button--export")

    # Click 'Download'
    find_by_class_and_click("confirm")


if __name__ == '__main__':
    """Example usage"""

    # Load saved cookies (specifically: kialo login)
    load_cookies(driver)

    # Download the debate of the specified URL.
    # Requires that you've saved your kialo login cookie as detailed in save_cookies()
    # Also, must have Firefox installed.
    download_debate("https://www.kialo.com/is-water-wet-6298/6298.0=6298.1/=6298.1")
    