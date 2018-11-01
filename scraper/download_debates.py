from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
import traceback
import requests
import csv
import urllib.parse
import time
import os
from bs4 import BeautifulSoup


# Init a Firefox driver (replace the next line to the path to your own user profile)
firefox_user_path = "~/.mozilla/firefox/gu16idx8.default/"
profile = webdriver.FirefoxProfile(os.path.expanduser(firefox_user_path))
driver = webdriver.Firefox(profile)


# Init a CSV to store the results
results_csv = open('debate_results.csv', 'w')
csv_writer = csv.writer(results_csv)


def find_by_class_and_click(class_name, timeout=5, verbose=True):
    """Wait for an element of the given class name to load, and click on it once it does."""
    success = False
    try:
        pop_up_exit = WebDriverWait(driver, timeout).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, class_name))
        )[0]

        pop_up_exit.click()
        success = True

    except Exception:
        if verbose:
            print(class_name)
            traceback.print_exc()

    return success


def click_through_first_time_popups():
    """
    Currently unused, since it seems to be unnecessary once login cookie is loaded.
    """

    # Escape first time user pop up
    find_by_class_and_click("pop-up-template__close")


def insufficient_permissions():
    try:
        pop_up_title = driver.find_element_by_class_name("dialog-template__title").text
    except NoSuchElementException:
        return False

    return pop_up_title == 'Insufficient permissions'


def download_debate(url):
    driver.get(url)

    success = True

    # Check that it is a public debate
    if insufficient_permissions():
        return False

    else:
        # Handle occasional debate banner
        find_by_class_and_click("discussion-info-dialog__close", verbose=False)

        # Click top title banner
        success = success and find_by_class_and_click("topbar-title__discussion-title")

        # Click 'Export Discussion'
        success = success and find_by_class_and_click("discussion-settings-tab__button--export")

        # Click 'Download'
        success = success and find_by_class_and_click("confirm")

        if len(driver.window_handles) > 1:
            window_before = driver.window_handles[0]
            window_after = driver.window_handles[1]

            # Close the popup
            driver.switch_to.window(window_after)
            driver.close()
            driver.switch_to.window(window_before)

    return success


def download_all_debates(ids=None, delay=2):
    """
    :param ids: list of publicly accessible debate IDs
    :param delay: millisecond delay between download attempts
    """

    base_url = "https://www.kialo.com/"

    if ids:
        id_range = ids
    else:
        id_range = range(1, 23084)  # 23084 should be the last valid discussion post

    for i in id_range:

        url = urllib.parse.urljoin(base_url, str(i))

        resp = requests.get(url)
        status_code = resp.status_code
        result = 'fail'
        filename = ''

        if status_code == 200:
            result = download_debate(url)
            time.sleep(delay)

        csv_writer.writerow([i, status_code, result, filename])


def get_public_discussion_ids(sitemap):
    """
    Get the list of public debates from the sitemap
    https://www.kialo.com/sitemap_discussions.xml
    """
    sitemap_file = open(sitemap, 'r').read()
    soup = BeautifulSoup(sitemap_file, 'lxml')
    urls = soup.find_all('loc')
    urls = [url.text for url in urls]
    ids = [url.split('-')[-1] for url in urls]
    return ids


def downloaded_ids():
    """
    Get list of downloaded debate ids that have already been downloaded
    """
    downloads = os.path.expanduser('~/Downloads/')
    downloaded = os.listdir(downloads)
    ids = [os.path.splitext(os.path.basename(file))[0]
               .split('-')[-1] for file in downloaded]

    return ids


def init_csv():
    header = ['id', 'status_code', 'success', 'filename']
    csv_writer.writerow(header)


if __name__ == '__main__':

    init_csv()

    debate_ids = sorted(list(
        set(get_public_discussion_ids('sitemap_discussions.xml')) -
        set(downloaded_ids())
    ))

    print('Downloading', len(debate_ids), 'debates')

    # Requires that you've logged in to kialo on Firefox and allowed cookies
    download_all_debates(ids=debate_ids)
