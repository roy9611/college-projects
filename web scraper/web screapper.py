import requests
from bs4 import BeautifulSoup
import re
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# URL to scrape
url = 'http://example.com'

# Function to fetch the page content
def fetch_page(url):
    """
    Fetches the content of the given URL.

    Args:
        url (str): The URL to fetch.

    Returns:
        str: The HTML content of the page or None if an error occurred.
    """
    try:
        # Send a GET request to the website
        response = requests.get(url)
        
        # Check if the request was successful
        if response.status_code == 200:
            logging.info(f"Successfully fetched the page: {url}")
            return response.text
        else:
            logging.error(f"Failed to retrieve the page. Status code: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"An error occurred while fetching the page: {e}")
        return None

# Function to parse the page content and extract useful data
def parse_page(page_content):
    """
    Parses the HTML content and extracts the title and links.

    Args:
        page_content (str): The HTML content of the page.
    """
    try:
        # Create a BeautifulSoup object to parse the HTML
        soup = BeautifulSoup(page_content, 'html.parser')
        
        # Extract the page title
        title = soup.find('title').text if soup.find('title') else "No Title Found"
        logging.info(f"Page Title: {title}")
        
        # Extract all the anchor tags (links) and print them
        logging.info("\nExtracting all links from the page:")
        links = soup.find_all('a', href=True)  # Extract only valid links
        extracted_links = []
        for link in links:
            href = link['href']
            logging.info(f"Link: {href}")
            extracted_links.append(href)
        
        return title, extracted_links

    except Exception as e:
        logging.error(f"An error occurred while parsing the page: {e}")

# Function to save links to a file
def save_links_to_file(links, filename):
    """
    Saves the extracted links to a specified file.

    Args:
        links (list): A list of links to save.
        filename (str): The name of the file to save the links to.
    """
    try:
        with open(filename, 'w') as file:
            for link in links:
                file.write(link + '\n')
        logging.info(f"Links saved to {filename}")
    except Exception as e:
        logging.error(f"An error occurred while saving links to file: {e}")

# Function to validate URL
def is_valid_url(url):
    """
    Validates the given URL.

    Args:
        url (str): The URL to validate.

    Returns:
        bool: True if the URL is valid, False otherwise.
    """
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  # ...or ipv4
        r'\[?[A-F0-9]*:[A-F0-9:]+\]?)'  # ...or ipv6
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(regex, url) is not None

# Function to get user input for URL
def get_user_input():
    """
    Prompts the user for a URL and validates it.

    Returns:
        str: A valid URL entered by the user.
    """
    while True:
        user_url = input("Please enter a URL to scrape: ")
        if is_valid_url(user_url):
            return user_url
        else:
            logging.warning("Invalid URL. Please try again.")

# Main function
def main():
    """
    Main function to execute the web scraping process.
    """
    # Get user input for URL
    user_url = get_user_input # Fetch the page content
    page_content = fetch_page(user_url)
    
    if page_content:
        # Parse the page content
        title, links = parse_page(page_content)
        
        # Save the extracted links to a file
        save_links_to_file(links, 'extracted_links.txt')
        
        # Display the title and number of links found
        logging.info(f"Title of the page: {title}")
        logging.info(f"Total links extracted: {len(links)}")

    # Additional feature: Ask user if they want to scrape another URL
    while True:
        another = input("Do you want to scrape another URL? (yes/no): ").strip().lower()
        if another == 'yes':
            user_url = get_user_input()
            page_content = fetch_page(user_url)
            if page_content:
                title, links = parse_page(page_content)
                save_links_to_file(links, 'extracted_links.txt')
                logging.info(f"Title of the page: {title}")
                logging.info(f"Total links extracted: {len(links)}")
        elif another == 'no':
            logging.info("Exiting the program.")
            break
        else:
            logging.warning("Invalid input. Please enter 'yes' or 'no'.")

# Function to display the menu
def display_menu():
    """
    Displays the main menu options to the user.
    """
    print("\n--- Web Scraper Menu ---")
    print("1. Scrape a URL")
    print("2. Exit")

# Function to handle menu selection
def handle_menu_selection(selection):
    """
    Handles the user's menu selection.

    Args:
        selection (int): The user's menu selection.
    """
    if selection == 1:
        user_url = get_user_input()
        page_content = fetch_page(user_url)
        if page_content:
            title, links = parse_page(page_content)
            save_links_to_file(links, 'extracted_links.txt')
            logging.info(f"Title of the page: {title}")
            logging.info(f"Total links extracted: {len(links)}")
    elif selection == 2:
        logging.info("Exiting the program.")
        exit()
    else:
        logging.warning("Invalid selection. Please choose a valid option.")

# Function to run the application
def run_application():
    """
    Runs the web scraper application.
    """
    while True:
        display_menu()
        try:
            selection = int(input("Select an option: "))
            handle_menu_selection(selection)
        except ValueError:
            logging.warning("Please enter a valid number.")

if __name__ == "__main__":
    run_application()