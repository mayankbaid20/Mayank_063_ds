#Lab Assignment - 1
Data Scraping and Preprocessing using Python and Scrapy
#name:Mayank Baid

Here is the complete project details summary in clean, point-by-point format:

#Project Details

Assignment Information

* Course: DS605: Fundamentals of Machine Learning
* Assignment: Lab Assignment 1 – Data Scraping and Preprocessing using Python and Scrapy
* Target Website: [https://books.toscrape.com/](https://books.toscrape.com/)
* Repository Name: Mayank_063_ds

Pipeline Architecture & Workflow

1. Task 1: Data Scraping
* Engine: Built an asynchronous Scrapy spider (BooksSpider in spider.py) that navigates catalog pagination up to 5 pages.


* Depth Crawling: Visited each individual product detail page to extract 9 structured fields: title, category, price, rating, availability, product_description, upc, number_of_reviews, and product_url.


* Total Records Collected: 100 books
* Duplicate UPCs: 0
* Missing Values: 0 across all scraped fields
* Output: Saved raw scraped output directly to books_raw.csv.


2. Task 2: Data Preprocessing & Feature Engineering
* Text Cleaning: Removed whitespace, escape characters, and standardized book descriptions.
* Currency Conversion: Cleaned currency symbols (£) from prices and cast values to numeric floats.
* Rating Mapping: Mapped star ratings (One to Five) into integer values (1 to 5).
* Stock Parsing: Parsed integer stock quantities from availability text strings using regular expressions.
* Feature 1 (description_word_count): Extracted total word count from product descriptions.
* Feature 2 (price_band): Discretized prices into equal quantile buckets (Low, Medium, High).
* Feature 3 (value_score): Engineered an efficiency metric defined as Rating / Price.
* Output: Exported the transformed dataset to books_cleaned.csv.


3. Task 3: Visualization & Analysis
* Price Distribution Plot: Revealed a bimodal pricing curve peaking at £20 (budget) and £55 (premium).
* Rating Breakdown Plot: Confirmed uniform representation across all 1 to 5-star tiers.
* Category Price Plot: Identified Historical Fiction as the highest-priced category (>£53) and Art as the lowest (~£44).
* Price vs. Rating Plot: Box plot showing that higher price does not dictate higher star ratings.
* Text Analysis: Generated a custom Word Cloud from combined book descriptions to highlight common genre keywords and thematic terms.


4. Task 4: Key Insights & Limitations
* Bimodal Pricing: Inventory naturally splits into affordable everyday picks (~£20) and premium titles (~£55), with a noticeable dip in mid-range £30–£35 titles.
* Rating Value: 5-star rated books feature a lower median price (~£25) compared to 1–4 star books (~£36–£37), offering high value for readers.
* Dataset Limitations: Sample size is restricted to 100 items (10% of total site catalog), and product text is limited to seller descriptions rather than genuine customer reviews.



File Structure

* Mayank_063_ds/
* ├── scraping_pipeline.ipynb (Complete runnable notebook with Scraper execution, EDA & Plots)


* ├── spider.py (Scrapy spider script defining BooksSpider class)


* ├── books_raw.csv (Raw scraped output from Scrapy)
* └── books_cleaned.csv (Cleaned dataset with engineered features)
