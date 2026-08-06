import scrapy
from scrapy.crawler import CrawlerProcess


class BooksSpider(scrapy.Spider):
    name = "books_spider"
    start_urls = ["https://books.toscrape.com/catalogue/page-1.html"]
    page_count = 1

    def parse(self, response):
        book_links = response.css("article.product_pod h3 a::attr(href)").getall()
        for link in book_links:
            yield response.follow(link, callback=self.parse_book_page)

        next_page = response.css("li.next a::attr(href)").get()
        if next_page and self.page_count < 5:
            self.page_count += 1
            yield response.follow(next_page, callback=self.parse)

    def parse_book_page(self, response):
        table_data = {}
        for row in response.css("table.table-striped tr"):
            key = row.css("th::text").get()
            val = row.css("td::text").get()
            if key and val:
                table_data[key.strip()] = val.strip()

        item = {
            "title": response.css("div.product_main h1::text").get(),
            "category": response.css("ul.breadcrumb li:nth-child(3) a::text").get(),
            "price": response.css("p.price_color::text").get(),
            "rating": response.css("p.star-rating::attr(class)").get().replace("star-rating ", "") if response.css("p.star-rating") else None,
            "availability": response.css("p.instock.availability::text").getall(),
            "product_description": response.css("#product_description + p::text").get(),
            "upc": table_data.get("UPC"),
            "number_of_reviews": table_data.get("Number of reviews"),
            "product_url": response.url
        }

        if isinstance(item["availability"], list):
            item["availability"] = " ".join([text.strip() for text in item["availability"] if text.strip()])

        yield item
