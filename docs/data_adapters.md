# External data adapters

BridgeTrend Vision keeps external datasets outside Git. The adapter scripts
convert approved local metadata into the common manifest without downloading or
redistributing images.

## Product1M

Product1M is useful as a cosmetic-domain retrieval source, not as the complete
48-category dataset. Its official repository states that each text record
contains two URLs for the same image. The official evaluation code reads the
item identifier from field 0 and labels from field 4 using five-hash separators.

Expected record layout:

~~~text
item_id#####image_url_1#####image_url_2#####caption#####label_1#;#label_2
~~~

Copy and edit the category mapping before importing:

~~~bash
cp configs/product1m_category_map.example.csv   data/product1m_category_map.csv

python scripts/import_product1m.py   --annotations /local/Product1M/product1m_train.txt   --image-root /local/Product1M/images   --category-map data/product1m_category_map.csv   --output data/imports/product1m_manifest.csv
~~~

The importer creates a second file named product1m_manifest_rejected.csv.
Unmapped and malformed rows are kept there for review instead of being silently
discarded.

The example mapping contains readable category names. If the downloaded release
uses numeric or product-level labels, inspect those labels and add their exact
values to the mapping file before the full import.

## Generic U.S. or Chinese catalog CSV

Use the generic adapter for a legally obtained CSV export or public dataset:

~~~bash
python scripts/import_catalog_csv.py   --input /local/catalog/items.csv   --output data/imports/us_catalog_manifest.csv   --market US   --source approved_catalog   --image-id-column sku   --image-path-column local_image   --category-column product_type   --product-id-column product_id   --title-column title   --timestamp-column observed_at   --source-url-column listing_url   --image-root /local/catalog/images   --category-map data/catalog_category_map.csv
~~~

A category map always has two columns:

~~~csv
raw_label,category
Running Shoes,sneakers
Mobile Headset,headphones
Table Lamp,lamp
~~~

The category value must use a slug from configs/taxonomy.yaml.

## Provenance and safety

- Keep raw images, cookies, API keys, and restricted exports outside Git.
- Record the source name and original URL when permitted.
- Review the dataset license or terms before publishing derived metadata.
- Validate each generated manifest before model inference.
- Treat rejected rows as a data-quality report, not as model examples.
