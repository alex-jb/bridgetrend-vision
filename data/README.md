# Data contract

Do not commit downloaded product images, credentials, cookies, private user data, or platform exports that cannot legally be redistributed.

Create `data/metadata.csv` locally with these columns:

| Column | Meaning |
|---|---|
| `image_id` | Unique image identifier |
| `image_path` | Absolute path or path relative to this directory |
| `market` | Market/domain label such as `US` or `CN` |
| `category` | Shared product category |
| `product_id` | Product-family identifier when known |
| `title` | Original listing title |
| `source` | Dataset or approved source name |
| `timestamp` | Observation time when available |

Example:

```csv
image_id,image_path,market,category,product_id,title,source,timestamp
us_shoe_001,raw/us/shoe_001.jpg,US,footwear,p001,Example sneaker,approved_dataset,2026-09-01
cn_shoe_001,raw/cn/shoe_001.jpg,CN,footwear,p101,Example shoe,approved_dataset,2026-09-02
```

Recommended first smoke test: 300-600 images across at least three categories. The main study will expand the taxonomy and report per-category results.

