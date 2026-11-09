# Mock food catalogue

`options.json` contains 30 entirely fictional food-item records, created with OpenAI Codex assistance for this assignment. Names, prices, ratings, locations, and delivery estimates are synthetic examples, not claims about actual businesses or availability. No private data or production services were used.

## Field definitions

| Field | Meaning |
| --- | --- |
| `id` | Unique stable identifier for one food option |
| `restaurant_name` | Fictional restaurant name |
| `dish_name` | Dish for one serving |
| `cuisine` | Catalogue cuisine label |
| `dietary_type` | Item-level label: `vegetarian`, `vegan`, or `non_vegetarian` |
| `price_inr` | Total mock payable amount for one serving, including any fees and taxes; no extra charges are modelled |
| `rating` | Mock restaurant rating on a 0–5 scale, not a dish-specific rating |
| `delivery_minutes` | Mock estimated delivery duration, not a delivery guarantee |
| `location` | Supported delivery locality for this option, matched by locality name; no distance or geocoding inference |

Localities are Indiranagar, Koramangala, and Whitefield in Bengaluru. Records are independent scenarios; delivery estimates do not change with an address or time of day.

Vegan items contain no animal products under this simplified catalogue convention and can also satisfy a vegetarian request. Vegetarian items may contain dairy; non-vegetarian items do not satisfy vegetarian or vegan requests. These labels do not establish allergen safety or preparation practices.

Organic status, allergens, spice level, offers, and live stock are unavailable. A request depending on such information must retain the requirement and report that it cannot be verified.

## Deliberate coverage

Indiranagar vegetarian biryanis include a cheaper, lower-rated item (`food_001`), a higher-rated item below ₹300 and within 30 minutes (`food_002`), an item exactly at ₹300 (`food_003`), a slow item (`food_004`), and an over-budget item (`food_005`). Additional records cover different dishes, cuisines, dietary types, and locations. These contrasts support later tests; they are not evaluation results.

## Request conventions

Schemas live in `app/schemas.py`. Unstated scalar preferences default to `null`; an empty unsupported-constraints list means none have been identified. “Highly rated” sets a qualitative preference, not an invented numeric minimum. Numeric rating thresholds are inclusive minimums.

Budget and delivery bounds explicitly store `lt` (strictly below) or `lte` (at most). For example, “under ₹300” uses `lt`, “₹300 or less” uses `lte`, and “within 30 minutes” uses `lte`. Ambiguous bounds require clarification rather than an invented value. INR is the supported currency; other currencies must not be silently converted.

The response schema defines the shared JSON shape only. Intent recognition, extraction, filtering, response consistency checks, and recommendation behaviour will be implemented in later commits.
