# Almouj Valuation Model

A hedonic pricing model for apartments in Almouj (The Wave), Muscat, Oman, deployed as a browser calculator.

**Live tool:** https://umardemir.github.io/almouj/

## What it does

Enter an apartment's size, rooms, floor, location, view and furnishing, and the calculator estimates its asking price. The estimate comes from a multiple linear regression (MLR) fit on the log of asking price, so each feature changes the price by a percentage.

## Data

- 50 public asking-price listings from Wave Homes and Dubizzle, collected September 2025 ([`analysis/data/almouj_data.csv`](analysis/data/almouj_data.csv))
- 10 further listings from OpenSooq, used only for testing ([`analysis/data/new_listings_opensooq.csv`](analysis/data/new_listings_opensooq.csv))
- Asking prices ranged 90,000–300,000 OMR (mean 165,410 OMR)

## Results

| | MLR | PCR |
|---|---|---|
| R² on 10 held-out listings | 0.90 | 0.93 |
| MAE on 10 new OpenSooq listings | **10,902 OMR** | 15,831 OMR |
| RMSE on 10 new OpenSooq listings | 14,445 OMR | 17,567 OMR |

Principal component regression (PCR) scored slightly higher on the held-out split, but MLR was more accurate on the new listings, so the calculator uses MLR.

- **Significant predictors (p < 0.05):** bedrooms, marina view, sea view
- **Strongest correlations with log price:** marina view (r = 0.83), bedrooms (r = 0.80), surface area (r = 0.79)
- Surface area and bedrooms overlap strongly (VIF 8.1 and 9.5), so surface area's separate effect is not significant.

## Reproduce the results

```bash
pip install -r analysis/requirements.txt
python analysis/model.py              # fits MLR and PCR, prints summaries
python analysis/test_new_listings.py  # tests both models on the OpenSooq listings
```

## Limitations

- Small sample (n = 50) from one community
- Asking prices, not sale prices
- Trained on September 2025 prices; not adjusted for later market changes
- Less accurate at the extremes of the size range
- Not a formal appraisal

## Background

Built as an IB Mathematics AA HL Internal Assessment, then extended into a public tool. Methodology reviewed by Prof. Abraham Varghese, PhD (University of Technology and Applied Sciences, Muscat).

## Author

Umar Demir · [github.com/UmarDemir](https://github.com/UmarDemir)

Licensed under MIT.
