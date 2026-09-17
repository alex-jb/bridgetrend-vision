# G1B 40-concept sampling protocol

## Decision

The first claim-bearing acquisition frame contains **40 product concepts**, not
four and not a positive-only list of viral products. It is balanced across eight
category groups and five sampling roles. The registry is ready for collection;
its trend-transfer labels are deliberately not ready.

`configs/g1b_concept_registry.yaml` is the authoritative preregistration. Run:

```bash
PYTHONPATH=src python scripts/check_g1b_concept_registry.py \
  --output-csv data/g1b_acquisition_matrix.csv
```

The validator fails if a concept disappears, a quota changes, a bilingual query
is missing, or an outcome label is inserted before evidence collection. Every
generated acquisition row also carries the source registry's SHA-256 digest.

## Why the frame is balanced

Every category contributes exactly one concept to each role:

| Sampling role | Count | Purpose |
|---|---:|---|
| Emerging candidate | 8 | Tests whether early attention becomes durable cross-market demand |
| Mature anchor | 8 | Verifies that the pipeline can recover already-visible shared concepts |
| Stable control | 8 | Measures false trend alarms on persistent utility products |
| Ambiguity probe | 8 | Tests visually similar but functionally different near-neighbors |
| Seasonal control | 8 | Prevents calendar-driven spikes from being mislabeled as market transfer |

This design does not pre-assign `China-first`, `U.S.-first`, `simultaneous`, or
`non-transfer`. Those are future labels computed from frozen time series under
the same threshold and adjudication policy.

## Registered concepts

| ID | English | 中文 | Group | Role | Wave |
|---|---|---|---|---|---:|
| BT-C001 | Blind-box collectible figure | 盲盒潮玩手办 | Collectibles | Mature anchor | 1 |
| BT-C002 | Plush bag charm | 毛绒包挂 | Collectibles | Emerging | 1 |
| BT-C003 | Trading-card display case | 集换式卡牌展示盒 | Collectibles | Stable | 2 |
| BT-C004 | Capsule-toy figure | 扭蛋公仔 | Collectibles | Ambiguity | 4 |
| BT-C005 | Collectible advent calendar | 收藏玩具圣诞倒数日历 | Collectibles | Seasonal | 3 |
| BT-C006 | Magnetic phone grip stand | 磁吸手机支架握把 | Personal tech | Mature anchor | 1 |
| BT-C007 | Clip-on open-ear earbuds | 耳夹式开放耳机 | Personal tech | Emerging | 1 |
| BT-C008 | Mini thermal label printer | 便携热敏标签机 | Personal tech | Stable | 2 |
| BT-C009 | Magnetic phone wallet | 磁吸手机卡包 | Personal tech | Ambiguity | 4 |
| BT-C010 | Wearable neck fan | 挂脖风扇 | Personal tech | Seasonal | 3 |
| BT-C011 | Sculptural novelty handbag | 雕塑感异形手提包 | Fashion | Emerging | 1 |
| BT-C012 | Crescent shoulder bag | 月牙腋下包 | Fashion | Mature anchor | 2 |
| BT-C013 | Tote organizer insert | 托特包内胆收纳 | Fashion | Stable | 3 |
| BT-C014 | Decorative bag-charm chain | 装饰包挂链 | Fashion | Ambiguity | 4 |
| BT-C015 | Woven beach tote | 编织沙滩托特包 | Fashion | Seasonal | 3 |
| BT-C016 | Heated eyelash curler | 电热睫毛夹 | Beauty tools | Emerging | 1 |
| BT-C017 | Heatless curling rod set | 免加热卷发带套装 | Beauty tools | Mature anchor | 2 |
| BT-C018 | Silicone scalp massager | 硅胶头皮按摩刷 | Beauty tools | Stable | 3 |
| BT-C019 | Facial ice globes | 面部冰球美容工具 | Beauty tools | Ambiguity | 4 |
| BT-C020 | Facial cooling roller | 面部冰敷滚轮 | Beauty tools | Seasonal | 3 |
| BT-C021 | Portable mushroom table lamp | 无线蘑菇台灯 | Home | Emerging | 1 |
| BT-C022 | Modular pegboard organizer | 模块化洞洞板收纳 | Home | Mature anchor | 2 |
| BT-C023 | Stackable drawer storage bins | 可叠加抽屉收纳盒 | Home | Stable | 4 |
| BT-C024 | Sculptural candle | 雕塑造型蜡烛 | Home | Ambiguity | 4 |
| BT-C025 | Holiday projection lamp | 节日投影灯 | Home | Seasonal | 3 |
| BT-C026 | Handled insulated tumbler | 手柄吸管保温杯 | Kitchen | Mature anchor | 1 |
| BT-C027 | Portable blender cup | 便携榨汁杯 | Kitchen | Emerging | 2 |
| BT-C028 | Divided bento lunch box | 分格便当盒 | Kitchen | Stable | 2 |
| BT-C029 | Glass can-shaped straw cup | 易拉罐造型玻璃吸管杯 | Kitchen | Ambiguity | 4 |
| BT-C030 | Countertop shaved-ice machine | 家用刨冰机 | Kitchen | Seasonal | 3 |
| BT-C031 | Compact walking pad | 家用平板走步机 | Fitness | Mature anchor | 1 |
| BT-C032 | Wearable wrist and ankle weights | 腕踝可穿戴负重环 | Fitness | Emerging | 2 |
| BT-C033 | Resistance loop bands | 健身环形阻力带 | Fitness | Stable | 4 |
| BT-C034 | Fitness balance board | 健身平衡板 | Fitness | Ambiguity | 4 |
| BT-C035 | Pickleball paddle set | 匹克球拍套装 | Fitness | Seasonal | 3 |
| BT-C036 | Pet backpack carrier | 宠物背包猫包 | Pet | Mature anchor | 1 |
| BT-C037 | Interactive rolling pet ball | 自动滚动宠物球 | Pet | Emerging | 2 |
| BT-C038 | Elevated pet bowl | 高脚宠物碗 | Pet | Stable | 2 |
| BT-C039 | Donut pet bed | 甜甜圈宠物窝 | Pet | Ambiguity | 4 |
| BT-C040 | Pet cooling mat | 宠物降温垫 | Pet | Seasonal | 3 |

## Minimum acquisition volume

The 40 concepts create 80 concept-market cells. Before expansion, the plan calls
for at least:

- **400 rights-cleared visual assets:** five per concept per market;
- **160 query anchors:** two per concept per market;
- **160 timestamped source series:** two independent series per cell;
- **104 historical weeks** per series where the source permits it; and
- **1,920 prospective source snapshots:** two sources × 12 weeks × 80 cells.

These are minimum coverage targets, not fabricated observations. A row remains
`not_started` until an actual source record, timestamp, rights basis, and hash
exist.

## Acquisition waves

1. **Wave 1 — pipeline-bearing anchors and candidates (10):** establish that
   both markets can supply real visual and temporal evidence.
2. **Wave 2 — breadth (10):** add stable products and additional categories.
3. **Wave 3 — seasonality (10):** collect calendar-sensitive controls without
   using future peaks in training.
4. **Wave 4 — hard negatives (10):** stress-test material, function, scale, and
   context confusion.

The waves control collection effort only. They must never become random
train/test folds. Final modeling uses a global chronological split and keeps
product families isolated.

## Discovery sources are not labels

- [Pinterest Predicts 2026](https://business.pinterest.com/en-gb/pinterest-predicts/)
  and [TikTok's newsroom](https://newsroom.tiktok.com/) are discovery aids. They
  do not prove that any concept transferred from one market to another.
- The user's Baidu Index screenshots suggested vocabulary such as 盲盒 and
  related collectible terms, but screenshots are supporting evidence only.
- [Trendshift](https://trendshift.io/weekly) tracks GitHub repositories. It is
  useful for BridgeTrend's engineering and paper watchlist, not for consumer
  product-demand labels.
- Temporal evaluation follows a global timeline because overlapping train and
  test periods can leak future availability; see
  [Time to Split](https://arxiv.org/abs/2507.16289) and
  [the critical leakage study](https://arxiv.org/abs/2010.11060).

## Expansion rule

Forty is the first frozen sampling frame, not the product ceiling. Expansion to
80–120 concepts is allowed only after Wave 1 reports keyword coverage, source
access, annotation time, and failure rates. New concepts go into registry
version 2 and never retroactively alter the held-out version-1 evaluation set.
