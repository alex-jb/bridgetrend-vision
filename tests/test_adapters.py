from pathlib import Path

import pandas as pd
import pytest

from bridgetrend_vision.adapters import (
    import_catalog_csv,
    import_product1m,
    load_category_map,
)


def test_load_category_map_is_case_insensitive(tmp_path: Path):
    path = tmp_path / "map.csv"
    path.write_text("raw_label,category\nLipstick,lipstick\n", encoding="utf-8")

    mapping = load_category_map(path)

    assert mapping["lipstick"] == "lipstick"


def test_import_product1m_accepts_mapped_rows_and_tracks_rejects(tmp_path: Path):
    annotation = tmp_path / "product1m.txt"
    annotation.write_text(
        "item1#####https://a.test/a.jpg#####https://b.test/a.jpg"
        "#####red lipstick#####Lipstick\n"
        "item2#####https://a.test/b.jpg#####https://b.test/b.jpg"
        "#####unknown item#####Unknown\n"
        "broken line\n",
        encoding="utf-8",
    )

    manifest, rejects = import_product1m(
        annotation,
        image_root=tmp_path / "images",
        category_map={"lipstick": "lipstick"},
    )

    assert len(manifest) == 1
    assert manifest.loc[0, "image_id"] == "product1m_item1"
    assert manifest.loc[0, "category"] == "lipstick"
    assert manifest.loc[0, "market"] == "CN"
    assert manifest.loc[0, "source_url"] == "https://a.test/a.jpg"
    assert sorted(rejects["reason"].tolist()) == [
        "malformed_record",
        "unmapped_category",
    ]


def test_import_catalog_csv_maps_columns_and_categories(tmp_path: Path):
    catalog = tmp_path / "catalog.csv"
    pd.DataFrame(
        {
            "sku": ["A-1", "A-2"],
            "photo": ["shoe/a.jpg", "shoe/b.jpg"],
            "type": ["Sneaker", "Unknown"],
            "name": ["Runner", "Mystery"],
        }
    ).to_csv(catalog, index=False)

    manifest, rejects = import_catalog_csv(
        catalog,
        market="US",
        source="approved_catalog",
        image_id_column="sku",
        image_path_column="photo",
        category_column="type",
        title_column="name",
        image_root=tmp_path / "images",
        category_map={"sneaker": "sneakers"},
    )

    assert len(manifest) == 1
    assert manifest.loc[0, "image_id"] == "A-1"
    assert manifest.loc[0, "image_path"] == str(
        tmp_path / "images" / "shoe/a.jpg"
    )
    assert manifest.loc[0, "category"] == "sneakers"
    assert rejects.loc[0, "reason"] == "unmapped_category"


def test_catalog_import_requires_source_columns(tmp_path: Path):
    catalog = tmp_path / "catalog.csv"
    pd.DataFrame({"id": ["a"]}).to_csv(catalog, index=False)

    with pytest.raises(ValueError, match="missing columns"):
        import_catalog_csv(
            catalog,
            market="US",
            source="test",
            image_id_column="id",
            image_path_column="path",
            category_column="category",
        )
