import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bridgetrend_vision.abo import collect, category_of, download, image_pair, pixel_hash, safe_image_path
from bridgetrend_vision.robustness import (CONDITIONS, classical_features, foreground_crop,
    perturb, rank_queries, summarize, validate_dataset)


def test_category_formats():
    assert category_of({'product_type':[{'value':'LAMP'}]}) == 'LAMP'
    assert category_of({'product_type':'SHOES'}) == 'SHOES'
    assert category_of({}) == ''


def test_pair_skips_self_and_missing():
    row={'main_image_id':'a','other_image_id':['a','missing','b']}
    assert image_pair(row, {'a':{}, 'b':{}}) == ['a','b']
    assert image_pair(row, {'a':{}}) is None


@pytest.mark.parametrize('value',['../bad.jpg','/abs.png','https://evil/a.jpg','ab/../a.jpg','ab/abc.exe'])
def test_image_paths_are_bounded(value):
    with pytest.raises(ValueError):
        safe_image_path(value)


def test_valid_image_path():
    assert safe_image_path('ab/abcdef.jpg') == 'ab/abcdef.jpg'


def test_download_rejects_foreign_host(tmp_path):
    with pytest.raises(ValueError):
        download('https://example.com/x',tmp_path/'x')


def test_download_rejects_changed_cache(tmp_path):
    p=tmp_path/'x'
    p.write_bytes(b'tampered')
    url='https://amazon-berkeley-objects.s3.amazonaws.com/test'
    p.with_name('x.receipt.json').write_text(json.dumps({'url':url,'sha256':'wrong'}))
    with pytest.raises(ValueError,match='Cached data changed'):
        download(url,p)


def test_download_reuses_verified_cache(tmp_path):
    p=tmp_path/'x'
    p.write_bytes(b'fixture')
    url='https://amazon-berkeley-objects.s3.amazonaws.com/test'
    receipt={'url':url,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    p.with_name('x.receipt.json').write_text(json.dumps(receipt))
    assert download(url,p) == receipt


@pytest.mark.parametrize('condition',CONDITIONS)
def test_variants_keep_dimensions_and_source(condition):
    image=Image.fromarray(np.arange(64*48*3,dtype=np.uint8).reshape(48,64,3))
    old=image.tobytes()
    result=perturb(image,condition)
    assert result.size == image.size
    assert result.mode == 'RGB'
    assert image.tobytes() == old


def test_unknown_variant():
    with pytest.raises(ValueError):
        perturb(Image.new('RGB',(64,64)),'unknown')


def test_crop_and_blank_fallback():
    white=Image.new('RGB',(100,100),'white')
    assert foreground_crop(white).size == (100,100)
    white.paste((255,0,0),(40,30,60,70))
    cropped=foreground_crop(white)
    assert cropped.size == (32,52)
    assert foreground_crop(Image.new('RGB',(64,64),'black')).size == (64,64)


def test_classical_is_finite_and_normalized():
    images=[Image.new('RGB',(64,64),color) for color in ['white','black','red','blue']]
    a=classical_features(images)
    assert a.shape == (4,152)
    assert np.isfinite(a).all()
    np.testing.assert_allclose(np.linalg.norm(a,axis=1),1,atol=1e-6)
    assert not np.array_equal(a[2],a[3])


def row(product, image, category='a'):
    return {'product_id':product,'image_id':image,'category':category}


def test_retrieval_different_images_same_product():
    queries=[row('p1','q1'),row('p2','q2')]
    gallery=[row('p1','g1'),row('p2','g2')]
    ranks=rank_queries(np.eye(2),np.eye(2),queries,gallery)
    assert [r['positive_rank'] for r in ranks] == [1,1]
    assert summarize(ranks)['recall_at_1'] == 1


def test_metrics_hand_calculation():
    result=summarize([{'positive_rank':1},{'positive_rank':2},{'positive_rank':11}])
    assert result['recall_at_1'] == pytest.approx(1/3)
    assert result['recall_at_5'] == pytest.approx(2/3)
    assert result['mrr'] == pytest.approx((1+.5+1/11)/3)
    assert result['map_at_10'] == pytest.approx(.5)
    assert result['recall_at_1_wilson95'][0] < 1/3 < result['recall_at_1_wilson95'][1]


def test_oracle_category_changes_candidates():
    q=[row('p1','q1','shoes')]
    g=[row('p1','g1','shoes'),row('p2','g2','chair')]
    assert rank_queries([[1,0]],[[0,1],[1,0]],q,g)[0]['positive_rank'] == 2
    assert rank_queries([[1,0]],[[0,1],[1,0]],q,g,category_filter=True)[0]['positive_rank'] == 1


def test_closed_set_refuses_missing_positive():
    with pytest.raises(ValueError,match='known gallery positive'):
        rank_queries([[1]],[[1]],[row('p','q')],[row('other','g')])


def test_ties_are_stable_by_image_id():
    r=rank_queries([[0,0]],[[0,0],[0,0]],[row('p','q')],[row('p','z'),row('o','a')])
    assert r[0]['positive_rank'] == 2


def make_dataset(tmp_path):
    rows=[]
    for role,color in [('query','red'),('gallery','blue')]:
        p=tmp_path/f'{role}.png'
        Image.new('RGB',(64,64),color).save(p)
        rows.append({**row('p',role),'role':role,'split':'test','image_path':p.name,
                     'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'pixel_sha256':pixel_hash(p)})
    return {'images':rows}


def test_dataset_integrity_and_split(tmp_path):
    data=make_dataset(tmp_path)
    assert len(validate_dataset(data,tmp_path)) == 2
    data['images'][1]['split']='development'
    with pytest.raises(ValueError,match='multiple splits'):
        validate_dataset(data,tmp_path)


def test_dataset_refuses_changed_image(tmp_path):
    data=make_dataset(tmp_path)
    (tmp_path/'query.png').write_bytes(b'bad')
    with pytest.raises(ValueError,match='checksum mismatch'):
        validate_dataset(data,tmp_path)


def test_dataset_refuses_duplicate_pixels(tmp_path):
    data=make_dataset(tmp_path)
    data['images'][1]['pixel_sha256']=data['images'][0]['pixel_sha256']
    with pytest.raises(ValueError,match='Duplicate'):
        validate_dataset(data,tmp_path)


def test_collector_offline_freezes_pairs_and_preserves_market(tmp_path):
    import csv
    import gzip
    import io
    entries = []
    metadata = io.StringIO()
    writer = csv.writer(metadata)
    writer.writerow(['image_id', 'path'])
    for i in range(5):
        ids = [f'im{i}{j}' for j in range(2)]
        entries.append({'item_id':f'p{i}', 'country':'US' if i < 4 else 'CN',
            'domain_name':'amazon.com', 'product_type':'LAMP',
            'main_image_id':ids[0], 'other_image_id':[ids[1]],
            'item_name':[{'language_tag':'zh_CN', 'value':'中文标题'}]})
        for j, image_id in enumerate(ids):
            writer.writerow([image_id, f'ab/{i*2+j:04x}.png'])
    def fetch(url, path, **kwargs):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.name == 'listings_0.json.gz':
            payload = gzip.compress(('\n'.join(json.dumps(e) for e in entries)).encode())
        elif path.name == 'images.csv.gz':
            payload = gzip.compress(metadata.getvalue().encode())
        elif path.suffix == '.png':
            n = int(path.stem, 16)
            out = io.BytesIO()
            Image.new('RGB',(64,64),(n*20,40,80)).save(out,format='PNG')
            payload = out.getvalue()
        else:
            payload = b'offline license fixture'
        path.write_bytes(payload)
        return {'url':url,'sha256':hashlib.sha256(payload).hexdigest(),'retrieved_at':'fixture'}
    data = collect(tmp_path/'data', per_category=4, categories=('LAMP',),fetch=fetch)
    assert data['complete'] and len(data['images']) == 8
    assert {r['market'] for r in data['images']} == {'US'}
    assert {r['product_id'] for r in data['images']} == {'p0','p1','p2','p3'}
    assert len(validate_dataset(data,tmp_path/'data')) == 8
    assert sum(r['split'] == 'development' for r in data['images']) == 2
    with pytest.raises(FileExistsError):
        collect(tmp_path/'data',per_category=4,categories=('LAMP',),fetch=fetch)


def test_collector_refuses_empty_categories(tmp_path):
    with pytest.raises(ValueError):
        collect(tmp_path,categories=())
