# Tensolr

**Tensolr**, Python tabanlı, minimal bir **Tensor + Computational Graph** motorudur.  
---

## ⚡ Özellikler
- Temel tensor operasyonları: `add`, `sub`, `matmul`, `transpose`
- Otomatik **Graph** oluşturma (`GLOBAL_GRAPH`)
- `forward()` ve `backward()` geçişleri
- Basit **Numba JIT** optimizasyonları
- `pytest` tabanlı test yapısı

---

## Quickstart

### Kurulum
```bash
git clone https://github.com/kullanici/tensolr.git
cd tensolr
pip install -r requirements.txt
```

## Sample

```python
from tensor import Tensolr
from global_graph import GLOBAL_GRAPH

# Tensor oluştur
a = Tensolr([[1, 2], [3, 4]])
b = Tensolr([[5, 6], [7, 8]])

# Toplama işlemi
c = a.add(b)
print("Add result:\n", c.data)

# Matris çarpımı + transpose
d = c.matmul(a.transpose())
print("Matmul result:\n", d.data)

# Graph işlemleri
result = GLOBAL_GRAPH.forward()
GLOBAL_GRAPH.backward(result)

print("Graph nodes:", len(GLOBAL_GRAPH.nodes))
```

“Tensolr, hesaplamayı bir sanat formuna dönüştüren minimalist bir tensor motoru.”
— Doğukan
