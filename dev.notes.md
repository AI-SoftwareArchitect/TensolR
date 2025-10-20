###### 

###### **passes/ içine ekleyebilirsin:**

* constant\_folding.py → sabitleri önden hesapla.
* operator\_fusion.py → matmul+relu gibi zincirleri tek kernel’e dönüştür.
* dead\_code\_elim.py → kullanılmayan node’ları kaldır.



###### **runners/**

* llvm\_runner.py → MLIR → LLVM → native CPU.



###### **src/tensor.py**

* Complex dizi tipleri, sparsity (seyrek matrisler) desteği.
* Mixed precision (float16, bfloat16) opsiyonu.



###### **src/api/**

* Optimizer genişlet: Adam, RMSProp, Adagrad.
* Layer çeşitleri: Conv2D, BatchNorm, Dropout.
* Loss fonksiyonları: MSE, CrossEntropy.



###### cache/

* JIT cache: bir kere derlenen IR’yi tekrar kullanabilmek.





