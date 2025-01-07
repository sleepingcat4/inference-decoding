A 6 year's rage repository. Where I explain and write easy to understandable inference technique and code to excecute them on European supercomputers like JUPITER, JUWELS and Leonardo. 

**What you can find?**
1. Most fastest concoction of inference tips and techniques. That are well documented
2. Code to run them on colab
3. Code to run them on European Pre-exa and Exa-scale supercomputers.

These are tested and well-documented. I am planning to maintain this repository religiously. Filenames are self-explanatory + if you see a flag like `leo` or `juwel` it means it is supercomputer compatible code. I include my slurm scripts as well so have fun. 

**For colab code:** https://colab.research.google.com/drive/17U4lj2YLNH0GdxR9iovBnHdONB4QEh_a?usp=sharing

`KVPress:` Fastest method according to my test on Italian supercomputer Leonardo. On a single A100 64GB card with 16 CPUs. 

`Prompt Caching:` Prompt Caching is relatively good from my tests but it certainly does not come same as par as KVPress. 

`Graph Inference:` One of my favourite methods that uses `torch.compile()` to get fast inference speed while `use_cache` method is turned-on. [Use cache uses KVPress caching] While it shows fast results on Google colab, it was significantly slower on Leonardo. 

`Prompt Lookup:` Again one of my favourite methods and it had the most fastest inference speed of 1.21 seconds. 

I am excited to share more methods in the coming future as I find more. Including batch inference technique using Ray lib and continous batching. 
