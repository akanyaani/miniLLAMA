# minLLAMA

![GitHub Logo](assests/llama.jpg)

****miniLLAMA**** is a concise implementation showcasing the LLAMA and LLAMA2 architectures. Inspired by Andrej Karpathy's MiniGPT, this project aims to elucidate the nuances of LLAMA architecture and its distinctions from GPT.

miniLLAMA provides a compact implementation of LLAMA and LLAMA2, making it easier for enthusiasts and learners to grasp the nuances of these architectures. By drawing parallels with MiniGPT, it simplifies the understanding of LLAMA and facilitates experimentation.

**Setup**

```
$ git clone https://github.com/akanyaani/miniLLAMA
$ cd miniLLAMA
$ pip install -r requirement.txt
```

You can pre-train the model using sample data available in repository or you can download the data using this github repo https://github.com/eukaryote31/openwebtext

Pre-Training model on sample data available in repository
```
$ python pre_process.py --help

Options:
  --data-dir TEXT        training data path  [default: /data/scraped]
  --vocab-size INTEGER   byte pair vocab size  [default: 24512]
  --min-seq-len INTEGER  minimum sequence length  [default: 15]
  --max-seq-len INTEGER  maximum sequence length  [default: 512]
  --help                 Show this message and exit.
  
  
>> python pre_process.py
```

Pre-train LLAMA

```
$ python train.py --help

  
>> python train.py \
  --num-layers=8 \
  --num-heads=12 \
  --hidden-size=768 \
  --max-seq-len=512 \
  --vocab-size=32000 \
  --batch-size=32 \
  --learning-rate=1e-4
```

TO DO
```
1. Loading pre-trained weights by META.
2. Multi-GPU support
3. SFT wrapper.
```

**References:**

* ["facebookresearch/llama"](https://github.com/facebookresearch/llama/tree/main)
* ["Huggingface transformers"](https://github.com/huggingface/transformers)
* ["karpathy/minGPT"](https://github.com/karpathy/minGPT)
* ["akanyaani/gpt-2-tensorflow2.0"](https://github.com/akanyaani/gpt-2-tensorflow2.0)
* ["The Illustrated GPT-2 "](https://jalammar.github.io/illustrated-gpt2/)

**Contribution**

* Contributions and feedback are welcome! Feel free to open issues for suggestions or bug reports, and submit pull requests to improve the project.

**Author**

* Abhay Kumar
* Author Email : akanyaani@gmail.com
* Follow me on [Twitter](https://twitter.com/akanyaani)

**License**

* [MIT](https://github.com/akanyaani/minGPTF/blob/master/LICENSE)
