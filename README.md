# minLLAMA


**Setup**

```
$ git clone https://github.com/akanyaani/miniLLAMA
$ cd miniLLAMA
$ pip install -r requirement.txt
```

**Usage**

For generating text using trained LLAMA model
```
$ open generate.ipynb
```

Here's how you'd instantiate a LLAMA model:

```
$ from mingptf.model import GPT
$ model_config = GPT.get_default_config()
$ model_config.vocab_size = 50257 # openai's model vocabulary
$ model_config.block_size = 1024  # openai's model block_size (i.e. input context length)
$ model = GPT(model_config)
```

And here's how you'd train it:
```
$ from minllama.model import LLAMA
$ model_config = LLAMA.get_default_config()

$ model_config.model_type = 'LLAMA-micro'
$ model_config.vocab_size = 50257
$ model_config.block_size = 128
$ model = LLAMA(model_config)

$ train_config = get_default_train_config()
$ train_config.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster
$ train_config.max_iters = 2000

$ model.configure_optimizers(train_config)
$ model.fit(train_data, test_data, test_freq=5)
```

TO DO
```
1. Tensorfboard loging.
2. Mixed precison training.
3. Fine-Tuning wrapper.
```

**References:**

* ["facebookresearch/llama"](https://github.com/facebookresearch/llama/tree/main)
* ["Huggingface transformers"](https://github.com/huggingface/transformers)
* ["karpathy/minGPT"](https://github.com/karpathy/minGPT)
* ["akanyaani/gpt-2-tensorflow2.0"](https://github.com/akanyaani/gpt-2-tensorflow2.0)
* ["The Illustrated GPT-2 "](https://jalammar.github.io/illustrated-gpt2/)

**Contribution**

* Your issues and PRs are always welcome.

**Author**

* Abhay Kumar
* Author Email : akanyaani@gmail.com
* Follow me on [Twitter](https://twitter.com/akanyaani)

**License**

* [MIT](https://github.com/akanyaani/minGPTF/blob/master/LICENSE)
