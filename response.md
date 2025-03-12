1. Here's a few things you could do:
  Just retrain one of the models on the tokenizer from the other.
  If both the models tokenize the same way, but just give different token ids for each token, this is actually very good because just need to track with id maps to which.
  But hopefully we never run into this problem...
2. On one hand:
   It seems to perform relatively well compared to other methods.
   It's relatively simple to implement.
   The paper seems to have come out in late 2022 which is also when chatgpt was released - maybe not a coincidence (ie this is what made chatgpt so good)

   On the other hand:
   It's computationally expensive. On the scale of internet chatbots, running a smaller amateur model could cost millions of dollars.
   Its too simple? Maybe there are more involved methods (ex beam search) that may be better.

   Overall, I'd say no, contrastive decoding isn't used in practice. I can always trust corporations to save as much money as possible!
   But of course, I wouldn't be surprised if it was used. It's a nice elegant way to sample :)

Note: I did most of my testing in a notebook, so i just copy and pasted my final code in. 
